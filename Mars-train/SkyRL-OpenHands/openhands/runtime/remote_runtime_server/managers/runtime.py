
import uuid
from typing import Dict, Optional, List
import time

import aiohttp
import asyncio
import aiodocker
from aiodocker.types import PortInfo
from aiodocker.containers import DockerContainer
import docker
import requests
import tenacity
from copy import deepcopy
from fastapi import HTTPException

from openhands.runtime.utils import find_available_tcp_port
from openhands.utils.async_utils import call_sync_from_async

from ..config import settings
from ..models import StartRequest
from ..utils  import get_logger
from .builder import BuildManager


logger = get_logger(__name__)
class RuntimeManager:
    def __init__(self):
        # set a large timeout just in case
        # await docker_client = docker.from_env(timeout=300)
        self.active_runtimes: Dict[str, Dict] = {}
        self.active_sessions: Dict[str, str] = {}
        self.build_manager = BuildManager()
        self._allocated_ports_in_flight = set()
        self.sync_docker_client = docker.from_env(timeout=300)
        self._port_lock = asyncio.Lock()

    async def _get_ports_in_use_docker(self, docker_client: aiodocker.Docker) -> set:
        containers = await docker_client.containers.list()
        ports = set()
        for container in containers:
            port_bindings = container._container.get("HostConfig", {}).get("PortBindings", {})
            for _, host_maps in port_bindings.items():
                if host_maps:
                    host_ports = [int(host_map["HostPort"]) for host_map in host_maps]
                    for port in host_ports:
                        ports.add(port)
        return ports

    async def release_ports(self, ports):
        async with self._port_lock:
            for port in ports:
                self._allocated_ports_in_flight.discard(port)

    def _sync_get_ports_in_use(self):
        ret = deepcopy(self._allocated_ports_in_flight)
        containers = self.sync_docker_client.containers.list()
        for container in containers:
            for _, host_maps in container.ports.items():
                if host_maps:
                    host_ports = [host_map["HostPort"] for host_map in host_maps]
                    for port_str in host_ports:
                        ret.add(int(port_str))
        return ret

    def _sync_find_available_port(self, port_range, ports_in_use, max_attempts=5):
        port = port_range[1]
        for _ in range(max_attempts):
            port = find_available_tcp_port(port_range[0], port_range[1])
            if port not in ports_in_use:
                return port
        # If no port is found after max_attempts, return the last tried port
        logger.info("Max attempts reached for port finding")
        return port

    async def _get_ports(self, docker_client: aiodocker.Docker, session_id: str):
        s = time.time()
        async with self._port_lock:
            ports_in_use = await self._get_ports_in_use_docker(docker_client)
            for port in self._allocated_ports_in_flight:
                ports_in_use.add(port)
            def _find_ports_task():
                container_port = self._sync_find_available_port(
                        settings.EXECUTION_SERVER_PORT_RANGE,
                        ports_in_use=ports_in_use
                    )
                vscode_port = self._sync_find_available_port(settings.VSCODE_PORT_RANGE,
                                                            ports_in_use=ports_in_use)
                app_ports = [
                    self._sync_find_available_port(settings.APP_PORT_RANGE_1, ports_in_use=ports_in_use),
                    self._sync_find_available_port(settings.APP_PORT_RANGE_2, ports_in_use=ports_in_use),
                ]
                return container_port, vscode_port, app_ports
            # Run in a different thread so that we don't block the event loop
            # At any point in time, the number of such threads running is 1 because we use a lock
            container_port, vscode_port, app_ports = await call_sync_from_async(_find_ports_task)
            for port in [container_port, vscode_port, *app_ports]:
                self._allocated_ports_in_flight.add(port)
        e = time.time()
        logger.info(f"Port finding took {e-s} seconds for session {session_id}")
        return container_port, vscode_port, app_ports


    async def start_runtime(self, docker_client: aiodocker.Docker, request: StartRequest) -> Dict:
        runtime_id = str(uuid.uuid4())
        container_name = f'{settings.CONTAINER_NAME_PREFIX}{request.session_id}'
        resource_factor = request.resource_factor
        ports = None
        try:
            # Only one `start_runtime` task can find ports at a time
            container_port, vscode_port, app_ports = await self._get_ports(docker_client, request.session_id)
            ports = [container_port, vscode_port, app_ports[0], app_ports[1]]

            network_mode = 'host' if settings.USE_HOST_NETWORK else None

            port_mapping = None
            if not settings.USE_HOST_NETWORK:
                port_mapping = {
                    f'{container_port}/tcp': [{'HostPort': str(container_port)}],
                    f'{vscode_port}/tcp': [{'HostPort': str(vscode_port)}],
                }

                for port in app_ports:
                    port_mapping[f'{port}/tcp'] = [{'HostPort': str(port)}]


            device_requests = []
            if request.enable_gpu:
                device_requests.append(
                    {
                        "Driver": "",
                        "Count": -1,
                        "DeviceIDs": None,
                        "Capabilities": [["gpu"]],
                        "Options": {}
                    }
                )
            docker_runtime_kwargs = settings.get_docker_runtime_kwargs(resource_factor)
            # API host config
            host_config = {
                    'NetworkMode': network_mode,
                    "PortBindings": port_mapping,
                    "DeviceRequests": device_requests,
                    **docker_runtime_kwargs,
            }
            # API doens't take `None` values
            host_config = {k: v for k, v in host_config.items() if v is not None}

            volumes = None
            if request.workspace_mount_path and request.workspace_mount_path_in_sandbox:
                # API format
                volumes = {
                    request.workspace_mount_path_in_sandbox: {}
                }
                # For Binds in HostConfig
                binds = [f"{request.workspace_mount_path}:{request.workspace_mount_path_in_sandbox}:rw"]
                host_config["Binds"] = binds

            environment = {
                'port': str(container_port),
                'PYTHONUNBUFFERED': '1',
                'VSCODE_PORT': str(vscode_port),
                **request.environment,
            }
            # Rewrite environment in the Docker API format
            environment = [f"{k}={v}" for k,v in environment.items()]

            # replace port used in command:
            for i, arg in enumerate(request.command):
                if arg.isdigit() and int(arg) == 88888888:
                    request.command[i] = str(container_port)


            container_config = {
                'Image': request.image,
                'Cmd': request.command,
                'WorkingDir': request.working_dir,
                'Env': environment,
                "ExposedPorts": {k: {} for k in port_mapping.keys()},
                # maybe containers are alwaysin detach mode?
                # 'Detach': True,
                'Volumes': volumes,
                "HostConfig": host_config
            }
            print(container_config)

            # API doesn't take None values
            container_config = {k: v for k, v in container_config.items() if v is not None}

            container = await docker_client.containers.run(config=container_config, name=container_name)

            # NOTE: we keep runtime-info in the older docker python SDK format to be compatible with the OH client
            runtime_info = {
                'container_id': container.id,
                'session_id': request.session_id,
                'status': 'running',
                'restart_count': 0,
                'restart_reasons': [],
                'ports': {
                    'container_port': container_port,
                    'vscode_port': vscode_port,
                    'app_ports': app_ports,
                },
                'workspace': {
                    'mount_path': request.workspace_mount_path,
                    'mount_path_in_sandbox': request.workspace_mount_path_in_sandbox,
                }
                if request.workspace_mount_path
                else None,
                'use_host_network': request.use_host_network,
                'gpu_enabled': request.enable_gpu,
            }

            self.active_runtimes[runtime_id] = runtime_info
            self.active_sessions[request.session_id] = runtime_id

            await self._wait_until_alive(container, runtime_id)

            host = settings.PUBLIC_HOST
            return {
                'runtime_id': runtime_id,
                'url': f'http://{host}:{container_port}',
                'work_hosts': {f'http://{host}:{port}': port for port in app_ports},
                'session_api_key': str(uuid.uuid4()),
            }

        except Exception as e:
            logger.error(f'Failed to start runtime: {str(e)}')
            # for port in ([container_port, vscode_port] + app_ports):
            #     await self.release_port(port)
            if runtime_id in self.active_runtimes:
                await self.stop_runtime(docker_client, runtime_id, remove=True)
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            if ports:
                await self.release_ports(ports)

    @tenacity.retry(
        stop=tenacity.stop_after_delay(120),
        retry=tenacity.retry_if_exception_type(
            (ConnectionError, requests.exceptions.ConnectionError)
        ),
        wait=tenacity.wait_fixed(2),
    )
    async def _wait_until_alive(self, container: DockerContainer, runtime_id: str):
        # print('#' * 30)
        status = await self._get_container_status(container)
        if status == "exited":
            raise HTTPException(
                status_code=503, detail=f'Container {runtime_id} has exited.'
            )

        runtime_info = self.active_runtimes[runtime_id]
        port = runtime_info['ports']['container_port']
        host = settings.PUBLIC_HOST
        url = f'http://{host}:{port}/alive'

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=2) as response:
                    if response.status != 200:
                        raise ConnectionError('Container not ready')
        except:
            raise ConnectionError('Container not responding')

    async def stop_runtime(self, docker_client: aiodocker.Docker, runtime_id: str, remove: bool = True):
        """Stop and optionally remove a runtime container."""
        if runtime_id not in self.active_runtimes:
            raise HTTPException(status_code=404, detail='Runtime not found')

        try:
            runtime_info = self.active_runtimes[runtime_id]
            container = await docker_client.containers.get(runtime_info['container_id'])

            # Stop the container
            await container.stop()

            if remove:
                await container.delete()
                # Clean up runtime records
                session_id = runtime_info['session_id']
                del self.active_runtimes[runtime_id]
                del self.active_sessions[session_id]
            else:
                runtime_info['status'] = 'stopped'

        except (docker.errors.NotFound, aiodocker.exceptions.DockerError):
            # Container already removed, clean up records
            session_id = self.active_runtimes[runtime_id]['session_id']
            del self.active_runtimes[runtime_id]
            del self.active_sessions[session_id]
        except Exception as e:
            logger.error(f'Failed to stop runtime: {str(e)}')
            raise HTTPException(
                status_code=500, detail=f'Failed to stop runtime: {str(e)}'
            )

    async def pause_runtime(self, docker_client: aiodocker.Docker, runtime_id: str):
        """Pause a running runtime container."""
        if runtime_id not in self.active_runtimes:
            raise HTTPException(status_code=404, detail='Runtime not found')

        try:
            runtime_info = self.active_runtimes[runtime_id]
            container = await docker_client.containers.get(runtime_info['container_id'])

            # First, ensure environment variables are properly persisted
            # This matches the local implementation's behavior
            try:
                await self._persist_environment(container)
            except Exception as e:
                logger.warning(f'Failed to persist environment variables: {str(e)}')

            # Stop the container but don't remove it
            await container.stop()
            runtime_info['status'] = 'paused'

        except docker.errors.NotFound:
            raise HTTPException(
                status_code=404, detail='Container not found. It may have been removed.'
            )
        except Exception as e:
            logger.error(f'Failed to pause runtime: {str(e)}')
            raise HTTPException(
                status_code=500, detail=f'Failed to pause runtime: {str(e)}'
            )

    async def resume_runtime(self, docker_client: aiodocker.Docker, runtime_id: str):
        """Resume a paused runtime container."""
        if runtime_id not in self.active_runtimes:
            raise HTTPException(status_code=404, detail='Runtime not found')

        try:
            runtime_info = self.active_runtimes[runtime_id]
            container = await docker_client.containers.get(runtime_info['container_id'])

            # Start the container
            await container.start()
            runtime_info['status'] = 'running'

            # Wait for container to be ready
            await self._wait_until_alive(container, runtime_id)

        except docker.errors.NotFound:
            raise HTTPException(
                status_code=404, detail='Container not found. It may have been removed.'
            )
        except Exception as e:
            logger.error(f'Failed to resume runtime: {str(e)}')
            raise HTTPException(
                status_code=500, detail=f'Failed to resume runtime: {str(e)}'
            )

    async def get_runtime_status(self, docker_client: aiodocker.Docker, runtime_id: str) -> Dict:
        """Get detailed runtime status information."""
        if runtime_id not in self.active_runtimes:
            return {'runtime_id': runtime_id, 'pod_status': 'not found'}

        try:
            runtime_info = self.active_runtimes[runtime_id]
            container: DockerContainer = await docker_client.containers.get(runtime_info['container_id'])
            # container.reload()  # Refresh container information

            # Get container stats
            stats_list = await container.stats(stream=False)
            stats = stats_list[-1]

            # Calculate memory usage
            memory_stats = stats.get('memory_stats', {})
            memory_usage = memory_stats.get('usage', 0)
            memory_limit = memory_stats.get('limit', 1)
            memory_percent = (memory_usage / memory_limit) * 100

            # Calculate CPU usage
            cpu_stats = stats.get('cpu_stats', {})
            precpu_stats = stats.get('precpu_stats', {})
            cpu_delta = cpu_stats.get('cpu_usage', {}).get(
                'total_usage', 0
            ) - precpu_stats.get('cpu_usage', {}).get('total_usage', 0)
            system_delta = cpu_stats.get('system_cpu_usage', 0) - precpu_stats.get(
                'system_cpu_usage', 0
            )
            cpu_percent = 0.0
            if system_delta > 0 and cpu_delta > 0:
                cpu_percent = (
                    (cpu_delta / system_delta) * 100.0 * cpu_stats.get('online_cpus', 1)
                )

            status_info = {
                'runtime_id': runtime_id,
                'pod_status': await self._get_container_status(container),
                'restart_count': runtime_info['restart_count'],
                'restart_reasons': runtime_info['restart_reasons'],
                'session_id': runtime_info['session_id'],
                'ports': runtime_info['ports'],
                'resources': {
                    'memory_usage_bytes': memory_usage,
                    'memory_limit_bytes': memory_limit,
                    'memory_percent': round(memory_percent, 2),
                    'cpu_percent': round(cpu_percent, 2),
                },
                'network': {
                    'networks': container['NetworkSettings']['Networks'],
                    'ports': container['NetworkSettings']['Ports'],
                },
                'created_at': container['Created'],
                'started_at': container['State']['StartedAt'],
            }

            # Add GPU information if enabled
            if runtime_info.get('gpu_enabled'):
                gpu_info = await self._get_gpu_info(container)
                if gpu_info:
                    status_info['resources']['gpu'] = gpu_info

            # Add health check information
            status_info['health_check'] = await self._check_container_health(
                container, runtime_info
            )

            return status_info

        except docker.errors.NotFound:
            return {'runtime_id': runtime_id, 'pod_status': 'not found'}
        except Exception as e:
            logger.error(f'Failed to get runtime status: {str(e)}')
            raise HTTPException(
                status_code=500, detail=f'Failed to get runtime status: {str(e)}'
            )

    async def _persist_environment(self, container: DockerContainer) -> None:
        """Persist environment variables to container's .bashrc file."""
        try:
            # Check if .bashrc exists and create it if it doesn't
            await container.exec('touch /root/.bashrc')

            # Get current environment variables
            env_result = await container.exec('env')
            if env_result.exit_code == 0:
                env_vars = env_result.output.decode().strip().split('\n')

                # Prepare environment variable export commands
                export_commands = []
                for env_var in env_vars:
                    if '=' in env_var:
                        key, value = env_var.split('=', 1)
                        # Escape special characters in the value
                        value = value.replace('"', '\\"')
                        export_commands.append(f'export {key}="{value}"')

                # Write to .bashrc
                bashrc_content = '\n'.join(export_commands)
                await container.exec(
                    f'bash -c \'echo "{bashrc_content}" > /root/.bashrc\''
                )
        except Exception as e:
            logger.warning(f'Failed to persist environment variables: {str(e)}')
            raise

    async def _get_gpu_info(self, container: DockerContainer) -> Optional[Dict]:
        """Get GPU information from container if available."""
        try:
            exec_result = await container.exec(
                'nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits'
            )
            if exec_result.exit_code == 0:
                gpu_stats = exec_result.output.decode().strip().split(',')
                return {
                    'utilization_percent': float(gpu_stats[0]),
                    'memory_used_mb': float(gpu_stats[1]),
                    'memory_total_mb': float(gpu_stats[2]),
                }
        except Exception as e:
            logger.warning(f'Failed to get GPU info: {str(e)}')
        return None

    async def _check_container_health(self, container: DockerContainer, runtime_info: Dict) -> Dict:
        """Check container health status including API endpoint responses."""
        health_info = {
            'container_status': await self._get_container_status(container),
            'api_responsive': False,
            'vscode_responsive': False
            if runtime_info['ports'].get('vscode_port')
            else None,
            'app_ports_responsive': {},
        }

        # Check main API endpoint
        try:
            async with aiohttp.ClientSession() as session:
                url = f"http://{settings.PUBLIC_HOST}:{runtime_info['ports']['container_port']}/alive"
                async with session.get(url, timeout=2) as response:
                    health_info['api_responsive'] = response.status == 200
        except:
            pass

        # Check VSCode endpoint if enabled
        if runtime_info['ports'].get('vscode_port'):
            try:
                async with aiohttp.ClientSession() as session:
                    url = f"http://{settings.PUBLIC_HOST}:{runtime_info['ports']['vscode_port']}"
                    async with session.get(url, timeout=2) as response:
                        health_info['vscode_responsive'] = response.status == 200
            except:
                pass

        # Check app ports
        for port in runtime_info['ports'].get('app_ports', []):
            try:
                async with aiohttp.ClientSession() as session:
                    url = f'http://{settings.PUBLIC_HOST}:{port}'
                    async with session.get(url, timeout=2) as response:
                        health_info['app_ports_responsive'][port] = (
                            response.status == 200
                        )
            except:
                health_info['app_ports_responsive'][port] = False

        return health_info

    @staticmethod
    async def _get_container_status(container: DockerContainer):
        container_info = await container.show()
        return container_info.get("State", {}).get("Status", "exited")
