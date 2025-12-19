from pydantic_settings import BaseSettings
import os
import socket
from copy import deepcopy


def get_memory_in_bytes(memory_str: str) -> int:
    units = {'b': 1, 'k': 1024, 'm': 1024*1024, 'g': 1024*1024*1024}
    memory_str = memory_str.lower()
    if memory_str[-1] in units:
        return int(int(float(memory_str[:-1])) * units[memory_str[-1]])
    return int(memory_str)

class Settings(BaseSettings):
    API_KEY: str = os.environ.get("OPENHANDS_API_KEY", 'sandbox-remote')
    DOCKER_REGISTRY_PREFIX: str = os.environ.get("OPENHANDS_DOCKER_REGISTRY_PREFIX", 'docker.io/xingyaoww/')
    BUILD_CACHE_DIR: str = os.environ.get("OPENHANDS_DOCKER_CACHE", '/tmp/.buildx-cache')
    BUILD_CACHE_MAX_AGE_DAYS: int = 7
    CONTAINER_NAME_PREFIX: str = 'openhands-runtime-'
    PUBLIC_HOST: str = "0.0.0.0"
    PORT: str = "3000"
    USE_HOST_NETWORK: bool = False
    REMOTE_RUNTIME_RESOURCE_FACTOR: int = 1
    REMOTE_RUNTIME_DOCKER_RUNTIME_KWARGS: dict = {
            'CpuPeriod': 100000,  # 100ms
            'CpuQuota': 100000,  # Can use 100ms out of each 100ms period (1 CPU)
            'Memory': get_memory_in_bytes('4G'),  # 4 GB of memory
            'OomKillDisable': False,  # Enable OOM killer
    }
    # Port ranges
    EXECUTION_SERVER_PORT_RANGE: tuple = (30000, 39999)
    VSCODE_PORT_RANGE: tuple = (40000, 49999)
    APP_PORT_RANGE_1: tuple = (50000, 54999)
    APP_PORT_RANGE_2: tuple = (55000, 59999)

    MAX_CONNECTIONS: int = 300

    def get_docker_runtime_kwargs(self, resource_factor: int) -> dict:
        docker_runtime_kwargs = deepcopy(self.REMOTE_RUNTIME_DOCKER_RUNTIME_KWARGS)
        docker_runtime_kwargs['Memory'] = int(resource_factor * docker_runtime_kwargs['Memory'])
        docker_runtime_kwargs['CpuQuota'] = int(resource_factor * docker_runtime_kwargs['CpuQuota'])
        return docker_runtime_kwargs



settings = Settings()
