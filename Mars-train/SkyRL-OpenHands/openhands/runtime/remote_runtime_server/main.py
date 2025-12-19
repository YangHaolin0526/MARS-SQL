import argparse
import base64
import os
import shutil
import tarfile
import tempfile
import uuid
from typing import Optional
import time

import docker
import asyncio
import aiodocker
import aiohttp
from fastapi import (
    BackgroundTasks,
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    UploadFile,
    Request,
    Depends
)
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .managers.runtime import RuntimeManager
from .models import (
    RuntimeRequest,
    StartRequest,
    StopRequest,
)
from .utils import setup_pool_and_host, get_logger

_task_lock = asyncio.Lock()
active_concurrent_tasks = 0
MAX_CONCURRENT_TASKS = 16

connector, host = None, None
session = None

SIMPLE_URLS = ["image_exists", "sessions", "build", "runtime/", "registry_prefix"]
logger = get_logger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    global connector, host, session
    if connector is None or host is None or session is None:
        connector, host = setup_pool_and_host(num_connections=settings.MAX_CONNECTIONS)
        session = aiohttp.ClientSession(connector=connector)
    yield
    await session.close()


app = FastAPI(title='Remote Runtime Server', lifespan=lifespan)


# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    url = request.url.path
    is_simple_url = any([val in url for val in SIMPLE_URLS])
    if is_simple_url:
        query_params = dict(request.query_params)
        logger.info(f"Processed {request.method} {url} {query_params} in {process_time:.4f}s - Status: {response.status_code}")
    else:
        logger.info(f"Processed {request.method} {url} in {process_time:.4f}s - Status: {response.status_code}")
    return response

# Initialize runtime manager
runtime_manager = RuntimeManager()

async def get_docker_client():
    global connector, host, session
    if connector is None or host is None or session is None:
        connector, host = setup_pool_and_host(num_connections=settings.MAX_CONNECTIONS)
        session = aiohttp.ClientSession(connector=connector)
    # Create client with default settings
    # default client timeout is 300
    docker_client = aiodocker.Docker(url=host, session=session, connector=connector)
    try:
        yield docker_client
    finally:
        # we don't use .close because it closes the session as well
        await docker_client.events.stop()

# Authentication middleware
async def verify_api_key(api_key: str = Header(None, alias='X-API-Key')):
    if not api_key or api_key != settings.API_KEY:
        raise HTTPException(status_code=401, detail='Invalid API key')
    return api_key


@app.get('/sessions/{sid}')
async def get_session(
    sid: str,
    api_key: str = Header(None, alias='X-API-Key'),
    docker_client: aiodocker.Docker = Depends(get_docker_client)
):
    """Get session information and associated runtime status."""
    await verify_api_key(api_key)

    runtime_id = runtime_manager.active_sessions.get(sid)
    if not runtime_id:
        raise HTTPException(status_code=404, detail='Session not found')

    runtime_info = runtime_manager.active_runtimes[runtime_id]
    container = await docker_client.containers.get(runtime_info['container_id'])

    host = settings.PUBLIC_HOST
    return {
        'status': await runtime_manager._get_container_status(container),
        'runtime_id': runtime_id,
        'url': f"http://{host}:{runtime_info['ports']['container_port']}",
        'work_hosts': {
            f'http://{host}:{port}': port
            for port in runtime_info['ports'].get('app_ports', [])
        },
    }


@app.post('/start')
async def start_runtime(
    request: StartRequest,
    api_key: str = Header(None, alias='X-API-Key'),
    docker_client: aiodocker.Docker = Depends(get_docker_client)
):
    """Start a new runtime container."""
    await verify_api_key(api_key)
    return await runtime_manager.start_runtime(docker_client, request)


@app.post('/stop')
async def stop_runtime(
    request: StopRequest,
    remove: bool = True,
    api_key: str = Header(None, alias='X-API-Key'),
    background_tasks: BackgroundTasks = BackgroundTasks(),
):
    """Stop and optionally remove a runtime container."""
    await verify_api_key(api_key)
    if request.runtime_id not in runtime_manager.active_runtimes:
        raise HTTPException(status_code=404, detail='Runtime not found')


    async def stop_runtime_task():
        global active_concurrent_tasks
        async with _task_lock:
            active_concurrent_tasks += 1
        docker_client = aiodocker.Docker()
        try:
            await runtime_manager.stop_runtime(docker_client, request.runtime_id, remove)
        finally:
            await docker_client.close()
            async with _task_lock:
                active_concurrent_tasks -= 1

    if active_concurrent_tasks < MAX_CONCURRENT_TASKS:
        background_tasks.add_task(stop_runtime_task)
    else:
        logger.info(f"Max concurrent tasks reached, stopping runtime right away {request.runtime_id}")
        await stop_runtime_task()
    return {'status': 'success'}


@app.post('/pause')
async def pause_runtime(
    request: RuntimeRequest,
    api_key: str = Header(None, alias='X-API-Key'),
    docker_client: aiodocker.Docker = Depends(get_docker_client)
):
    """Pause a running runtime container."""
    await verify_api_key(api_key)
    await runtime_manager.pause_runtime(docker_client, request.runtime_id)
    return {'status': 'success'}


@app.post('/resume')
async def resume_runtime(
    request: RuntimeRequest,
    api_key: str = Header(None, alias='X-API-Key'),
    docker_client: aiodocker.Docker = Depends(get_docker_client)
):
    """Resume a paused runtime container."""
    await verify_api_key(api_key)
    await runtime_manager.resume_runtime(docker_client, request.runtime_id)
    return {'status': 'success'}


@app.get('/runtime/{runtime_id}')
async def get_runtime_status(
    runtime_id: str,
    api_key: str = Header(None, alias='X-API-Key'),
    docker_client: aiodocker.Docker = Depends(get_docker_client)
):
    """Get detailed runtime status information."""
    await verify_api_key(api_key)
    return await runtime_manager.get_runtime_status(docker_client, runtime_id)


@app.get('/registry_prefix')
async def get_registry_prefix(api_key: str = Header(None, alias='X-API-Key')):
    """Get the registry prefix for runtime images."""
    await verify_api_key(api_key)
    return {'registry_prefix': settings.DOCKER_REGISTRY_PREFIX}


@app.get('/image_exists')
async def check_image_exists(
    image: str,
    api_key: str = Header(None, alias='X-API-Key'),
    docker_client: aiodocker.Docker = Depends(get_docker_client)
):
    """Check if an image exists in the registry."""
    await verify_api_key(api_key)
    try:
        image_info = await docker_client.images.inspect(image)
        return {
            'exists': True,
            'image': {
                'upload_time': image_info['Created'],
                'image_size_bytes': image_info['Size'],
            },
            "url": f"http://{settings.PUBLIC_HOST}:{settings.PORT}",
        }
    except (docker.errors.ImageNotFound, aiodocker.exceptions.DockerError):
        return {'exists': False,  "url": f"http://{settings.PUBLIC_HOST}:{settings.PORT}"}


@app.post('/build')
async def build_image(
    context: UploadFile = File(...),
    target_image: str = Form(...),
    tags: str = Form(''),
    platform: Optional[str] = Form(None),
    extra_build_args: Optional[str] = Form(None),
    use_cache: bool = Form(False),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    api_key: str = Header(None, alias='X-API-Key'),
):
    """Build a container image."""
    await verify_api_key(api_key)

    build_id = str(uuid.uuid4())

    # Parse form data
    tag_list = [tag.strip() for tag in tags.split(',') if tag.strip()] if tags else []
    extra_args_list = (
        [arg.strip() for arg in extra_build_args.split(',') if arg.strip()]
        if extra_build_args
        else None
    )

    # Create a persistent temporary directory (don't use context manager)
    temp_dir = tempfile.mkdtemp()

    try:
        # Read the uploaded file
        context_data = await context.read()

        # If the content is base64 encoded (as per client code)
        try:
            # Try to decode if it's base64
            context_bytes = base64.b64decode(context_data)
        except:
            # If not base64, use as is
            context_bytes = context_data

        tar_path = os.path.join(temp_dir, 'context.tar.gz')

        with open(tar_path, 'wb') as f:
            f.write(context_bytes)

        # Extract tar file
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(temp_dir)

        # Add target_image to tags if not already there
        if target_image not in tag_list:
            all_tags = [target_image] + tag_list
        else:
            all_tags = tag_list

        # Define a cleanup function to remove the temp directory after build completes
        async def build_and_cleanup():
            try:
                # Create a new Docker client for the background task
                bg_client = aiodocker.Docker()
                try:
                    await runtime_manager.build_manager.build_image(
                        docker_client=bg_client,
                        build_id=build_id,
                        context_path=temp_dir,
                        target_image=target_image,
                        tags=all_tags,
                        platform=platform,
                        extra_build_args=extra_args_list,
                        use_local_cache=use_cache,
                    )
                finally:
                    await bg_client.close()
            finally:
                # Clean up the temp directory after the build is done (success or failure)
                try:
                    shutil.rmtree(temp_dir)
                except Exception as cleanup_error:
                    logger.warning(
                        f'Failed to clean up temp directory {temp_dir}: {cleanup_error}'
                    )

        # Start build process in background with cleanup
        background_tasks.add_task(build_and_cleanup)

        return {'build_id': build_id, "url": f"http://{settings.PUBLIC_HOST}:{settings.PORT}"}

    except Exception as e:
        # Clean up on error in the setup phase
        try:
            shutil.rmtree(temp_dir)
        except:
            pass

        logger.error(f'Build context error: {str(e)}')
        raise HTTPException(status_code=400, detail=f'Invalid build context: {str(e)}')


@app.get('/build_status')
async def get_build_status(
    build_id: str, api_key: str = Header(None, alias='X-API-Key')
):
    """Get the status of an ongoing build."""
    await verify_api_key(api_key)

    if build_id not in runtime_manager.build_manager.builds:
        raise HTTPException(status_code=404, detail='Build not found')

    build_info = runtime_manager.build_manager.builds[build_id]

    # Handle rate limiting
    if (
        build_info.get('status') == 'FAILURE'
        and 'rate limit' in build_info.get('error', '').lower()
    ):
        raise HTTPException(
            status_code=429, detail='Build was rate limited. Please try again later.'
        )

    if build_info.get('status') == 'SUCCESS':
        print('s' * 20)

    return build_info


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Remote Runtime Server')
    parser.add_argument(
        '--host', type=str, default='0.0.0.0', help='Host to bind the server to'
    )
    parser.add_argument(
        '--port', type=int, default=8000, help='Port to bind the server to'
    )
    parser.add_argument(
        '--log-level',
        type=str,
        default='info',
        choices=['debug', 'info', 'warning', 'error', 'critical'],
        help='Logging level',
    )
    return parser.parse_args()


if __name__ == '__main__':
    import uvicorn
    from uvicorn.config import LOGGING_CONFIG

    args = parse_args()
    settings.PUBLIC_HOST = args.host
    if settings.PUBLIC_HOST == '0.0.0.0':
        settings.PUBLIC_HOST = 'localhost'
    settings.PORT = args.port
    LOGGING_CONFIG["formatters"]["default"]["fmt"] = "%(asctime)s [%(name)s] %(levelprefix)s %(message)s"
    LOGGING_CONFIG["formatters"]["access"]["fmt"] = "%(asctime)s [%(name)s] %(levelprefix)s %(client_addr)s - \"%(request_line)s\" %(status_code)s"

    uvicorn.run(app, host='0.0.0.0', port=args.port, log_level=args.log_level)
