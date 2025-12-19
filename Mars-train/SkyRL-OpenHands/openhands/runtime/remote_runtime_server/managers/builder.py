import asyncio
import os
import time
from typing import Dict, List, Optional

import docker
import aiodocker
from collections import deque

from openhands.core.logger import RollingLogger

from ..config import settings
from ..utils import get_logger

MAX_LOG_LINES = 15
logger = get_logger(__name__)
class BuildManager:
    def __init__(self):
        self.builds: Dict[str, Dict] = {}
        self.cache_dir = settings.BUILD_CACHE_DIR

    def _is_cache_usable(self, cache_dir: str) -> bool:
        if not os.path.exists(cache_dir):
            try:
                os.makedirs(cache_dir, exist_ok=True)
                logger.debug(f'Created cache directory: {cache_dir}')
            except OSError as e:
                logger.debug(f'Failed to create cache directory {cache_dir}: {e}')
                return False

        if not os.access(cache_dir, os.W_OK):
            logger.warning(f'Cache directory {cache_dir} is not writable')
            return False

        self._prune_old_cache_files(cache_dir)
        return True

    def _prune_old_cache_files(self, cache_dir: str):
        try:
            current_time = time.time()
            max_age = settings.BUILD_CACHE_MAX_AGE_DAYS * 24 * 60 * 60

            for root, _, files in os.walk(cache_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    try:
                        file_age = current_time - os.path.getmtime(file_path)
                        if file_age > max_age:
                            os.remove(file_path)
                            logger.debug(f'Removed old cache file: {file_path}')
                    except Exception as e:
                        logger.warning(f'Error processing cache file {file_path}: {e}')
        except Exception as e:
            logger.warning(f'Error during cache pruning: {e}')

    async def build_image(
        self,
        docker_client: aiodocker.Docker,
        build_id: str,
        context_path: str,
        target_image: str,
        tags: List[str],
        platform: Optional[str] = None,
        extra_build_args: Optional[List[str]] = None,
        use_local_cache: bool = False,
    ) -> None:
        logger.info(
            '================ DOCKER BUILD STARTED ================'
        )
        self.builds[build_id] = {'status': 'IN_PROGRESS'}
        history = deque(maxlen=MAX_LOG_LINES)
        assert use_local_cache is False
        try:
            if not os.path.exists(context_path):
                raise ValueError(f'Docker build context path not found: {context_path}')

            buildx_cmd = [
                'docker',
                'buildx',
                'build',
                '--progress=plain',
                f'--tag={target_image}',
                '--load',
            ]

            if use_local_cache and self._is_cache_usable(self.cache_dir):
                buildx_cmd.extend(
                    [
                        f'--cache-from=type=local,src={self.cache_dir}',
                        f'--cache-to=type=local,dest={self.cache_dir},mode=max',
                    ]
                )

            if platform:
                buildx_cmd.append(f'--platform={platform}')

            if extra_build_args:
                buildx_cmd.extend(extra_build_args)

            buildx_cmd.append(context_path)

            process = await asyncio.create_subprocess_exec(
                *buildx_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )

            async def read_stream(stream, history):
                while True:
                    line = await stream.readline()
                    if not line:
                        break
                    decoded_line = line.decode('utf-8').rstrip()
                    history.append(decoded_line)

            stdout_task = asyncio.create_task(read_stream(process.stdout, history=history))

            status = await process.wait()

            if status != 0:
                # Extract the most likely error message from the output
                await stdout_task
                error_message = '\n'.join(history)

                # Look for common Docker error patterns
                for line in reversed(history):
                    if 'ERROR' in line or 'Error' in line:
                        error_message = f'Build error (exit code {status}): {line}'
                        break

                raise RuntimeError(
                    f'Build failed with status {status}. Error details:\n{error_message}'
                )

            image = await docker_client.images.inspect(target_image)
            for tag in tags:
                if tag != target_image:
                    repo, tag_name = tag.split(':')
                    await docker_client.images.tag(name=target_image, repo=repo, tag=tag_name)

            self.builds[build_id] = {
                'status': 'SUCCESS',
                'image': target_image,
                'tags': tags,
            }

        except Exception as e:
            error_message = str(e)
            self.builds[build_id] = {
                'status': 'FAILURE',
                'error': error_message,
                'logs': '\n'.join(history),
            }
            logger.error(
                f'Docker build failed for build ID {build_id}: {error_message}'
            )
            raise
