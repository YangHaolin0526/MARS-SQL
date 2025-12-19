import aiodocker
from aiodocker import Docker
from pathlib import Path
import aiohttp
import os
from aiodocker.docker import _rx_tcp_schemes, _sock_search_paths, _rx_version
import sys
import logging
from logging import getLogger, Logger
from .config import settings

def _get_docker_host():
    docker_host = os.environ.get("DOCKER_HOST", None)
    if docker_host is None:
        if sys.platform == "win32":
            try:
                if Path("\\\\.\\pipe\\docker_engine").exists():
                    docker_host = "npipe:////./pipe/docker_engine"
            except OSError as ex:
                if ex.winerror == 231:  # type: ignore
                    # All pipe instances are busy
                    # but the pipe definitely exists
                    docker_host = "npipe:////./pipe/docker_engine"
                else:
                    raise
        else:
            for sockpath in _sock_search_paths:
                if sockpath.is_socket():
                    docker_host = "unix://" + str(sockpath)
                    break
    return docker_host

def setup_pool_and_host(num_connections: int = 300):
    docker_host = _get_docker_host()
    ssl_context = None
    # Code below is copied from Docker class init in aiodocker
    # the main reason is that docker host resolution is not provided as a helper function
    # so we use it here
    UNIX_PRE = "unix://"
    UNIX_PRE_LEN = len(UNIX_PRE)
    WIN_PRE = "npipe://"
    WIN_PRE_LEN = len(WIN_PRE)
    if _rx_tcp_schemes.search(docker_host):
        if os.environ.get("DOCKER_TLS_VERIFY", "0") == "1":
            if ssl_context is None:
                ssl_context = Docker._docker_machine_ssl_context()
                docker_host = _rx_tcp_schemes.sub("https://", docker_host)
        else:
            ssl_context = None
        connector = aiohttp.TCPConnector(ssl=ssl_context, limit=num_connections)  # type: ignore[arg-type]
        docker_host = docker_host
    elif docker_host.startswith(UNIX_PRE):
        connector = aiohttp.UnixConnector(docker_host[UNIX_PRE_LEN:], limit=num_connections)
        # dummy hostname for URL composition
        docker_host = UNIX_PRE + "localhost"
    elif docker_host.startswith(WIN_PRE):
        connector = aiohttp.NamedPipeConnector(
            docker_host[WIN_PRE_LEN:].replace("/", "\\"), limit=num_connections
        )
        # dummy hostname for URL composition
        docker_host = WIN_PRE + "localhost"
    else:
        raise ValueError("Missing protocol scheme in docker_host.")
    return connector, docker_host


def get_logger(name: str) -> Logger:
    logger = getLogger(name)
    if len(logger.handlers):
        # pre-configured
        return logger
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s',
                                        datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    return logger
