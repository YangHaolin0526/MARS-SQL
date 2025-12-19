from typing import Dict, List, Optional

from pydantic import BaseModel


class StartRequest(BaseModel):
    image: str
    command: List[str]
    working_dir: str
    environment: Dict[str, str]
    session_id: str
    resource_factor: float
    runtime_class: Optional[str] = None
    enable_gpu: bool = False
    workspace_mount_path: Optional[str] = None
    workspace_mount_path_in_sandbox: Optional[str] = None
    use_host_network: bool = False


class StopRequest(BaseModel):
    runtime_id: str


class RuntimeRequest(BaseModel):
    runtime_id: str
