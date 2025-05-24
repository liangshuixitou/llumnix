from dataclasses import dataclass
from pydantic import BaseModel
from typing import Any


class APIResponse(BaseModel):
    code: int
    message: str
    data: Any


class InferenceInstanceInfo(BaseModel):
    instance_id: str
    gpu_count: int
    request_count: int
    running_request_count: int
    waiting_request_count: int
    total_gpu_blocks_count: int
    used_gpu_blocks_count: int
    waiting_gpu_blocks_count: int


class BenchmarkRequest(BaseModel):
    qps: float
    num_prompts: int
