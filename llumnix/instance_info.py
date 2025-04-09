# Copyright (c) 2024, Alibaba Group;
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import ABC, abstractmethod
import copy
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple
import numpy as np

from llumnix.logging.logger import init_logger
from llumnix.llumlet.request import RequestInferenceType

logger = init_logger(__name__)


class InstanceType(str, Enum):
    NO_CONSTRAINTS = "no_constraints"
    PREFILL = "prefill"
    DECODE = "decode"


@dataclass
class InstanceInfo:
    instance_id: str = ""
    instance_type: InstanceType = None

    step_id: int = None
    timestamp: float = None
    num_batched_tokens: int = None
    num_seqs = None
    running_seq_lens: List[int] = field(default_factory=list)
    last_inference_latency: float = None
    inference_type: RequestInferenceType = None

    num_total_gpu_blocks: int = 0
    num_watermark_blocks: int = 0
    num_used_gpu_blocks: int = 0
    num_free_gpu_blocks: int = 0
    gpu_cache_usage: float = 0.0
    num_running_requests: int = 0
    num_waiting_requests: int = 0
    num_killed_requests: int = 0
    num_blocks_first_waiting_request: int = 0
    waiting_time_first_waiting_request: int = 0
    num_blocks_all_waiting_requests: int = 0
    num_blocks_last_running_request: int = 0

    # on-demand init infos
    dispatch_load_metric: float = -np.inf
    migration_load_metric: float = np.inf
    migration_load_metric_after_migrate_in: float = -np.inf
    migration_load_metric_after_migrate_out: float = np.inf

    # lazy init infos
    num_available_gpu_blocks: int = 0
    num_available_gpu_blocks_waiting: int = 0

    # manual init infos
    profiling_data: Tuple[str, int, int, float] = None

    def __post_init__(self) -> None:
        self.num_available_gpu_blocks = (
            self.num_free_gpu_blocks - self.num_watermark_blocks
        )
        self.num_available_gpu_blocks_waiting = (
            self.num_available_gpu_blocks - self.num_blocks_all_waiting_requests
        )


class InstanceLoadCalculator:
    def __init__(
        self, dispatch_load_metric: str, migration_load_metric: str, enable_defrag: bool
    ) -> None:
        logger.info(
            f"dispatch_load_metric: {dispatch_load_metric}, migration_load_metric: {migration_load_metric}, enable_defrag: {enable_defrag}"
        )
        self.dispatch_load_calculator = DispatchLoadComputation(dispatch_load_metric)
        self.migration_load_calculator = MigrationLoadComputation(
            migration_load_metric, enable_defrag
        )

    def compute_instance_load(self, instance_info: InstanceInfo):
        instance_info.dispatch_load_metric = (
            self.dispatch_load_calculator.compute_instance_load(instance_info)
        )
        instance_info.migration_load_metric = (
            self.migration_load_calculator.compute_instance_load(instance_info)
        )
        instance_info.migration_load_metric_after_migrate_out = (
            self.migration_load_calculator.compute_instance_load_after_migrate(
                instance_info, is_migrate_in=False
            )
        )
        instance_info.migration_load_metric_after_migrate_in = (
            self.migration_load_calculator.compute_instance_load_after_migrate(
                instance_info, is_migrate_in=True
            )
        )


class LoadComputationStrategy(ABC):
    def __init__(self, load_metric: str, enable_defrag: bool = False) -> None:
        self.load_metric = load_metric
        self.enable_defrag = enable_defrag

    @abstractmethod
    def compute_instance_load(self, instance_info: InstanceInfo) -> float:
        pass


class DispatchLoadComputation(LoadComputationStrategy):
    def compute_instance_load(self, instance_info: InstanceInfo) -> float:
        instance_load = -np.inf
        if self.load_metric == "usage_ratio":
            instance_load = (
                instance_info.num_used_gpu_blocks
                + instance_info.num_blocks_all_waiting_requests
            ) / instance_info.num_total_gpu_blocks
        elif self.load_metric == "virtual_usage":
            # GPU资源使用率（含等待块）[0,1]
            gpu_usage = (
                instance_info.num_used_gpu_blocks
                + instance_info.num_blocks_all_waiting_requests
            ) / instance_info.num_total_gpu_blocks

            # 等待请求压力（归一化等待时间与可用资源）[0,1]
            if instance_info.num_available_gpu_blocks_waiting > 0:
                # 假设合理的等待时间范围是0-60秒
                normalized_waiting_time = min(
                    instance_info.waiting_time_first_waiting_request / 60.0, 1.0
                )
                waiting_pressure = (
                    instance_info.num_waiting_requests * normalized_waiting_time
                ) / (instance_info.num_available_gpu_blocks_waiting + 1e-6)
            else:
                waiting_pressure = 1.0  # 如果没有可用块，压力最大

            # 运行请求长度压力（考虑序列长度分布）[0,1]
            if instance_info.running_seq_lens:
                avg_running_len = sum(instance_info.running_seq_lens) / len(
                    instance_info.running_seq_lens
                )
                max_running_len = max(instance_info.running_seq_lens)
                # 假设合理的序列长度范围是0-1000
                normalized_avg_len = min(avg_running_len / 1000.0, 1.0)
                normalized_max_len = min(max_running_len / 1000.0, 1.0)
                length_pressure = (
                    normalized_avg_len + 0.2 * normalized_max_len
                ) / 1.2  # 归一化到[0,1]
            else:
                length_pressure = 0

            # 完成请求压力（考虑系统稳定性）[0,1]
            completion_pressure = min(
                instance_info.num_killed_requests
                / (instance_info.num_running_requests + 1e-6),
                1.0,
            )

            # 性能压力（基于profiling数据）[0,1]
            performance_pressure = 0.0
            if instance_info.profiling_data:
                _, num_seqs, total_seq_len, latency = instance_info.profiling_data
                if latency > 0 and num_seqs > 0:
                    # 计算每个token的平均处理时间（毫秒）
                    avg_token_latency = (latency * 1000) / (total_seq_len + 1e-6)
                    # 计算序列密度（每个序列的平均长度）
                    seq_density = total_seq_len / (num_seqs + 1e-6)

                    # 归一化处理时间和序列密度
                    normalized_latency = min(avg_token_latency / 10.0, 1.0)
                    normalized_density = min(seq_density / 1000.0, 1.0)

                    # 性能压力综合考虑归一化后的延迟和序列密度
                    performance_pressure = (
                        normalized_latency * normalized_density
                    ) / 1.0  # 已经是[0,1]范围

            # 综合权重组合（所有指标都在[0,1]范围内）
            instance_load = (
                gpu_usage * 0.30  # GPU使用率权重 [0,1]
                + waiting_pressure * 0.20  # 等待压力权重 [0,1]
                + length_pressure * 0.15  # 长度压力权重 [0,1]
                + completion_pressure * 0.10  # 完成压力权重 [0,1]
                + performance_pressure * 0.15  # 性能压力权重 [0,1]
            )
        elif self.load_metric == "remaining_steps":
            num_requests = (
                instance_info.num_running_requests + instance_info.num_waiting_requests
            )
            num_available_gpu_blocks = (
                instance_info.num_available_gpu_blocks
                - instance_info.num_blocks_all_waiting_requests
            )
            if num_requests == 0:
                return -np.inf
            instance_load = (num_available_gpu_blocks / num_requests) * (-1)
        return instance_load


class MigrationLoadComputation(LoadComputationStrategy):
    def compute_instance_load_after_migrate(
        self, instance_info: InstanceInfo, is_migrate_in: bool
    ) -> float:
        instance_info_after_migrate = copy.deepcopy(instance_info)
        num_blocks_last_running_request = (
            instance_info_after_migrate.num_blocks_last_running_request
        )

        if is_migrate_in:
            instance_info_after_migrate.num_running_requests += 1
            instance_info_after_migrate.num_available_gpu_blocks -= (
                num_blocks_last_running_request
            )
        else:
            instance_info_after_migrate.num_running_requests -= 1
            instance_info_after_migrate.num_available_gpu_blocks += (
                num_blocks_last_running_request
            )

        return self.compute_instance_load(instance_info_after_migrate)

    def compute_instance_load(self, instance_info: InstanceInfo) -> float:
        instance_load = -np.inf
        if self.load_metric == "usage_ratio":
            instance_load = (
                instance_info.num_used_gpu_blocks
                + instance_info.num_blocks_first_waiting_request
            ) / instance_info.num_total_gpu_blocks
        elif self.load_metric == "remaining_steps":
            if not self.enable_defrag:
                num_requests = instance_info.num_running_requests
                num_available_gpu_blocks = instance_info.num_available_gpu_blocks
            else:
                num_requests = instance_info.num_running_requests
                if instance_info.num_waiting_requests != 0:
                    num_requests += 1
                num_available_gpu_blocks = (
                    instance_info.num_available_gpu_blocks
                    - instance_info.num_blocks_first_waiting_request
                )
            if num_requests == 0:
                return -np.inf
            instance_load = (num_available_gpu_blocks / num_requests) * (-1)
        elif self.load_metric == "remaining_tokens":
            instance_load = instance_info.num_available_gpu_blocks
        return instance_load


# TODO(KuilongCui): currently scaling and dispatch use the same load calculator, leave
# it in the future to refine
class ScalingLoadComputation(LoadComputationStrategy):
    def __init__(self, load_metric):
        super().__init__(load_metric)
        self.load_calculator = DispatchLoadComputation(load_metric)

    def compute_instance_load(self, instance_info: InstanceInfo) -> float:
        return self.load_calculator.compute_instance_load(instance_info)
