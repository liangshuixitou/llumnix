import time

from fastapi import requests


def generate_launch_command(result_filename: str = "",
                            launch_ray_cluster: bool = True,
                            HEAD_NODE_IP: str = "127.0.0.1",
                            ip: str = "127.0.0.1",
                            port: int = 37000,
                            instances_num: int = 1,
                            dispatch_policy: str = "load",
                            migration_backend: str = "gloo",
                            model: str = "facebook/opt-125m",
                            max_model_len: int = 4096,
                            log_instance_info: bool = False,
                            log_request_timestamps: bool = False,
                            request_migration_policy: str = 'SR',
                            max_num_batched_tokens: int = 16000,
                            enable_pd_disagg: bool = False,
                            instance_type: str = "no_constraints",
                            tensor_parallel_size: int = 1,
                            enable_simulator: bool = False):
    command = (
        f"RAY_DEDUP_LOGS=0 HEAD_NODE_IP={HEAD_NODE_IP} HEAD_NODE=1 "
        f"nohup python -u -m llumnix.entrypoints.vllm.api_server "
        f"--host {ip} "
        f"--port {port} "
        f"--initial-instances {instances_num} "
        f"{'--log-filename manager ' if log_instance_info else ''}"
        f"{'--log-instance-info ' if log_instance_info else ''}"
        f"{'--log-request-timestamps ' if log_request_timestamps else ''}"
        f"--enable-migration "
        f"--model {model} "
        f"--worker-use-ray "
        f"--max-model-len {max_model_len} "
        f"--dispatch-policy {dispatch_policy} "
        f"--trust-remote-code "
        f"--request-migration-policy {request_migration_policy} "
        f"--migration-backend {migration_backend} "
        f"--migration-buffer-blocks 32 "
        f"--tensor-parallel-size {tensor_parallel_size} "
        f"--request-output-queue-port {1234+port} "
        f"{'--launch-ray-cluster ' if launch_ray_cluster else ''}"
        f"{'--enable-pd-disagg ' if enable_pd_disagg else ''}"
        f"--instance-type {instance_type} "
        f"--max-num-batched-tokens {max_num_batched_tokens} "
        f"{'--simulator-mode ' if enable_simulator else ''}"
        f"{'--profiling-result-file-path /mnt/model/simulator/Qwen-7B.pkl' if enable_simulator else ''}"
        f"{'> instance_'+result_filename if len(result_filename)> 0 else ''} 2>&1 &"
    )
    return command

def generate_serve_command(result_filename: str = "",
                           ip: str = "127.0.0.1",
                           port: int = 37000,
                           dispatch_policy: str = "load",
                           migration_backend: str = "gloo",
                           model: str = "facebook/opt-125m",
                           max_model_len: int = 4096,
                           log_instance_info: bool = False,
                           log_request_timestamps: bool = True,
                           request_migration_policy: str = 'SR',
                           max_num_batched_tokens: int = 16000,
                           enable_pd_disagg: bool = False,
                           pd_ratio: str = "1:1",
                           enable_simulator: bool = False):
    command = (
        f"RAY_DEDUP_LOGS=0 "
        f"nohup python -u -m llumnix.entrypoints.vllm.serve "
        f"--host {ip} "
        f"--port {port} "
        f"{'--log-filename manager ' if log_instance_info else ''}"
        f"{'--log-instance-info ' if log_instance_info else ''}"
        f"{'--log-request-timestamps ' if log_request_timestamps else ''}"
        f"--enable-migration "
        f"--model {model} "
        f"--worker-use-ray "
        f"--max-model-len {max_model_len} "
        f"--dispatch-policy {dispatch_policy} "
        f"--trust-remote-code "
        f"--request-migration-policy {request_migration_policy} "
        f"--migration-backend {migration_backend} "
        f"--migration-buffer-blocks 32 "
        f"--tensor-parallel-size 1 "
        f"--request-output-queue-port {1234+port} "
        f"--max-num-batched-tokens {max_num_batched_tokens} "
        f"--pd-ratio {pd_ratio} "
        f"--enable-port-increment "
        f"{'--enable-pd-disagg ' if enable_pd_disagg else ''}"
        f"{'--simulator-mode ' if enable_simulator else ''}"
        f"--max-instances 4 "
        f"{'--profiling-result-file-path /mnt/model/simulator/Qwen-7B.pkl' if enable_simulator else ''}"
        f"{'> instance_'+result_filename if len(result_filename)> 0 else ''} 2>&1 &"
    )
    return command

def wait_for_llumnix_service_ready(ip_ports, timeout=120):
    start_time = time.time()
    while True:
        all_ready = True
        for ip_port in ip_ports:
            try:
                response = requests.get(f"http://{ip_port}/is_ready", timeout=5)
                if 'true' not in response.text.lower():
                    all_ready = False
                    break
            except requests.RequestException:
                all_ready = False
                break

        if all_ready:
            return True

        elapsed_time = time.time() - start_time
        if elapsed_time > timeout:
            raise TimeoutError(f"Wait for llumnix service timeout ({timeout}s).")

        time.sleep(1)

def generate_bench_command(ip_ports: str,
                           model: str,
                           num_prompts: int,
                           dataset_type: str,
                           dataset_path: str,
                           qps: float,
                           verbose: bool = False,
                           results_filename: str = "",
                           query_distribution: str = "poisson",
                           coefficient_variation: float = 1.0,
                           priority_ratio: float = 0.0):
    command = (
        f"python -u ./benchmark/benchmark_serving.py "
        f"--ip_ports {ip_ports} "
        f"--backend vLLM "
        f"--tokenizer {model} "
        f"--trust_remote_code "
        f"--log_filename bench_{ip_ports.split(':')[1]} "
        f"--random_prompt_count {num_prompts} "
        f"--dataset_type {dataset_type} "
        f"--dataset_path {dataset_path} "
        f"--qps {qps} "
        f"{'-v ' if verbose else ''} "
        f"--distribution {query_distribution} "
        f"--coefficient_variation {coefficient_variation} "
        f"--priority_ratio {priority_ratio} "
        f"--log_latencies "
        f"--fail_on_response_failure "
        f"{'> bench_'+results_filename if len(results_filename)> 0 else ''}"
    )
    return command

def generate_config_command(config_path: str,
                            ip: str,
                            port: str,
                            model: str = 'facebook/opt-125m',
                            launch_mode: str = 'local',
                            result_filename: str = ""):
    if launch_mode == 'local':
        command = (
            f"HEAD_NODE_IP='127.0.0.1' "
            f"HEAD_NODE=1 "
            f"python -m llumnix.entrypoints.vllm.api_server "
            f"--config-file {config_path} "
            f"--host {ip} "
            f"--port {port} "
            f"--initial-instances 1 "
            f"--model {model} "
            f"--worker-use-ray "
            f"--download-dir /mnt/model "
            f"{'> instance_'+result_filename if len(result_filename)> 0 else ''} 2>&1 &"
        )
    else: # launch_mode == 'global'
        command = (
            f"python -m llumnix.entrypoints.vllm.serve "
            f"--config-file {config_path} "
            f"--host {ip} "
            f"--port {port} "
            f"--max-instances 1 "
            f"--model {model} "
            f"--worker-use-ray "
            f"--download-dir /mnt/model "
            f"{'> instance_'+result_filename if len(result_filename)> 0 else ''} 2>&1 &"
        )
    return command