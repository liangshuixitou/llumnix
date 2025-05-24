def generate_bench_command(
    ip_ports: str,
    model: str,
    num_prompts: int,
    dataset_type: str,
    dataset_path: str,
    qps: float,
    verbose: bool = False,
    results_filename: str = "",
    query_distribution: str = "poisson",
    coefficient_variation: float = 1.0,
    priority_ratio: float = 0.0,
):
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
