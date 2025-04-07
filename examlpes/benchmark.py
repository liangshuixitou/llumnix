import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch

from examlpes.command_util import generate_bench_command


BENCH_TEST_TIMEOUT_MINS = 30


device_count = torch.cuda.device_count()
ip = "127.0.1.1"
base_port = 37000
model = "/home/l1hy/data/model/opt-125m"
num_prompts = 50

def run_bench_command(command):
    process = subprocess.Popen(command, shell=True)
    return process

tasks = []
for i in range(device_count):
    bench_command = generate_bench_command(
        ip_ports=f"{ip}:{base_port + i}",
        model=model,
        num_prompts=num_prompts,
        dataset_type="sharegpt",
        dataset_path="/home/l1hy/data/dataset/sharegpt4/sharegpt_gpt4.jsonl",
        qps=5,
        results_filename=f"{base_port + i}.out"
    )
    print(bench_command)
    tasks.append(bench_command)

with ThreadPoolExecutor() as executor:
    future_to_command = {executor.submit(run_bench_command, command): command for command in tasks}

    for future in as_completed(future_to_command):
        try:
            process = future.result()
            process.wait(timeout=60*BENCH_TEST_TIMEOUT_MINS)

            assert process.returncode == 0, "bench_test failed with return code {}.".format(process.returncode)
        # pylint: disable=broad-except
        except subprocess.TimeoutExpired:
            process.kill()
            assert False, "bench_test timed out after {} minutes.".format(BENCH_TEST_TIMEOUT_MINS)

