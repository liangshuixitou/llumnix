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

from datetime import datetime
import time
import shutil
import os
import subprocess
import ray
from ray._raylet import PlacementGroupID
from ray._private.utils import hex_to_binary
from ray.util.placement_group import PlacementGroup
from ray.util.state import list_actors, list_placement_groups
import pytest

from llumnix.utils import random_uuid


def ray_start():
    for _ in range(5):
        subprocess.run(["ray", "stop"], check=False, stdout=subprocess.DEVNULL)
        subprocess.run(
            ["ray", "start", "--head", "--port=6379"],
            check=False,
            stdout=subprocess.DEVNULL,
        )
        time.sleep(5.0)
        result = subprocess.run(
            ["ray", "status"], check=False, capture_output=True, text=True
        )
        if result.returncode == 0:
            return
        print("Ray start failed, exception: {}".format(result.stderr.strip()))
        time.sleep(3.0)
    raise Exception("Ray start failed after 5 attempts.")


def ray_stop():
    subprocess.run(["ray", "stop", "--force"], check=False, stdout=subprocess.DEVNULL)


def cleanup_ray_env_func():
    try:
        actor_states = list_actors()
        for actor_state in actor_states:
            try:
                if actor_state["name"] and actor_state["ray_namespace"]:
                    actor_handle = ray.get_actor(
                        actor_state["name"], namespace=actor_state["ray_namespace"]
                    )
                    ray.kill(actor_handle)
            # pylint: disable=bare-except
            except:
                continue
    # pylint: disable=bare-except
    except:
        pass

    try:
        # list_placement_groups cannot take effects.
        pg_states = list_placement_groups()
        for pg_state in pg_states:
            try:
                pg = PlacementGroup(
                    PlacementGroupID(hex_to_binary(pg_state["placement_group_id"]))
                )
                ray.util.remove_placement_group(pg)
            # pylint: disable=bare-except
            except:
                pass
    # pylint: disable=bare-except
    except:
        pass

    time.sleep(1.0)

    alive_actor_states = list_actors(filters=[("state", "=", "ALIVE")])
    if alive_actor_states:
        print(
            "There are still alive actors, alive_actor_states: {}".format(
                alive_actor_states
            )
        )
        try:
            ray.shutdown()
        # pylint: disable=bare-except
        except:
            pass


def pytest_sessionstart(session):
    ray_start()


def pytest_sessionfinish(session):
    ray_stop()


@pytest.fixture
def ray_env():
    ray.init(namespace="llumnix", ignore_reinit_error=True)
    yield
    cleanup_ray_env_func()


def backup_error_log(func_name):
    curr_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    dst_dir = os.path.expanduser(
        f"/home/lhy/data/error_log/{curr_time}_{random_uuid()}"
    )
    os.makedirs(dst_dir, exist_ok=True)

    src_dir = os.getcwd()

    for filename in os.listdir(src_dir):
        if filename.startswith("instance_"):
            src_file = os.path.join(src_dir, filename)
            shutil.copy(src_file, dst_dir)

        elif filename.startswith("bench_"):
            src_file = os.path.join(src_dir, filename)
            shutil.copy(src_file, dst_dir)

    file_path = os.path.join(dst_dir, "test.info")
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(f"{func_name}")

    print(f"Backup error instance log to directory {dst_dir}")


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()

    if report.when == "call" and report.failed:
        func_name = item.name
        backup_error_log(func_name)
