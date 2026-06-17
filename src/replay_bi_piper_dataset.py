import argparse
import time
import sys
import types
import numpy as np


class MockErrors(types.ModuleType):
    pass


errors_module = MockErrors("lerobot.utils.errors")


class DeviceNotConnectedError(ConnectionError):
    pass


class DeviceAlreadyConnectedError(ConnectionError):
    pass


errors_module.DeviceNotConnectedError = DeviceNotConnectedError
errors_module.DeviceAlreadyConnectedError = DeviceAlreadyConnectedError
sys.modules["lerobot.utils.errors"] = errors_module

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.robots.bi_piper.bi_piper import BiPiper
from lerobot.robots.bi_piper.configuration_bi_piper import BiPiperConfig

try:
    from lerobot.utils.constants import ACTION
except ImportError:
    ACTION = "action"


def map_action_name(name):
    # 兼容旧脚本使用的 right_/left_ 命名，以及当前数据集里的 leader_* 命名
    if name.startswith("right_arm_joint_") and name.endswith("_rad"):
        joint_num = name[len("right_arm_joint_"):-len("_rad")]
        return f"right_joint_{joint_num}_pos"

    if name.startswith("left_arm_joint_") and name.endswith("_rad"):
        joint_num = name[len("left_arm_joint_"):-len("_rad")]
        return f"left_joint_{joint_num}_pos"

    if name == "right_gripper_open":
        return "right_gripper_pos"

    if name == "left_gripper_open":
        return "left_gripper_pos"

    if name.startswith("leader_joint") and name.endswith(".pos"):
        core_name = name[:-len(".pos")]
        suffix = core_name[len("leader_joint"):]
        joint_num, sep, side = suffix.partition("_")
        if sep and side in {"left", "right"} and joint_num.isdigit():
            return f"{side}_joint_{joint_num}_pos"

    if name == "leader_gripper_right.pos":
        return "right_gripper_pos"

    if name == "leader_gripper_left.pos":
        return "left_gripper_pos"

    return name


def replay(repo_id, episode_idx, fps, can_left, can_right, velocity, dryrun):
    import lerobot.datasets.utils as utils
    import pathlib
    import pandas as pd

    try:
        from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
        has_dataset_metadata = True
    except ImportError:
        has_dataset_metadata = False

    # Check if repo_id is a local path
    root = None
    repo_path = pathlib.Path(repo_id)
    if repo_path.exists() and repo_path.is_dir():
        root = str(repo_path)
        repo_id = repo_path.name

    # Monkey patch check_version_compatibility to bypass the v2.1 error
    print(f"Loading dataset: {repo_id} (root: {root})")
    try:
        # v0.4.x or earlier doesn't need to monkeypatch everything and is simple
        import lerobot.datasets.utils as utils
        original_check = utils.check_version_compatibility

        def bypass_check(repo, version_to_check, current_version, enforce_breaking_major=True):
            pass

        utils.check_version_compatibility = bypass_check

        original_get_safe_version = utils.get_safe_version

        def bypass_get_safe_version(repo_id, version):
            return version

        utils.get_safe_version = bypass_get_safe_version

        # In v0.4.x, dataset_metadata might exist but be inside another file
        try:
            import lerobot.datasets.lerobot_dataset as lerobot_dataset_module
            if hasattr(lerobot_dataset_module, "check_version_compatibility"):
                original_check_in_dataset = lerobot_dataset_module.check_version_compatibility
                lerobot_dataset_module.check_version_compatibility = bypass_check
        except ImportError:
            pass

        dataset = LeRobotDataset(repo_id, root=root, episodes=[episode_idx], revision="v2.1")
        actions = dataset.select_columns(ACTION)

        utils.check_version_compatibility = original_check
        utils.get_safe_version = original_get_safe_version
        try:
            if hasattr(lerobot_dataset_module, "check_version_compatibility"):
                lerobot_dataset_module.check_version_compatibility = original_check_in_dataset
        except Exception:
            pass
    except Exception as e:
        # Attempt to patch for v3.0 code format
        try:
            # Check if we got the huggingface datasets error with parquet features
            if "Feature type 'List' not found" in str(e):
                print(
                    "\n\n由于你当前的 datasets/pyarrow 等环境版本与你本地存放的 v2.1 格式数据有严重冲突"
                    "（抛出了 Feature type 'List' not found 异常），\n直接在当前环境中强行读取会失败，"
                    "请务必执行以下转换命令先将它转换到标准的 v3.0 格式后运行！\n"
                )
                print(
                    f"python src/lerobot/scripts/convert_dataset_v21_to_v30.py "
                    f"--repo-id={repo_id} --root={root} --push-to-hub=false\n"
                )
                print(
                    "注意：如果执行上面的转换脚本报错，请先更新你的 lerobot 到主分支"
                    "（git checkout main && pip install -e .）后再转换。转换完成后你可以切回你当前需要的版本。"
                )
                return

            # Check if io_utils exists first to fail fast to the conversion instruction
            import importlib.util
            if not importlib.util.find_spec("lerobot.datasets.io_utils"):
                raise ImportError("lerobot.datasets.io_utils not found, likely an older version or unsupported structure")

            import lerobot.datasets.io_utils as io_utils
            original_check = utils.check_version_compatibility

            def bypass_check(repo, version_to_check, current_version, enforce_breaking_major=True):
                pass

            utils.check_version_compatibility = bypass_check

            original_check_metadata = None
            if has_dataset_metadata:
                import lerobot.datasets.dataset_metadata as dataset_metadata
                original_check_metadata = dataset_metadata.check_version_compatibility
                dataset_metadata.check_version_compatibility = bypass_check

            original_load_tasks = io_utils.load_tasks

            def bypass_load_tasks(local_dir):
                if (local_dir / "meta/tasks.jsonl").exists():
                    import jsonlines
                    with jsonlines.open(local_dir / "meta/tasks.jsonl", "r") as reader:
                        tasks_list = list(reader)
                    import pandas as pd
                    tasks_dict = {item["task"]: item["task_index"] for item in sorted(tasks_list, key=lambda x: x["task_index"])}
                    df = pd.DataFrame(list(tasks_dict.values()), index=list(tasks_dict.keys()), columns=["task_index"])
                    df.index.name = "task"
                    return df
                elif (local_dir / "meta/tasks.parquet").exists():
                    return original_load_tasks(local_dir)
                else:
                    import pandas as pd
                    df = pd.DataFrame([0], index=["dummy_task"], columns=["task_index"])
                    df.index.name = "task"
                    return df

            io_utils.load_tasks = bypass_load_tasks

            original_load_subtasks = io_utils.load_subtasks

            def bypass_load_subtasks(local_dir):
                return None

            io_utils.load_subtasks = bypass_load_subtasks

            original_load_episodes = io_utils.load_episodes

            def bypass_load_episodes(local_dir):
                if (local_dir / "meta/episodes.jsonl").exists():
                    import jsonlines
                    with jsonlines.open(local_dir / "meta/episodes.jsonl", "r") as reader:
                        episodes = list(reader)
                    import pandas as pd
                    import datasets
                    df = pd.DataFrame(episodes)
                    return datasets.Dataset.from_pandas(df)
                return original_load_episodes(local_dir)

            io_utils.load_episodes = bypass_load_episodes

            original_load_stats = io_utils.load_stats

            def bypass_load_stats(local_dir):
                if (local_dir / "meta/stats.json").exists():
                    return io_utils.cast_stats_to_numpy(io_utils.load_json(local_dir / "meta/stats.json"))
                return original_load_stats(local_dir)

            io_utils.load_stats = bypass_load_stats

            original_get_safe_version = utils.get_safe_version

            def bypass_get_safe_version(repo_id, version):
                return version

            utils.get_safe_version = bypass_get_safe_version

            original_get_safe_version_meta = None
            if has_dataset_metadata:
                original_get_safe_version_meta = dataset_metadata.get_safe_version
                dataset_metadata.get_safe_version = bypass_get_safe_version

            # Bypass info load to pretend codebase version is v3.0 to skip check_version_compatibility
            original_load_info = io_utils.load_info

            def bypass_load_info(local_dir, file_name="info.json"):
                info = original_load_info(local_dir, file_name)
                if file_name == "info.json" and info:
                    info["codebase_version"] = "v3.0"
                return info

            io_utils.load_info = bypass_load_info

            dataset = LeRobotDataset(repo_id, root=root, episodes=[episode_idx], revision="v2.1")
            actions = dataset.select_columns(ACTION)

            # Restore the original check just in case
            utils.check_version_compatibility = original_check
            if has_dataset_metadata:
                dataset_metadata.check_version_compatibility = original_check_metadata
                dataset_metadata.get_safe_version = original_get_safe_version_meta
            io_utils.load_tasks = original_load_tasks
            io_utils.load_subtasks = original_load_subtasks
            io_utils.load_episodes = original_load_episodes
            io_utils.load_stats = original_load_stats
            utils.get_safe_version = original_get_safe_version
            io_utils.load_info = original_load_info

        except Exception as v3_e:
            import traceback
            traceback.print_exc()
            print("\n\n由于新版 lerobot 不再向后兼容本地加载的 v2.1 数据集，建议通过以下命令先将数据集转换为 v3.0 格式后再运行回放：")
            print(f"python src/lerobot/scripts/convert_dataset_v21_to_v30.py --repo-id={repo_id} --root={root} --push-to-hub=false\n")
            print("注意：如果执行上面的转换脚本报错，请先更新你的 lerobot 到主分支（git checkout main && pip install -e .）后再转换。转换完成后你可以切回你当前需要的版本。")
            return

    if fps is None:
        fps = dataset.fps

    print(f"➡️ 准备进行 replay，回放 episode {episode_idx}，速度 {fps} FPS...")

    robot = None
    if not dryrun:
        # 初始化双臂配置
        print("初始化双臂配置...")
        config = BiPiperConfig(can_left=can_left, can_right=can_right, velocity=velocity)
        robot = BiPiper(config)

        print("尝试连接左右机械臂...")
        robot.connect()
        print("✅ 机械臂连接成功！")
    else:
        print("⚠️ 当前为 Dryrun 模式，仅打印数据，不连接机械臂。")

    # 按照实际数据集的长度读取
    num_frames = dataset.num_frames

    # 记录整体运行时间，用于计算平均帧率
    total_start_time = time.perf_counter()

    try:
        action_names = dataset.features[ACTION]["names"]
        frames_per_sec = max(1, int(fps))

        for idx in range(num_frames):
            start_t = time.perf_counter()

            action_array = actions[idx][ACTION]
            action = {}

            # ---- 标准 14 维双臂动作映射 ----
            for i, name in enumerate(action_names):
                val = float(action_array[i])
                action[map_action_name(name)] = val

            # ---- 夹爪值安全处理 ----
            for k in list(action.keys()):
                if "gripper" in k:
                    val = action[k]
                    if val < 0.0:
                        action[k] = 0.0
                    else:
                        action[k] = val

            for k in list(action.keys()):
                if "gripper" in k:
                    val = action[k]
                    if val < 0.0:
                        action[k] = 0.0
                    else:
                        action[k] = val

            # 每隔一秒（即经过 fps 帧）打印一次设定的关节值
            if idx % frames_per_sec == 0:
                print(f"\n[进度: {idx}/{dataset.num_frames} frames] 当前设定关节值:")
                print(
                    f"right_gripper_pos: {action.get('right_gripper_pos', float('nan')):.4f}  "
                    f"left_gripper_pos: {action.get('left_gripper_pos', float('nan')):.4f}"
                )

            if not dryrun:
                # 原始的 send_action 会在每次下发后同步读取机械臂状态，导致双臂耗时翻倍，最高只能达到 5~10 FPS
                # 为了达到 30 FPS，我们绕过状态读取，直接进行数值转换和底层下发
                action_left = {k.replace("left_", ""): v for k, v in action.items() if k.startswith("left_")}
                action_right = {k.replace("right_", ""): v for k, v in action.items() if k.startswith("right_")}
                missing_left = [k for k in robot.left_robot._motors_ft.keys() if k not in action_left]
                missing_right = [k for k in robot.right_robot._motors_ft.keys() if k not in action_right]
                if missing_left or missing_right:
                    raise KeyError(
                        f"动作字段映射不完整，缺少 left={missing_left} right={missing_right}，"
                        f"dataset action names={action_names}"
                    )

                import threading

                # 处理左臂
                def set_left():
                    arr_l = np.array([action_left[k] for k in robot.left_robot._motors_ft.keys()])
                    arr_l = robot.left_robot.model_joint_transform.input_transform(arr_l)
                    # 转换单位 (standard -> joint) 并下发。连发两次防止丢包。
                    for _ in range(2):
                        robot.left_robot.set_joint_state(arr_l)

                # 处理右臂
                def set_right():
                    arr_r = np.array([action_right[k] for k in robot.right_robot._motors_ft.keys()])
                    arr_r = robot.right_robot.model_joint_transform.input_transform(arr_r)
                    for _ in range(2):
                        robot.right_robot.set_joint_state(arr_r)

                t1 = threading.Thread(target=set_left)
                t2 = threading.Thread(target=set_right)
                t1.start()
                t2.start()
                t1.join()
                t2.join()

            dt = time.perf_counter() - start_t
            sleep_time = max(1.0 / fps - dt, 0.0)
            time.sleep(sleep_time)

        print("✅ 动作指令下发完成！")

        if not dryrun:
            # 等待机械臂平滑运动到位
            print("⏳ 正在运动，请等待让其完全停止...")
            for i in range(3):
                time.sleep(1)
                print(f"   等待中... {3-i}s")

            obs_after = robot.get_observation()

        total_time = time.perf_counter() - total_start_time
        avg_fps = num_frames / total_time if total_time > 0 else 0
        print(f"✅ 回放结束！总耗时: {total_time:.2f} 秒，平均帧率: {avg_fps:.2f} FPS")

    except Exception as e:
        print(f"❌ 运行过程中出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if robot is not None:
            robot.disconnect()
            print("✅ 连接已安全断开。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replay a lerobot dataset on BiPiper")
    parser.add_argument("--repo_id", type=str, required=True, help="Dataset repo ID or local path (e.g. lerobot/aloha_mobile_cabinet)")
    parser.add_argument("--episode", type=int, default=0, help="Episode index to replay")
    parser.add_argument("--fps", type=float, default=None, help="Replay FPS (default: use dataset FPS)")
    parser.add_argument("--can_left", type=str, default="can_left", help="CAN interface for left arm")
    parser.add_argument("--can_right", type=str, default="can_right", help="CAN interface for right arm")
    parser.add_argument("--velocity", type=int, default=80, help="Robot velocity")
    parser.add_argument("--dryrun", action="store_true", help="Dryrun mode: only print joint values, do not control robot")

    args = parser.parse_args()
    replay(args.repo_id, args.episode, args.fps, args.can_left, args.can_right, args.velocity, args.dryrun)

"""
python src/lerobot/scripts/convert_dataset_v21_to_v30.py \
    --repo-id=data/"Franka_Put the sandwich in the basket_cover_bread_2365_723378" \
    --root=/home/agilex/.why/lerobot/data/"Franka_Put the sandwich in the basket_cover_bread_2365_723378" \
    --push-to-hub=false


python /home/agilex/.why/lerobot/src/replay_bi_piper_dataset.py \
    --repo_id /home/agilex/.why/lerobot/data/"Franka_Put the sandwich in the basket_cover_bread_2365_723378" \
    --episode 0 \
    --fps 30
"""
