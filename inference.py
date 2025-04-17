import os
import argparse
from rjrobot.datasets.lerobot_dataset_wrap import RJLeRobotDataset
from rjrobot.rlrobot import RJRobotPolicy
import rjrobot.base_cfg as config_manager


def main(args):
    if os.path.exists(args.cfg):
        config_manager.merge_param(args.cfg)
    else:
        raise ValueError("{} is not exist".format(args.cfg))
    input_dir = args.input_dir
    dataset = RJLeRobotDataset(
        repo_id="local_dataset",  # 数据集唯一标识（任意自定义名称）
        root=input_dir,  # 本地数据集路径
        episodes=None,  # 如果需要加载所有 episodes，设置为 None
        image_transforms=None,  # 如果需要图像增强，可以传入 torchvision.transforms
        delta_timestamps=None,  # 如果需要时间戳差值信息，可以传入相应字典
        tolerance_s=1e-4,  # 时间同步的容差
        revision=None,  # 不需要远程版本控制
        force_cache_sync=False,  # 不需要强制同步远程缓存
        download_videos=False,  # 不会从远程下载视频
    )
    policy_args_dict=config_manager.param["policy_cfg"]
    policy=RJRobotPolicy(policy_args_dict)
    for data in dataset:
        result=policy(data)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process directory paths.")
    parser.add_argument("--cfg", type=str, help="policy config path")
    parser.add_argument("--input_dir", type=str, help="Input directory path",default=r"E:\ur_grasp")
    args = parser.parse_args()
    main(args)
