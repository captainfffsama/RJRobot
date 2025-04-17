import unittest
import time
import sys
import os


current_dir = os.path.dirname(os.path.realpath(__file__))
top_level_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(top_level_dir)

from rjrobot.datasets.lerobot_dataset.lerobot_dataset import LeRobotDataset
import rjrobot.debug_tools as D


class MultiModalRobotDatasetTestCase(unittest.TestCase):
    def setUp(self):
        return super().setUp()

    def test_pass(self):
        dataset = LeRobotDataset(
            repo_id="local_dataset",       # 数据集唯一标识（任意自定义名称）
            root=r"E:\ur_grasp",       # 本地数据集路径
            episodes=None,                 # 如果需要加载所有 episodes，设置为 None
            image_transforms=None,         # 如果需要图像增强，可以传入 torchvision.transforms
            delta_timestamps=None,         # 如果需要时间戳差值信息，可以传入相应字典
            tolerance_s=1e-4,              # 时间同步的容差
            revision=None,                 # 不需要远程版本控制
            force_cache_sync=False,        # 不需要强制同步远程缓存
            download_videos=False          # 不会从远程下载视频
        )
        previous_es=0
        for data in dataset:
            # D.show_img([data["observation.images.0_top"],data["observation.images.1_right"]])
            
            # if data["episode_index"].item()!=previous_es:
            #     print("episode switch")
            #     previous_es=data["episode_index"].item()
            #     breakpoint()

            if data['task_index'].item()==1:
                print("task index changed")
                breakpoint()


if __name__ == "__main__":
    unittest.main()
