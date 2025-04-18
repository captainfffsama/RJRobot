# RJ_robot 项目说明

本项目基于机械臂平台，聚焦于具身智能模型框架的开发与实现。项目集成了深度学习、计算机视觉、强化学习与机器人控制等技术，旨在提升机械臂的智能感知、决策与自主操作能力，助力具身智能相关研究与实际应用。

## 主要功能

- 机械臂运动控制与任务执行
- 多模态感知（视觉、传感等）与目标识别
- 自主决策
- 任务规划与安全保障
- 数据采集、管理与分析

## 代码结构

```plaintext
rjrobot/
├── __init__.py
├── base_cfg.py                # 基础配置
├── common.py                  # 通用工具与函数
├── constants.py               # 常量定义
├── debug_tools.py             # 调试工具
├── rlrobot.py                 # 具身智能主策略与推理流程
├── saver.py                   # 模型与数据保存
├── utils.py                   # 工具函数
├── datasets/                  # 数据集相关模块
│   ├── __init__.py
│   ├── lerobot_dataset_wrap.py
│   ├── lerobot_dataset/       # 机械臂数据集实现
│   └── ...
├── models/                    # 模型相关
│   ├── act_experts/           # 行为专家模型
│   ├── act_safe_guard/        # 安全保障模型
│   ├── encoders/              # 编码器
│   ├── expert_tools_models/   # 专家工具模型
│   └── vlm/                   # 多模态大模型
└── ...
```

## 依赖环境

- Python 3.8+
- PyTorch
- OpenCV
- HuggingFace Hub
- 机械臂SDK（根据实际硬件选择）

## 快速开始

1. 克隆代码仓库：
    ```bash
    git clone https://github.com/your-repo/RJ_robot.git
    cd RJ_robot
    ```
2. 安装依赖：
    ```bash
    pip install -r requirements.txt
    ```
3. 配置机械臂SDK与相关参数。
4. 运行示例或主程序：
    ```bash
    python inference.py -cfg your_config.yaml --input_dir your_lerobot_dataset_dir
    ```

## 数据集说明

- 数据集模块支持本地与 HuggingFace Hub 数据集的加载与管理，支持多模态数据（如视频、图像、传感器数据等）。
- 详见 `rjrobot/datasets/lerobot_dataset/` 及相关文档。

## 贡献指南

欢迎提交 Issue 或 Pull Request 参与项目共建。请确保代码规范、注释清晰，并附带必要的说明文档。

## 许可证

本项目采用 MIT 许可证，详见 LICENSE 文件。

