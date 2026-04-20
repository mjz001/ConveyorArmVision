# YOLOv8-seg + DeepSort ConveyorArmVision
## 项目简介
本项目聚焦工业背景下的智能分拣车间，旨在实现： 实例分割->动态追踪->智能诊断->边缘设备部署的技术闭环
***
## 项目进度
已实现模块
- 实例分割：使用YOLOv8-seg模型进行本次项目数据的简单微调
- 动态追踪：使用deepsort卡尔曼滤波来实现对多目标的持续追踪
***
## 项目使用说明
- 克隆仓库到本地：git clone https://github.com/mjz001/ConveyorArmVision.git
- 安装环境依赖：pip install -r requirements.txt
- 不带参数运行项目：python main.py
- 带参数运行项目：python main.py --output_dir 自定义输出路径 --model_path 自定义模型路径 --video_path 自定义视频路径 --class_name_yaml 类别映射配置文件路径
