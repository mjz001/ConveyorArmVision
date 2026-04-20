# YOLOv8-seg + DeepSort ConveyorArmVision
## 项目简介
本项目聚焦工业背景下的智能分拣车间，旨在实现： 实例分割->动态追踪->智能诊断->边缘设备部署的技术闭环
***
## 项目进度
已实现模块
- 实例分割：使用YOLOv8-seg模型进行本次项目数据的简单微调
- 动态追踪：使用deepsort卡尔曼滤波来实现对多目标的持续追踪
***
## 项目文件说明
- config：项目所需要的配置文件，目前只有绘制检测框所需要的类别映射文件
- deep_sort_pytorch：追踪项目的核心模块，deep_sort.py里面包含追踪器的定义和追踪器的维护与更新流程
- models：项目使用到的预训练模型权重文件路径
- weights：针对本项目所检测的纸箱目标所训练的分割模型权重路径
- ultralytics：yolo模型训练的官方工具包，封装了超简便的模型训练，验证和推理的工具方法，根据https://docs.ultralytics.com/zh/的api调用指引，开箱即用
- output：推理追踪的视频流导出路径
- utils：  
1.包含了一些项目刚开始的尝试，detection路径下的源码是检测+opencv传统视觉的方法，因为传送带与目标粘连导致效果不理想，被弃用。  
2.seg路径下则是分割模型对图像的推理代码，可以找几张图像去推理，把结果作为演示的静态部分。代码没有设置参数配置，所以需要手动修改所需要的路径。  
3.train是训练路径，包含训练脚本和训练的配置
4.transfers是手动打标签得到的json文件转换为yolo所需要格式的转换脚本，正确的代码我找不到了，这个还没有修改，等后面找时间改  
- main是追踪的入口程序
- tracker是封装了追踪器实例初始化、对输入追踪器数据进行前后处理，以及启动追踪器的类定义脚本
- sel_model封装了模型加载、前后数据处理、推理
***
## 项目使用说明
- 克隆仓库到本地：git clone https://github.com/mjz001/ConveyorArmVision.git
- 安装环境依赖：pip install -r requirements.txt
- 不带参数运行项目：python main.py
- 带参数运行项目：python main.py --output_dir 自定义输出路径 --model_path 自定义模型路径 --video_path 自定义视频路径 --class_name_yaml 类别映射配置文件路径
