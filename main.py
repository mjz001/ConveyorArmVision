import os

#允许windows重复加载dll加速工具包
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
#omp设置并行计算的线程数
os.environ["OMP_NUM_THREADS"] = "1"
#mkl数学计算加速库内部的线程数
os.environ["MKL_NUM_THREADS"] = "1"
import cv2
from seg_model import SegModel
from tracker import DeepSORTTracker
import yaml
from pathlib import Path
import argparse

    


#获取类名映射
def get_class_names(yaml_path="config/class_name.yaml"):
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    return data.get('class_names', None)

def main(args):
    # ==================  初始化 ==================
    class_names = get_class_names(args.class_name_yaml)
    
    print(f"加载模型: {args.yolo_weights}")
    try:
        seg_model = SegModel(model_path=args.yolo_weights, class_names=class_names)
    except Exception as e:
        print(f"加载模型失败: {e}")
        return
    print(f"加载模型成功")
    tracker = DeepSORTTracker()

    # ================== 打开视频流 ==================
    video_path = args.video_path  # 填0为摄像头，也可以填视频路径   
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("视频打开失败")
        return

    # ================== 获取视频信息（保存用） ==================
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # ==================  初始化视频保存 ==================
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 编码格式
    output_video_path = Path(args.output_dir) / "output_tracking.mp4"
    out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))

    # ==================  循环读取帧 ==================
    try:
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 模型推理
            bboxes = seg_model.detect(frame)
            
            # 追踪器更新
            tracked_bboxes = tracker.update(bboxes, frame)
            
            # 匹配推理结果和追踪结果，过滤轮廓
            contour_list, cls_id = seg_model.match_contours(bboxes, tracked_bboxes)

            # 绘制追踪结果
            frame = seg_model.draw_tracked_boxes(frame, tracked_bboxes, contour_list, cls_id)

            # 保存每一帧 
            out.write(frame)

            # 显示
            cv2.imshow("Tracking", frame)
            if cv2.waitKey(10) == 27:
                break
    except Exception as e:
        print(f"error: {e}")
        # 释放资源
        # 读取视频必须释放
        cap.release()
        # 保存视频必须释放
        out.release()  
        cv2.destroyAllWindows()
        exit(0)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo_weights', type=str, default='models/yolov8n-seg.pt', help='model.pt path')
    parser.add_argument('--video_path',type=str,default='test.mp4',help='path to input video')
    parser.add_argument('--class_name_yaml', type=str, default='config/class_name.yaml', help='path to class name yaml')
    parser.add_argument('--output_dir', type=str, default='output', help='directory to save output video')
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    main(args)