import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import cv2
from seg_model import SegModel
from tracker import DeepSORTTracker
import yaml

model_path = "./model/best.pt"

#获取类名映射
def get_class_names(yaml_path="config/class_name.yaml"):
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    return data.get('class_names', None)

def main():
    # ==================  初始化 ==================
    class_names = get_class_names()
    print(f"加载模型: yolov8n-seg.pt")
    try:
        seg_model = SegModel(model_path="yolov8n-seg.pt", class_names=class_names)
    except Exception as e:
        print(f"加载模型失败: {e}")
        return
    print(f"加载模型成功")
    tracker = DeepSORTTracker()

    # ================== 打开视频流 ==================
    video_path = "test.mp4"  # 0=摄像头，也可以填视频路径   
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
    out = cv2.VideoWriter("output_tracking.mp4", fourcc, fps, (width, height))

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
            if cv2.waitKey(1) == 27:
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
    main()