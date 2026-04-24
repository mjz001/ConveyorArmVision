import os
import logging
from typing import List, Dict, Any, Optional
import os

# 输出当前工作目录
print("当前运行目录:", os.getcwd())
# 允许windows重复加载dll加速工具包
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# omp设置并行计算的线程数
os.environ["OMP_NUM_THREADS"] = "1"
# mkl数学计算加速库内部的线程数
os.environ["MKL_NUM_THREADS"] = "1"
import cv2
import sys
# 添加项目根目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from seg_model import SegModel
from tracker import DeepSORTTracker
import yaml
from pathlib import Path
import time
from fastapi import FastAPI, HTTPException, Query
import uvicorn
from pydantic import BaseModel
from PIL import Image
import io
import base64
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse


# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 获取类别名称映射
def get_class_names(yaml_path="config/class_name.yaml"):
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    return data.get('class_names', None)


# API服务初始化区域
app = FastAPI(
    title="YOLOv8-Seg-DeepSORT",
)

# 添加CORS中间件以支持前端跨域请求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 在生产环境中应该指定具体的域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 请求和响应模型
class ImageRequest(BaseModel):
    image: str

class MultipleImagesRequest(BaseModel):
    image: str

class DetectionResult(BaseModel):
    tracker_boxes: List[List[float]]
    class_names: List[str]
    contour_list: List[Any]
    object_count: int
    processing_time: float

class SystemStatus(BaseModel):
    status: str
    model_loaded: bool
    tracker_initialized: bool
    total_processed: int
    uptime: float


# 全局变量
model_path = "weights/best.onnx"
class_yaml = "config/class_name.yaml"
start_time = time.time()

print(f"加载模型: {model_path}")
class_dict = get_class_names(class_yaml)
seg_model = SegModel(model_path=model_path, class_names=class_dict)
print(f"加载模型成功")
tracker = DeepSORTTracker()
processed_count = 0  # 统计已处理的图片数量


def decodebase64(base64_str: str) -> np.ndarray:
    """
    解码base64字符串为OpenCV图像
    """
    try:
        image_bytes = base64.b64decode(base64_str)
        np_arr = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if image is None:
            raise HTTPException(status_code=400, detail="无法解码图像数据，请检查base64格式是否正确")
        return image
    except Exception as e:
        logger.error(f"解码base64图像时出错: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"无效的图像数据: {str(e)}"
        )


def encodebase64(frame: np.ndarray) -> str:
    """
    将OpenCV图像编码为base64字符串
    """
    try:
        _, buffer = cv2.imencode('.jpg', frame)
        image_base64_str = base64.b64encode(buffer).decode('utf-8')
        return image_base64_str
    except Exception as e:
        logger.error(f"编码base64图像时出错: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"编码图像为base64时出错: {str(e)}"
        )


def solve(base64_str: str) -> Dict[str, Any]:
    """
    处理单张图像的主要逻辑
    """
    start_time = time.time()
    
    image = decodebase64(base64_str)
    pre_boxes = seg_model.detect(image)
    
    logger.info(f"检测到的对象数量: {len(pre_boxes) if pre_boxes else 0}")
    
    tracker_boxes = tracker.update(pre_boxes, image)
    
    # 转换NumPy数组为Python原生类型以确保JSON序列化兼容
    tracker_boxes = [
        [float(val) for val in box.tolist()] if isinstance(box, np.ndarray) else [float(val) for val in box]
        for box in tracker_boxes
    ]
    
    # 匹配轮廓和类别ID
    try:
        contour_list, cls_id = seg_model.match_contours(pre_boxes, tracker_boxes)
        
        # 根据类别ID获取类别名称
        class_names = [class_dict[int(id)] if int(id) in class_dict else f"unknown_{id}" for id in cls_id]
        
        processing_time = round(time.time() - start_time, 3)
        
        result = {
            "tracker_boxes": tracker_boxes,
            "class_names": class_names,
            "contour_list": contour_list,
            "object_count": len(tracker_boxes),
            "processing_time": processing_time
        }
        
        global processed_count
        processed_count += 1
        
        return result
    except Exception as e:
        logger.error(f"处理图像时出错: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"处理图像时出错: {str(e)}"
        )


@app.get("/")
async def root():
    """
    根路径,返回API基本信息
    """
    response = {
        "title": "YOLOv8-Seg-DeepSORT",
        "endpoints": [
            "GET /", 
            "GET /health", 
            "POST /track", 
            "POST /track_multiple"
        ]
    }
    return JSONResponse(content=response,status_code=200)


@app.get("/health", response_model=SystemStatus)
async def health_check():
    """
    检查端点组件是否加载
    """
    components_status = {
        "model_loaded": seg_model is not None,
        "tracker_initialized": tracker is not None,
        "model_has_detect_method": hasattr(seg_model, 'detect') if seg_model else False,
        "tracker_has_update_method": hasattr(tracker, 'update') if tracker else False
    }
    overall_status = "healthy" if all(components_status.values()) else "unhealthy"
    response = {
        "status": overall_status,
        "components": components_status,
        "timestamp": time.time()
    }
    
    return JSONResponse(content=response,status_code=200)




@app.post("/track", response_model=DetectionResult)
async def track(request: ImageRequest):
    """
    处理单张图像的追踪请求
    """
    try:
        result = solve(request.image)
        return JSONResponse(content=result,status_code=200)
    except Exception as e:
        logger.error(f"处理单帧图像请求时出错: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"处理单帧图像请求时出错: {str(e)}"
        )





@app.post("/track_multiple",response_model=MultipleImagesRequest)
async def track_multiple(request: ImageRequest):
    """
    对图像进行标注并返回带标注的图像
    """
    try:
        image = decodebase64(request.image)
        original_image = image.copy()
        
        pre_boxes = seg_model.detect(image)
        tracker_boxes = tracker.update(pre_boxes, image)
        
        # 转换NumPy数组为Python原生类型
        tracker_boxes = [
            [float(val) for val in box.tolist()] if isinstance(box, np.ndarray) else [float(val) for val in box]
            for box in tracker_boxes
        ]
        
        # 匹配轮廓和类别ID
        contour_list, cls_id = seg_model.match_contours(pre_boxes, tracker_boxes)
        class_names = [class_dict[int(id)] if int(id) in class_dict else f"unknown_{id}" for id in cls_id]
        
        # 在图像上绘制边界框和标签
        annotated_image = original_image.copy()
        for i, box in enumerate(tracker_boxes):
            if i < len(contour_list) and i < len(cls_id):
                x1, y1, x2, y2, track_id = map(int, box[:5])  # 假设box格式为[x1, y1, x2, y2, track_id, ...]
                
                # 绘制边界框
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # 绘制标签
                label = f"{class_names[i]} ID:{track_id}"
                cv2.putText(
                    annotated_image, 
                    label, 
                    (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, 
                    (0, 255, 0), 
                    2
                )
                
                # 绘制轮廓
                if i < len(contour_list) and contour_list[i] is not None:
                    cv2.drawContours(annotated_image, [np.array(contour_list[i], dtype=np.int32)], -1, (255, 0, 0), 2)
        
        # 编码带标注的图像
        annotated_base64 = encodebase64(annotated_image)
        response = {
            "annotated_image": annotated_base64
        }
        
        return JSONResponse(content=response,status_code=200)
    except Exception as e:
        logger.error(f"处理图像标注时出错: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"处理图像标注时出错: {str(e)}"
        )



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)