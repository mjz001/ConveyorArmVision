import cv2
import numpy as np
from pathlib import Path
import os 
def draw_segmentation_contours(label_dir, img_dir, output_dir, class_names=None):
    """
    读取YOLO分割标签，在对应图像上绘制轮廓并保存（使用Path.glob方法）
    
    Args:
        label_dir (str/Path): 标签文件目录（.txt）
        img_dir (str/Path): 图像文件目录（支持jpg/png/jpeg）
        output_dir (str/Path): 输出图像保存目录
        class_names (dict): 类别ID到名称的映射，如 {0: "box"}
    """
    # 统一转换为Path对象，方便后续操作
    label_dir = Path(label_dir)
    img_dir = Path(img_dir)
    output_dir = Path(output_dir)
    
    # 默认类别名（如果未指定）
    if class_names is None:
        print("error:未提供类别映射!!!")
        return
    
    # 创建输出目录（不存在则创建）
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ========== 核心修改：用glob获取所有.txt标签文件 ==========
    label_files = list(label_dir.glob("*.txt"))
    if not label_files:
        print(f"标签目录 {label_dir} 下未找到任何.txt标签文件")
        return
    
    # 支持的图像格式（按优先级排序）
    img_extensions = [".jpg", ".jpeg", ".png", ".bmp"]
    
    # 遍历所有标签文件
    for label_file in label_files:
        try:
            # 1. 解析标签文件名（无需手动处理字符串，Path更便捷）
            img_stem = label_file.stem  # 去掉后缀的文件名（如box_001.txt → box_001）
            print(f"处理文件: {img_stem}")
            
            # 2. 寻找同名图像（用glob匹配更高效）
            img_path = None
            for ext in img_extensions:
                # 用glob精准匹配同名同格式文件
                
                if os.path.exists(str(img_dir / f"{img_stem}{ext}")):
                    img_path = img_dir / f"{img_stem}{ext}"
                    break
            
            if img_path is None:
                print(f"未找到 {img_stem} 对应的图像文件，跳过")
                continue
            
            # 3. 读取图像
            img = cv2.imread(str(img_path))  # cv2需要字符串路径
            if img is None:
                print(f"图像 {img_path} 读取失败，跳过")
                continue
            img_h, img_w = img.shape[:2]
            
            # 4. 读取并解析分割标签
            with open(label_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # 5. 遍历每个目标的轮廓
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # 解析标签行：class_id + 归一化坐标点
                parts = list(map(float, line.split()))
                class_id = int(parts[0])
                coords = np.array(parts[1:]).reshape(-1, 2)  # (n, 2) 形状
                
                # 反归一化：将0~1的坐标转成图像像素坐标
                coords[:, 0] *= img_w
                coords[:, 1] *= img_h
                coords = coords.astype(np.int32)  # 转整数像素坐标
                
                # 6. 绘制轮廓（优化视觉效果）
                class_name = class_names.get(class_id, f"class_{class_id}")
                # 为不同类别分配不同颜色（随机但固定）
                np.random.seed(class_id)
                color = tuple(np.random.randint(0, 255, 3).tolist())
                
                # 绘制轮廓（粗线条）
                cv2.drawContours(img, [coords], -1, color, 2)
                # 绘制类别名称（左上角）
                cx, cy = int(np.mean(coords[:, 0])), int(np.mean(coords[:, 1]))
                cv2.putText(img, class_name, (cx-20, cy), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # 7. 保存绘制后的图像（统一保存为jpg格式）
            output_path = output_dir / f"{img_stem}.jpg"
            cv2.imwrite(str(output_path), img)
            print(f"completed: {output_path}")
        
        except Exception as e:
            print(f"处理 {label_file.name} 时出错: {str(e)}")
            continue
    
    print("\n所有文件处理完成！")

# ====================== 主函数（修改这里的路径即可） ======================
if __name__ == "__main__":
    LABEL_DIR = "/home/T7/zmj/机器臂实时抓取传送带货物系统/data/boxes_parcels_det_data/valid/labels"       # 标签文件目录（存放.txt）
    IMG_DIR = "/home/T7/zmj/机器臂实时抓取传送带货物系统/data/boxes_parcels_det_data/valid/images"         # 图像文件目录（存放jpg/png等）
    OUTPUT_DIR = "/home/T7/zmj/机器臂实时抓取传送带货物系统/data/boxes_parcels_det_data/valid/output_imgs" # 输出目录（绘制后保存）
    
    # 类别映射（key=类别ID，value=类别名）
    CLASS_NAMES = {
        0: "box",
        1: "parcel"  
    }
    
    # 执行绘制
    draw_segmentation_contours(
        label_dir=LABEL_DIR,
        img_dir=IMG_DIR,
        output_dir=OUTPUT_DIR,
        class_names=CLASS_NAMES
    )