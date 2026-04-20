from ultralytics import YOLO
import cv2
from pathlib import Path
import numpy as np

#===========配置参数===========
model_path = "./model/best.pt"
print(f"加载模型: {model_path}")
try:
    model = YOLO(model_path)
except Exception as e:
    print(f"加载模型 {model_path} 时出错: {e}")
    exit(0)
print("模型加载成功")
test_image_path = "./data/seg_test/images/GqxnhRddjFzxovny2Hr2_jpg.rf.f61be4a9fd66d848c9a9a585dfac34e8.jpg"
output_image_dir = "./data/seg_test/output_images"
output_label_dir = "./data/seg_test/output_labels"


# 创建输出文件夹
Path(output_image_dir).mkdir(exist_ok=True)
Path(output_label_dir).mkdir(exist_ok=True)
# 预测并保存标签
def predict_and_save_labels():
    img_path = Path(test_image_path)
    if not img_path.exists():
        print(f"图像文件 {img_path} 不存在")
        return
    stem = img_path.stem
    result = model(test_image_path, conf=0.25)[0] # 预测结果
    contour_list = []
    output_label_list = []
    output_label_path  = Path(output_label_dir) / f"{stem}.txt"
    cls = result.boxes.cls.cpu().numpy().astype(int) # 获取类别ID
    contour_list = result.masks.xy# 获取分割掩码
    contour_list = [contour.astype(np.float64) for contour in contour_list]# 转换为float64类型
    contour_list = [np.round(coord, 6) for coord in contour_list]# 坐标保留6位小数
    out_contour_list = [contour.reshape(-1).tolist() for contour in contour_list]#(n,m,2)->(n,m*2)
    for i,contour in enumerate(out_contour_list):
        out_str = " ".join([str(x) for x in contour])
        output_label_list.append(f"{cls[i]} {out_str}")
    try:
        with open(output_label_path, "w") as f:
            f.write("\n".join(output_label_list))
    except Exception as e:
        print(f"写入标签文件 {output_label_path} 时出错: {e}")
    return contour_list

# 绘制分割结果
def draw_contours(contour_list):
    img_path = Path(test_image_path)
    if not img_path.exists():
        print(f"图像文件 {img_path} 不存在")
        return
    stem = img_path.stem
    output_image_path = Path(output_image_dir) / f"{stem}.jpg"
    output_label_path = Path(output_label_dir) / f"{stem}.txt"
    
    img = cv2.imread(test_image_path)
    draw_img = img.copy()
    for contour in contour_list:
        if contour is not None:
            contour = contour.astype(np.int32) # 转换为整数类型
            cv2.drawContours(draw_img, [contour], -1, (255, 0, 0), 2)
    cv2.imwrite(str(output_image_path), draw_img)
    return draw_img

def main():
    contour_list = predict_and_save_labels()
    output_img = draw_contours(contour_list)
    
if __name__ == "__main__":
    main()
