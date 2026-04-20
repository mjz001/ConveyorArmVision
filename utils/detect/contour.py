import cv2
import numpy as np
from ultralytics import YOLO


image_path = "/home/T7/zmj/机器臂实时抓取传送带货物系统/data/boxes_parcels_det_data/valid/images/4wMoyPeb5qQQLqi1Mpzq_jpg.rf.3d6354f4f97bd30d2452772aed9fea0c.jpg"
label_path = "/home/T7/zmj/机器臂实时抓取传送带货物系统/data/boxes_parcels_det_data/valid/labels/4wMoyPeb5qQQLqi1Mpzq_jpg.rf.3d6354f4f97bd30d2452772aed9fea0c.txt"
draw_dir = "/home/T7/zmj/机器臂实时抓取传送带货物系统/data/detection_test/draw_images"
pred_name = "pred_img.jpg"
contours_name  = "contours_img.jpg"

# ==========================
# 轮廓处理类（你要的核心封装）
# ==========================
class ContourProcessor:
    def __init__(self):
        pass

    def extract_contour_from_box(self, img, box, padding=2):
        """
        从 YOLO 检测框中提取 货物/纸箱 轮廓
        :param img: 原始图像
        :param box: YOLO 的 xyxy 框 [x1, y1, x2, y2]
        :return: 最大轮廓 (contour)
        """
        # 截取检测框区域
        x1, y1, x2, y2 = box
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(img.shape[1], x2 + padding)
        y2 = min(img.shape[0], y2 + padding)

        crop = img[y1:y2, x1:x2]

        # 预处理：灰度 + 二值化（适合纸箱/货物）
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        #   _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, binary = cv2.threshold(blur, 125, 255, cv2.THRESH_BINARY )
        cv2.imwrite("binary.jpg", binary)

        # 形态学，让轮廓更干净
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        cv2.imwrite("binary_morph.jpg", binary)

        # 查找轮廓
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return None
        """
        # 5. 取最大轮廓（货物一般只有一个主体）
        #max_contour = max(contours, key=cv2.contourArea)
        # 6. 把轮廓坐标转回原图坐标系
        max_contour[:, :, 0] += x1
        max_contour[:, :, 1] += y1
        """
        corrected_contours = []
        for contour in contours:
            #if cv2.contourArea(contour)>500 and cv2.contourArea(contour)<1000:
                contour[:, :, 0] += x1
                contour[:, :, 1] += y1
                corrected_contours.append(contour)
                
                
        return corrected_contours

    def draw_all_contours(self, img, contours_list):
        """
        绘制所有轮廓
        img: 原图
        contours_list: 轮廓列表
        """
        output = img.copy()
        for cnt in contours_list:
            if cnt is not None:
                cv2.drawContours(output, [cnt], -1, (255, 0, 0), 2)
        return output
    def cv_show(self, name, img):
        cv2.imshow(name,img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
# ==========================
# YOLO 推理主流程
# ==========================
def main():
    """
    # 1. 加载 YOLO 模型
    model = YOLO("yolov8n.pt")  # 你可以换成自己的 best.pt

    # 2. 读取图像
    img = cv2.imread("test.jpg")

    # 3. YOLO 推理
    results = model(img)
    # 4. 初始化轮廓处理器
    contour_proc = ContourProcessor()
    all_contours = []

    # 5. 遍历每个检测框
    for result in results:
        for box in result.boxes:
            # 获取框坐标
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            # 提取轮廓
            cnt = contour_proc.extract_contour_from_box(img, xyxy)
            all_contours.append(cnt)
    """
    contour_proc = ContourProcessor()
    print("测试单张图像的轮廓提取")
    
    #读取图像，计算图像的长宽属性
    img = cv2.imread(image_path)
    pred_img = img.copy()
    cv2.imwrite("img.jpg", img)
    img_h,img_w = img.shape[:2]
    print(f"图像宽度: {img_w}, 图像高度: {img_h}")
    # 所有检测框轮廓列表
    all_contours = []
    try:
        with open(label_path, "r") as f:
            for line in f:
                line = line.strip().split()
                if not line:
                    continue
                x_center, y_center, width, height = map(float, line[1:5])
                xmin = int((x_center - width / 2) * img_w)
                ymin = int((y_center - height / 2) * img_h)
                xmax = int((x_center + width / 2) * img_w)
                ymax = int((y_center + height / 2) * img_h)
                cv2.rectangle(pred_img, (xmin, ymin), (xmax, ymax), (0, 255, 0), thickness=2)
                xyxy = [xmin, ymin, xmax, ymax]
                print(f"检测框坐标: {xyxy}")
                # 检测框轮廓列表
                contours = contour_proc.extract_contour_from_box(img, xyxy)
                if contours is None:
                    print("没有找到轮廓")
                    return
                print(f"轮廓数量: {len(contours)}")
                all_contours.extend(contours)
            
    except Exception as e:
        print(f"读取{label_path}文件错误: {e}")
        return
    # 保存检测框图像
    cv2.imwrite(f"{draw_dir}/{pred_name}", pred_img)
    # 绘制轮廓
    img_out = contour_proc.draw_all_contours(img, all_contours)
    # 显示保存
    cv2.imwrite(f"{draw_dir}/{contours_name}", img_out)   
    

if __name__ == "__main__":
    main()