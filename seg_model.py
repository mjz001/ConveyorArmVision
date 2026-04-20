from ultralytics import YOLO
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


class SegModel:
    def __init__(self, model_path="yolov8n-seg.pt", class_names=None):
        self.model = YOLO(model_path)
        self.conf_threshold = 0.5
        self.contour_list = []
        self.class_names = class_names

    def detect(self, frame):
        contour_list = []
        results = self.model(frame, conf=self.conf_threshold)
        bboxes = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                cls = box.cls[0].cpu().numpy()
                bboxes.append([x1, y1, x2, y2, conf, cls])
            if result.masks is not None and len(result.masks.xy) > 0:
                contour_list = result.masks.xy
                contour_list = [contour.astype(np.float64) for contour in contour_list]
                contour_list = [np.round(coord, 6) for coord in contour_list]
                self.contour_list = contour_list
        return bboxes

    # ====================== 已修改：轮廓固定红色，框不红色 ======================
    def draw_tracked_boxes(self, frame, tracked_bboxes, contour_list, cls_id):
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1

        for i, box in enumerate(tracked_bboxes):
            x1, y1, x2, y2, track_id = box
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # 生成随机颜色，但强制禁止红色 (BGR格式)
            np.random.seed(int(track_id))
            b = np.random.randint(100, 255)
            g = np.random.randint(100, 255)
            r = np.random.randint(0, 100)  # 红色通道调低，绝对不会出现纯红
            color = (b, g, r)

            # 获取文字
            cid = int(cls_id[i]) if i < len(cls_id) else 0
            cls_text = self.class_names[cid] if self.class_names else str(cid)
            id_text = f"ID:{int(track_id)}"
            info_text = f"{id_text} | {cls_text}"

            # 计算文字背景大小
            (text_w, text_h), baseline = cv2.getTextSize(info_text, font, font_scale, thickness)
            text_bg_x1 = x1
            text_bg_y1 = y1 - text_h - baseline - 2
            text_bg_x2 = x1 + text_w
            text_bg_y2 = y1

            # 绘制背景
            cv2.rectangle(frame, (text_bg_x1, text_bg_y1), (text_bg_x2, text_bg_y2), color, -1)

            # 白色文字
            cv2.putText(frame, info_text,
                        (x1, text_bg_y2 - baseline),
                        font, font_scale, (255,255,255), thickness)

            # 绘制目标框（非红色）
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # 轮廓 固定 红色 (0,0,255)
            if i < len(contour_list):
                cnt = contour_list[i].astype(np.int32)
                cv2.drawContours(frame, [cnt], -1, (0, 0, 255), 2)

        return frame

    def box_iou(self, box1, box2):
        x1_pred, y1_pred, x2_pred, y2_pred = box1
        w_pred = x2_pred - x1_pred
        h_pred = y2_pred - y1_pred
        box1_area = w_pred * h_pred

        x1_trk, y1_trk, x2_trk, y2_trk = box2
        w_trk = max(0, x2_trk - x1_trk)
        h_trk = max(0, y2_trk - y1_trk)
        box2_area = w_trk * h_trk

        x1_abs = max(x1_pred, x1_trk)
        y1_abs = max(y1_pred, y1_trk)
        x2_abs = min(x2_pred, x2_trk)
        y2_abs = min(y2_pred, y2_trk)

        w_abs = max(0, x2_abs - x1_abs)
        h_abs = max(0, y2_abs - y1_abs)
        inter_area = w_abs * h_abs
        union_area = box1_area + box2_area - inter_area
        return inter_area / union_area if union_area > 0 else 0

    def match_boxes(self, pre_boxes, tracker_bboxes, match_iou=0.5):
        cost_matrix = np.zeros((len(pre_boxes), len(tracker_bboxes)))
        
        for i, pre_box in enumerate(pre_boxes):
            for j, tracker_box in enumerate(tracker_bboxes):
                cost = 1 - self.box_iou(pre_box, tracker_box)
                cost_matrix[i, j] = cost

        pre_indices, tracker_indices = linear_sum_assignment(cost_matrix)

        match_pre_indices = []
        match_tracker_indices = []
        match_pairs = []

        for i, j in zip(pre_indices, tracker_indices):
            iou_val = 1 - cost_matrix[i, j]
            if iou_val >= match_iou:
                match_pre_indices.append(i)
                match_tracker_indices.append(j)
                match_pairs.append((i, j, iou_val))

        return match_pre_indices, match_tracker_indices, match_pairs

    def match_contours(self, pred_bboxes, tracker_bboxes, iou=0.5):
        contour_list = []
        match_cls_id = []
        pred_cls_id = [box[5] for box in pred_bboxes]
        pre_boxes = [[x1, y1, x2, y2] for x1, y1, x2, y2, conf, cls in pred_bboxes]
        trk_boxes = [[x1, y1, x2, y2] for x1, y1, x2, y2, track_id in tracker_bboxes]

        pre_indices, trk_indices, match_pairs = self.match_boxes(pre_boxes, trk_boxes, iou)

        for i, j, _ in match_pairs:
            if i < len(self.contour_list):
                contour_list.append(self.contour_list[i])
                match_cls_id.append(pred_cls_id[i])
        return contour_list, match_cls_id