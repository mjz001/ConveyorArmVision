import numpy as np
from deep_sort_pytorch.deep_sort import DeepSort
from deep_sort_pytorch.utils.parser import get_config

class DeepSORTTracker:
    def __init__(self):
        cfg = get_config()
        cfg.merge_from_file("deep_sort_pytorch/configs/deep_sort.yaml")

        self.deepsort = DeepSort(
            cfg.DEEPSORT.REID_CKPT,
            max_dist=cfg.DEEPSORT.MAX_DIST,
            min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
            nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP,
            max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
            max_age=cfg.DEEPSORT.MAX_AGE,
            n_init=cfg.DEEPSORT.N_INIT,
            nn_budget=cfg.DEEPSORT.NN_BUDGET,
            use_cuda=True
        )

    def xyxy_to_xywh(self, x1, y1, x2, y2):
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        return cx, cy, w, h

    # ==============================================
    # 这里！接收 bboxes 和 frame（图像）
    # ==============================================
    def update(self, bboxes, img):
        if len(bboxes) == 0:
            self.deepsort.increment_ages()
            return []

        xywhs = []
        confs = []
        for box in bboxes:
            x1, y1, x2, y2, conf, cls = box
            cx, cy, w, h = self.xyxy_to_xywh(x1, y1, x2, y2)
            xywhs.append([cx, cy, w, h])
            confs.append(conf)

        xywhs = np.array(xywhs)
        confs = np.array(confs)

        # ==============================================
        # 标准 DeepSORT 3 个参数！
        # ==============================================
        outputs = self.deepsort.update(xywhs, confs, img)
        return outputs