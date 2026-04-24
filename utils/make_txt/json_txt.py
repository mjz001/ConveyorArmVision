import json
from pathlib import Path

def mk_boxxes(labels_dir,json_labels_dir,out_labels_dir):
    for label_path in Path(json_labels_dir).glob("*.json"):
        stem = label_path.stem
        out_label_path = Path(labels_dir)/(stem+".txt")
        out_only_path=Path(out_labels_dir)/(stem+".txt")
        targes=[]
        only_targes=[]
        try:
            with open(out_label_path,"r") as rf:
                targes=rf.read().strip().split("\n")
        except Exception as e:
            print(f"txt文件: {out_label_path}\t 错误：{e}")
        try:
            with open(label_path, "r") as f:
                json_data = json.load(f)
                h_size, w_size = float(json_data["imageHeight"]), float(json_data["imageWidth"])
                bboxes = json_data["shapes"]
                for bbox in bboxes:
                    cls = float(bbox["label"])
                    xs = [float(p[0]) for p in bbox["points"]]
                    ys = [float(p[1]) for p in bbox["points"]]
                    x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
                    w_abs = (x2 - x1) / w_size
                    h_abs = (y2 - y1) / h_size
                    x_abs = (x1 + x2) / 2
                    y_abs = (y1 + y2) / 2
                    x_abs /= w_size
                    y_abs /= h_size
                    targes.append(f"{int(cls)} {x_abs:.6f} {y_abs:.6f} {w_abs:.6f} {h_abs:.6f}")
                    only_targes.append(f"{int(cls)} {x_abs:.6f} {y_abs:.6f} {w_abs:.6f} {h_abs:.6f}")
                    print(f"{int(cls)} {x_abs:.6f} {y_abs:.6f} {w_abs:.6f} {h_abs:.6f}")
        except Exception as e:
            print(f"json文件: {label_path}\t 错误：{e}")
        try:
            with open (out_label_path,"w") as f_out:
                f_out.write("\n".join(targes))
        except Exception as e:
            print(f"txt文件: {out_label_path}\t 错误：{e}")
        try:
            with open(out_only_path,"w") as f_only_out:
                f_only_out.write("\n".join(only_targes))
        except Exception as e:
            print(f"txt文件: {out_only_path}\t 错误：{e}")
        print(f"complete:{label_path}")

if __name__=="__main__":
    labels_dir=r"D:/AIM/MyAim/new_datas/valid_qipao_large/labels"
    json_labels_dir = r"D:/AIM/MyAim/new_datas/qipao/valid/labels_json_large"
    out_labels_dir=r"D:/AIM/MyAim/new_datas/qipao/valid/labels_json_txt"
    mk_boxxes(labels_dir,json_labels_dir,out_labels_dir)