import os
import json

# === Configuration ===
json_dir = "../datasets/spatially_compressed/labels-json"  # JSON input
labels_dir = "../datasets/spatially_compressed/labels"     # YOLO output
images_dir = "../datasets/spatially_compressed/images"     # Images directory
dataset_root = "../datasets/spatially_compressed"          # Root directory

train_img_dir = os.path.join(images_dir, "train")
val_img_dir = os.path.join(images_dir, "val")
train_lbl_dir = os.path.join(labels_dir, "train")
val_lbl_dir = os.path.join(labels_dir, "val")

# === Class map ===
class_map = {
    "pedestrian": 0
}
id_to_class = {v: k for k, v in class_map.items()}

os.makedirs(train_lbl_dir, exist_ok=True)
os.makedirs(val_lbl_dir, exist_ok=True)

def convert_box(obj, img_w, img_h):
    x0, y0, x1, y1 = obj['x0'], obj['y0'], obj['x1'], obj['y1']
    x_center = ((x0 + x1) / 2) / img_w
    y_center = ((y0 + y1) / 2) / img_h
    width = (x1 - x0) / img_w
    height = (y1 - y0) / img_h
    return x_center, y_center, width, height

def convert_json_to_yolo(json_path, output_txt_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    img_w = data.get('imagewidth')
    img_h = data.get('imageheight')
    objects = data.get('children', [])

    yolo_lines = []
    for obj in objects:
        label = obj.get('identity')
        if label not in class_map:
            continue

        class_id = class_map[label]
        try:
            x_center, y_center, width, height = convert_box(obj, img_w, img_h)
            yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
        except KeyError as e:
            print(f"Missing bbox key in {json_path}: {e}")

    if yolo_lines:
        with open(output_txt_path, 'w') as f:
            f.write('\n'.join(yolo_lines))

# === Split train/val (you can customize this logic) ===
json_files = sorted([f for f in os.listdir(json_dir) if f.endswith('.json')])
split_index = int(0.8 * len(json_files))
train_files = json_files[:split_index]
val_files = json_files[split_index:]

# === Process files ===
for split, file_list, label_subdir in [('train', train_files, train_lbl_dir), ('val', val_files, val_lbl_dir)]:
    for fname in file_list:
        json_path = os.path.join(json_dir, fname)
        base_name = os.path.splitext(fname)[0]
        label_path = os.path.join(label_subdir, base_name + ".txt")
        convert_json_to_yolo(json_path, label_path)

# === Write data.yaml ===
data_yaml = f"""train: {os.path.abspath(train_img_dir)}
val: {os.path.abspath(val_img_dir)}

nc: {len(class_map)}
names: {list(class_map.keys())}
"""

with open(os.path.join(dataset_root, "data.yaml"), "w") as f:
    f.write(data_yaml)

print("✅ Conversion complete. `data.yaml` written.")
