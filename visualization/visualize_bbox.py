import os
import cv2
import matplotlib.pyplot as plt

# === Configuration ===
img_dir = "../datasets/mixed/img/train"  # or val
lbl_dir = "../datasets/mixed/img/train"  # label files are in same structure
class_names = ["pedestrian"]

def yolo_to_box(x_center, y_center, width, height, img_w, img_h):
    x0 = int((x_center - width / 2) * img_w)
    y0 = int((y_center - height / 2) * img_h)
    x1 = int((x_center + width / 2) * img_w)
    y1 = int((y_center + height / 2) * img_h)
    return x0, y0, x1, y1

def draw_boxes_on_image(image_path, label_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_h, img_w, _ = img.shape

    if not os.path.exists(label_path):
        print(f"Label file not found: {label_path}")
        return img

    with open(label_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        class_id, x_center, y_center, width, height = map(float, line.strip().split())
        x0, y0, x1, y1 = yolo_to_box(x_center, y_center, width, height, img_w, img_h)
        color = (255, 0, 0)
        label = class_names[int(class_id)]
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)
        cv2.putText(img, label, (x0, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return img

# === Visualize a few samples ===
sample_files = [f for f in os.listdir(img_dir) if f.endswith('.png')][:5]

for fname in sample_files:
    image_path = os.path.join(img_dir, fname)
    label_path = os.path.join(lbl_dir, os.path.splitext(fname)[0] + ".txt")
    img = draw_boxes_on_image(image_path, label_path)
    
    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    plt.title(fname)
    plt.axis('off')
    plt.show()
