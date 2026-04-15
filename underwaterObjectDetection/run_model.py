from ultralytics import YOLO
import cv2
import os
import matplotlib.pyplot as plt
import glob

# Load trained model
model = YOLO("model/best1.pt")

# Folder containing input images (dataset valid split ships with the repo)
input_folder = "aquarium_pretrain/valid/images"

# Run detection on all images in folder
results = model.predict(
    source=input_folder,
    save=True,
    project=".",
    name="yolo_predictions",
    exist_ok=True,
    conf=0.4,
)

# Ultralytics writes under runs/detect/<name>/ — use save_dir from the first result
output_folder = results[0].save_dir if results else ""
image_files = sorted(
    glob.glob(os.path.join(output_folder, "**", "*.*"), recursive=True)
)

if len(image_files) == 0:
    print("No output images found!")
else:
    for img_path in image_files:
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.imshow(img)
            plt.title(os.path.basename(img_path))
            plt.axis("off")
            plt.show()
        else:
            print("Could not load:", img_path)

