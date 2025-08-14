import os
import cv2
import numpy as np

data_dir = r"D:\compvision\trafficsigns\GTSRB-Training_fixed\GTSRB\Training"


classes = [
    f for f in os.listdir(data_dir)
    if os.path.isdir(os.path.join(data_dir, f))
]
print("Number of classes:", len(classes))

X = []
y = []
IMG_SIZE = 32

for label, class_folder in enumerate(classes):
    folder_path = os.path.join(data_dir, class_folder)

    image_files = [
        f for f in os.listdir(folder_path)
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.ppm', '.bmp'))
    ]

    for file_name in image_files:
        img_path = os.path.join(folder_path, file_name)
        img = cv2.imread(img_path)

        if img is None:
            print(f"Error: {img_path}")
            continue

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        X.append(img)
        y.append(label)

X = np.array(X) / 255.0
y = np.array(y)
print("Data shape:", X.shape, "Labels:", y.shape)