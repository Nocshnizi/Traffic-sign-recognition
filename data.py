import os
import cv2
import matplotlib.pyplot as plt

data_dir = r"D:\compvision\trafficsigns\GTSRB-Training_fixed\GTSRB\Training"
classes = os.listdir(data_dir)
sample_class = os.path.join(data_dir, classes[0])  # Перша папка-клас

# Отримуємо список файлів, ігноруючи CSV
images = [
    f for f in os.listdir(sample_class)
    if not (f.startswith('GT_') or f.endswith('.csv'))
]

# Виводимо тільки перші 5 зображень
plt.figure(figsize=(10, 5))
for i, img_name in enumerate(images[:5]):
    img_path = os.path.join(sample_class, img_name)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.subplot(1, 5, i+1)
    plt.imshow(img)
    plt.axis('off')

plt.show(block=False)
plt.pause(3)
plt.close()