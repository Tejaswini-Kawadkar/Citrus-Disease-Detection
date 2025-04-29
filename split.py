import os
import shutil
import random

# ✅ Define dataset path (CHANGE THIS TO YOUR DIRECTORY)
DATASET_PATH = r"C:/Users/HP/Desktop/MobileNetV2Leaf/Dataset"
OUTPUT_PATH = r"C:/Users/HP/Desktop/MobileNetV2Leaf/Dataset_split"

# ✅ Define split ratio
TRAIN_RATIO = 0.8  

# ✅ Create Train and Test folders
for category in os.listdir(DATASET_PATH):
    category_path = os.path.join(DATASET_PATH, category)
    if os.path.isdir(category_path):
        train_folder = os.path.join(OUTPUT_PATH, "train", category)
        test_folder = os.path.join(OUTPUT_PATH, "test", category)
        
        os.makedirs(train_folder, exist_ok=True)
        os.makedirs(test_folder, exist_ok=True)

        images = os.listdir(category_path)
        random.shuffle(images)

        train_count = int(len(images) * TRAIN_RATIO)

        for i, image in enumerate(images):
            src = os.path.join(category_path, image)
            dst_folder = train_folder if i < train_count else test_folder
            shutil.copy(src, dst_folder)

print("✅ Dataset successfully split into 'train' and 'test' folders!")
