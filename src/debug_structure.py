import os

DATA_PATH = r"C:\Users\TASNIM\Desktop\MLOps-Brain-Tumor\data\brain-tumor-data"

def count_images(path):
    classes = os.listdir(path)
    for cls in classes:
        cls_path = os.path.join(path, cls)
        if os.path.isdir(cls_path):
            count = len(os.listdir(cls_path))
            print(f"{cls}: {count} images")

count_images(DATA_PATH)
