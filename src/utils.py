import os
import cv2
import numpy as np

def load_images(data_path, img_size=(150,150)):
    X, y = [], []
    class_names = []

    for cls in ['no', 'yes']: 
        cls_path = os.path.join(data_path, cls)
        
        if not os.path.exists(cls_path):
            print(f"Warning: Path {cls_path} does not exist")
            continue

        class_names.append(cls)

        for img_name in os.listdir(cls_path):
            img_path = os.path.join(cls_path, img_name)
            img = cv2.imread(img_path)

            if img is None:
                continue

            img = cv2.resize(img, img_size)
            X.append(img)
            y.append(0 if cls == 'no' else 1) 

    print(f"Loaded {len(X)} images")
    print(f"Class distribution: {np.bincount(y)}")
    return np.array(X), np.array(y), class_names 