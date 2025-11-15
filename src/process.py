import os
import cv2
import numpy as np

def load_images(data_path, img_size=(150,150)):
    X, y = [], []
    class_names = []

    for cls in os.listdir(data_path):
        cls_path = os.path.join(data_path, cls)

        if not os.path.isdir(cls_path):
            continue

        # ignore extra folder
        if cls == "brain_tumor_dataset":
            continue

        class_names.append(cls)

        for img_name in os.listdir(cls_path):
            img_path = os.path.join(cls_path, img_name)
            img = cv2.imread(img_path)

            if img is None:
                continue

            img = cv2.resize(img, img_size)
            X.append(img)
            y.append(class_names.index(cls))

    return np.array(X)/255.0, np.array(y), class_names
