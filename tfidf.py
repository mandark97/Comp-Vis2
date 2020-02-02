from joblib import dump, load
from utils import get_imgs
import cv2

kmeans = load("kmeans100000.joblib")

imgs = get_imgs("database/")

sift = cv2.xfeatures2d.SIFT_create()

desc_list = []
for word, img_list in imgs.items():
    for img in img_list:
        kp, desc = sift.detectAndCompute(img, None)

        desc_list.append(desc)

