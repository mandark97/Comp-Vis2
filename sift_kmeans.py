from utils import get_imgs
from joblib import dump, load
from sklearn.cluster import MiniBatchKMeans
import cv2
import numpy as np

N_CLUSTER = 100000


def train(n_clusters=N_CLUSTER):
    imgs = get_imgs("database/")
    sift = cv2.xfeatures2d.SIFT_create()

    desc_list = []
    for word, img_list in imgs.items():
        for img in img_list:
            kp, desc = sift.detectAndCompute(img, None)

            desc_list.extend(desc)

    kmeans = MiniBatchKMeans(n_clusters=n_clusters, n_init=1)
    kmeans.fit(desc_list)

    dump(kmeans, f'kmeans{n_clusters}.joblib')
