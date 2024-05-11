import cv2
from sklearn.cluster import KMeans
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC 
from sklearn.model_selection import train_test_split
import os
import numpy as np
from xgboost import XGBClassifier
from cv2.xfeatures2d import SIFT_create as sift_create
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
import sys
import time

import utility as utils

# Define paths & Params
classifier_path = "../models/classifier_ada.pkl"
kmeans_path     = "../models/kmeans_model.pkl"
kmeans_sets     = 360

kmeans = None
clf = None


print("Loading cfg files ..")
print("Preparing Kmeans (Bag of words)")
with open(kmeans_path, 'rb') as f:
    kmeans = pickle.load(f)
if kmeans is not None:
    print("Loaded")
else:
    print("Failed to Load")

with open(classifier_path, 'rb') as f:
    clf = pickle.load(f)
if clf is not None:
    print("Loaded")
else:
    print("Failed to Load")

if kmeans is None or clf is None:
    print("Error, Failed to load server files ..")


data_path = sys.argv[1]
print("Loading Images Paths.. ")
images_paths = []
for img_path in os.listdir(data_path):
    images_paths.append(int(img_path[:-5])) # remove the .jpeg
images_paths.sort()

print(f"Loaded: {len(images_paths)} images.")

print("Beginning test.")
timimg_file = open("time.txt" , 'w')
results_file = open("results.txt" , 'w')

for img_path in images_paths:
    # load the image
    img = cv2.imread(os.path.join(data_path , f"{str(img_path)}.jpeg"))
    start = time.time()
    prediction , _ = utils.classify_image(img , kmeans , clf)
    delta = time.time() - start
    print(f"Prediction for: \"{img_path}.jpeg\" = {prediction}  , in {delta:.3f} seconds")
    timimg_file.write(f"{delta:.3f}\n")
    results_file.write(f"{3 - prediction}\n")

timimg_file.close()
results_file.close()
print("Test ended")

