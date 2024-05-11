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

# utility functions
def image_preprocess(img):
    #img = cv2.resize(img, (512, 512))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img = cv2.GaussianBlur(img, (3, 3), 0.1)
    img = cv2.medianBlur(img, 3)  # Kernel size can be adjusted as needed
    histogram = cv2.calcHist([img], [0], None, [256], [0, 256])
    bg_color = np.argmax(histogram)

    thresh = 0
    if bg_color < 50:  # If the background is brighter than a threshold, invert
        _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    else:
        _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)
    return thresh

def apply_pca(descriptors, n_components=64):
    pca = PCA(n_components=n_components)
    descriptors_reduced = pca.fit_transform(descriptors)
    return descriptors_reduced

def create_image_histogram(img, kmeans):
  descriptors = extract_features(img)[1]
  return create_histogram(descriptors , kmeans)

def extract_features(img):
  surf = sift_create()
  keypoints, descriptors = surf.detectAndCompute(img, None)
  return keypoints, descriptors

def create_histogram(features , kmeans , kmeans_sets = 360):
  histogram = np.zeros(kmeans_sets)
  if (features is None or len(features) == 0):
    return histogram
  # descriptors_reduced = apply_pca(features, n_components=64)  # Apply PCA
  prediction = kmeans.predict(features)
  for p in prediction:
    histogram[p] += 1
  return histogram


def classify_image(img, kmeans, clf):
  img = image_preprocess(img)
  histogram = create_image_histogram(img, kmeans)
  predictions, probabilities = clf.predict([histogram]), clf.predict_proba([histogram])
  return predictions[0], probabilities[0]