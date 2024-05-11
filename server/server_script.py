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

from flask import Flask, request
import base64
import io
import tempfile
import datetime
import json

import utility as utils

# Define paths & Params
classifier_path = "../models/classifier_ada.pkl"
kmeans_path     = "../models/kmeans_model.pkl"
kmeans_sets     = 360


kmeans = None
clf = None
app = Flask(__name__)


class_names = [
   "IBM Plex Sans Arabic",
   "Lemonada",
   "Marhey",
   "Schererazade New"
]

def get_output_json(classifier_output):
   return {
      "class_index": int(3 - classifier_output[0]),
      "class_name": class_names[classifier_output[0]],
      "meta": [
         {
            "prbability": classifier_output[1][0].astype(float),
            "class" : class_names[0]
         },
         {
            "prbability": classifier_output[1][1].astype(float),
            "class" : class_names[1]
         },
         {
            "prbability": classifier_output[1][2].astype(float),
            "class" : class_names[2]
         },
         {
            "prbability": classifier_output[1][3].astype(float),
            "class" : class_names[3]
         }
      ]
   }

@app.route("/")
def index():
   return """<!DOCTYPE html>
<html>
<body>
  <form method="POST" enctype="multipart/form-data" action="/classify">
    <input type="file" name="image">
    <button type="submit">Upload Image</button>
  </form>
</body>
</html>"""

@app.route('/classify', methods=['POST'])
def upload_image_base64():
  image_file = request.files.get('image')

  # Check if image data is present
  if image_file:
    now = datetime.datetime.now()
    timestamp_format = "%Y-%m-%d_%H-%M-%S"  # format (YYYY-MM-DD_HH-MM-SS)
    timestamp_string = f"cache/{now.strftime(timestamp_format)}.jpg"

    # image_file.save(timestamp_string)
    # image = cv2.imread(timestamp_string)
    img_bytes = image_file.read()
    image = cv2.imdecode(np.fromstring(img_bytes, np.uint8), cv2.IMREAD_COLOR)

    # print(utils.classify_image(image , kmeans , clf))
    
    return json.dumps(get_output_json(utils.classify_image(image , kmeans , clf)))
  else:
    return 'No image data found!'

def start_server():
    global kmeans
    global clf
    print("Loading files ..")
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
        return

    app.run(debug=True)

if __name__ == "__main__":
    start_server()