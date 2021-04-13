#!/usr/bin/env python
# coding: utf-8

# Multi-Task Convoluted Neural Networks
from mtcnn.mtcnn import MTCNN
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
from scipy.spatial.distance import cosine
# matplot for reading out images
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from numpy import asarray
from PIL import Image

import tensorflow as tf


# Get images from an external server and store it locally
import urllib.request

def store_image(url, local_file_name):
  with urllib.request.urlopen(url) as resource:
    with open(local_file_name, 'wb') as f:
      f.write(resource.read())


def highlight_faces(image_path, faces):
    # display image
    image = plt.imread(image_path)
    plt.imshow(image)
    # To get the current polar axes on the current figure:
    ax = plt.gca()

    # for each face, draw a rectangle based on coordinates
    for face in faces:
        x, y, width, height = face['box']
        face_border = Rectangle((x, y), width, height, fill=False, color='green')
        ax.add_patch(face_border)
    plt.show()



def extract_face_from_image(image_path, required_size=(224, 224)):
    # load image and detect faces
    image = plt.imread(image_path)
    detector = MTCNN()
    faces = detector.detect_faces(image)

    print(f"faces : {faces}")

    face_images = []

    face_imgobj = []

    
    for face in faces:
        # extract the bounding box from the requested face
        x1, y1, width, height = face['box']
        x2, y2 = x1 + width, y1 + height

        # extract the face
        face_boundary = image[y1:y2, x1:x2]

        # resize pixels to the model size
        face_image = Image.fromarray(face_boundary)
        face_image = face_image.resize(required_size)
        face_array = asarray(face_image)
        face_images.append(face_array)

        face_imgobj.append(face_image)
    
    # return face_images
    
    return face_images, face_imgobj

def get_model_scores(faces):
    samples = asarray(faces, 'float32')

    # prepare the data for the model
    samples = preprocess_input(samples, version=2)

    # create a vggface model object

    # model=resnet50
    model = VGGFace(model='senet50',
      include_top=False,
      input_shape=(224, 224, 3),
      pooling='avg')

    # perform prediction
    return model.predict(samples)

def detect_face(image_one, image_two):
  """Take to images and detects the faces
    Return:
      `detected_faces`: A list containing images i.e faces
  """

  face1, face1_img = extract_face_from_image(image_one)
  face2, face2_img = extract_face_from_image(image_two)
  
  model_scores_1 = get_model_scores(face1)
  model_scores_2 = get_model_scores(face2)

  image_pairs = []

  for idx, face_score_1 in enumerate(model_scores_1):
    for idy, face_score_2 in enumerate(model_scores_2):
      score = cosine(face_score_1, face_score_2)
      if score <= 0.4:
        image_pairs.append((face1_img[idx], face2_img[idy]))

  return image_pairs

