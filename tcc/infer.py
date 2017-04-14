#!/usr/bin/env python2
#
# Example to classify faces.
# Brandon Amos
# 2015/10/11
#
# Copyright 2015-2016 Carnegie Mellon University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time

start = time.time()

import argparse
import cv2
import os
import pickle

from operator import itemgetter

import numpy as np
np.set_printoptions(precision=2)
import pandas as pd

import openface

from sklearn.pipeline import Pipeline
from sklearn.lda import LDA
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn.mixture import GMM
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

fileDir = os.path.dirname(os.path.realpath(__file__))
modelDir = os.path.join(fileDir, '..', 'models')
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')

start = time.time()

align = openface.AlignDlib(os.path.join(dlibModelDir,"shape_predictor_68_face_landmarks.dat"))
net = openface.TorchNeuralNet(os.path.join(openfaceModelDir,'nn4.small2.v1.t7'), 96)

def getRep(imgPath, multiple=False):
    start = time.time()
    bgrImg = cv2.imread(imgPath)
    if bgrImg is None:
        raise Exception("Unable to load image: {}".format(imgPath))

    rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)

    #if args.verbose:
    #    print("  + Original size: {}".format(rgbImg.shape))
    #if args.verbose:
    #    print("Loading the image took {} seconds.".format(time.time() - start))

    start = time.time()

    if multiple:
        bbs = align.getAllFaceBoundingBoxes(rgbImg)
    else:
        bb1 = align.getLargestFaceBoundingBox(rgbImg)
        bbs = [bb1]
    if len(bbs) == 0 or (not multiple and bb1 is None):
        raise Exception("Unable to find a face: {}".format(imgPath))
    #if args.verbose:
    #    print("Face detection took {} seconds.".format(time.time() - start))

    reps = []
    for bb in bbs:
        start = time.time()
        alignedFace = align.align(
            96,
            rgbImg,
            bb,
            landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
        if alignedFace is None:
            raise Exception("Unable to align image: {}".format(imgPath))
        #if args.verbose:
         #   print("Alignment took {} seconds.".format(time.time() - start))
         #   print("This bbox is centered at {}, {}".format(bb.center().x, bb.center().y))

        start = time.time()
        rep = net.forward(alignedFace)
        #if args.verbose:
        #    print("Neural network forward pass took {} seconds.".format(
        #        time.time() - start))
        reps.append((bb.center().x, rep))
    sreps = sorted(reps, key=lambda x: x[0])
    return sreps

def infer(classifierModel, imgPath, multiple=True):
    with open(classifierModel, 'r') as f:
        (le, clf) = pickle.load(f)
    
    json = '{ "result":['

    print("\n=== {} ===".format(imgPath))
    reps = getRep(imgPath, multiple)
    if len(reps) > 1:
        print("List of faces in image from left to right")
    for r in reps:
        rep = r[1].reshape(1, -1)
        bbx = r[0]
        start = time.time()
        predictions = clf.predict_proba(rep).ravel()
        maxI = np.argmax(predictions)
        person = le.inverse_transform(maxI)
        confidence = predictions[maxI]
        #if args.verbose:
        print("Prediction took {} seconds.".format(time.time() - start))
        if multiple:
            print("Predict {} @ x={} with {:.2f} confidence.".format(person, bbx,
                                                                     confidence))	
        else:
            print("Predict {} with {:.2f} confidence.".format(person, confidence))
        if isinstance(clf, GMM):
            dist = np.linalg.norm(rep - clf.means_[maxI])
            print("  + Distance from the mean: {}".format(dist))
	json += "{"+'"Prediction": "{}", "confidence": {}'.format(person, confidence)+"},"

    json += "]}"
    return json

