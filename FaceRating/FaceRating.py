#!/usr/bin/python

import sys
import numpy as np
import cv2
import dlib
import itertools
from math import sqrt
from sklearn.model_selection import LeaveOneOut
from sklearn import decomposition
from sklearn.ensemble import RandomForestRegressor
from sklearn.externals import joblib


class FaceRating():

    def __init__(self, featuresPath='./data/features_ALL.txt',
                 ratingsPath='./data/ratings.txt'):

        self.features = np.loadtxt(featuresPath, delimiter=',')
        self.ratings = np.loadtxt(ratingsPath, delimiter=',')
        self.loo_index = LeaveOneOut().split(self.features, self.ratings)
        self.best_model = None

    def PCA(self, X_train, X_test):

        pca = decomposition.PCA(n_components=20)
        pca.fit(X_train)
        X_train, X_test = pca.transform(X_train), pca.transform(X_test)

        return X_train, X_test

    def getLandmarks(self, img_path):

        PREDICTOR_PATH = './model/shape_predictor_68_face_landmarks.dat'
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(PREDICTOR_PATH)
        face_img = cv2.imread(img_path)
        res = detector(face_img, 1)
        if len(res) >= 1:
            print("{} faces detected".format(len(res)))
        else:
            print('No faces detected')
            exit()
        with open('./results/landmarks.txt', 'w') as f:
            f.truncate()
            for face in res:
                landmarks = np.matrix(
                    [[p.x, p.y] for p in predictor(face_img, face).parts()])
                face_img = face_img.copy()
                for p in landmarks:
                    f.write(str(p[0, 0]) + ',' + str(p[0, 1]) + ',')
                f.write('\n')
            f.close()
        print('Get landmarks successfully!')
        return np.loadtxt('./results/landmarks.txt',
                          delimiter=',', usecols=range(136))

    def getFeatures(self, img_path):
        """
        Method for geting features refers to:
        http://www.learnopencv.com/computer-vision-for-predicting-facial-attractiveness/
        """
        a = [18, 22, 23, 27, 37, 40, 43, 46, 28,
             32, 34, 36, 5, 9, 13, 49, 55, 52, 58]
        combinations = itertools.combinations(a, 4)
        pointIndices1 = []
        pointIndices2 = []
        pointIndices3 = []
        pointIndices4 = []
        for combination in combinations:
            pointIndices1.append(combination[0])
            pointIndices2.append(combination[1])
            pointIndices3.append(combination[2])
            pointIndices4.append(combination[3])
            pointIndices1.append(combination[0])
            pointIndices2.append(combination[2])
            pointIndices3.append(combination[1])
            pointIndices4.append(combination[3])
            pointIndices1.append(combination[0])
            pointIndices2.append(combination[3])
            pointIndices3.append(combination[1])
            pointIndices4.append(combination[2])
        landmarks = self.getLandmarks(img_path)
        size = landmarks.shape
        rows = size[0] if len(size) > 1 else 1
        allFeatures = np.zeros((rows, len(pointIndices1)))
        for x in range(rows):
            landmarkCoordinates = landmarks[x, :] if rows != 1 else landmarks
            ratios = []
            for i in range(len(pointIndices1)):
                x1 = landmarkCoordinates[2 * (pointIndices1[i] - 1)]
                y1 = landmarkCoordinates[2 * pointIndices1[i] - 1]
                x2 = landmarkCoordinates[2 * (pointIndices2[i] - 1)]
                y2 = landmarkCoordinates[2 * pointIndices2[i] - 1]
                x3 = landmarkCoordinates[2 * (pointIndices3[i] - 1)]
                y3 = landmarkCoordinates[2 * pointIndices3[i] - 1]
                x4 = landmarkCoordinates[2 * (pointIndices4[i] - 1)]
                y4 = landmarkCoordinates[2 * pointIndices4[i] - 1]
                dist1 = sqrt((x1 - x2)**2 + (y1 - y2)**2)
                dist2 = sqrt((x3 - x4)**2 + (y3 - y4)**2)
                ratios.append(dist1 / (dist2 + 10 ** -5))
            allFeatures[x, :] = np.asarray(ratios)
        np.savetxt("./results/img_features.txt",
                   allFeatures, delimiter=',', fmt='%.04f')
        print("Generate Feature Successfully!")
        return allFeatures

    def train(self):

        tmp_score, tmp_model = -float('inf'), None
        for train_index, test_index in self.loo_index:
            X_train, X_test = self.features[
                train_index], self.features[test_index]
            y_train, y_test = self.ratings[
                train_index], self.ratings[test_index]
            X_train, X_test = self.PCA(X_train, X_test)

            model = RandomForestRegressor(
                n_estimators=50, max_depth=None,
                min_samples_split=10, random_state=0)
            model = model.fit(X_train, y_train)
            model_score = model.score(X_test, y_test)
            if tmp_score < model_score:
                tmp_score = model_score
                tmp_model = model
        self.best_model = tmp_model
        with open('./model/face_rating_model.pkl', 'wb') as f:
            joblib.dump(tmp_model, f, compress=1)

    def rating(self, img_path='./test_img/test.jpg'):

        if self.best_model:
            model = self.best_model
        else:
            with open('./model/face_rating_model.pkl', 'rb') as f:
                model = joblib.load(f)
        img_features = self.getFeatures(img_path)
        pca = decomposition.PCA(n_components=20)
        pca.fit(self.features)
        rows = (img_features.shape)[0] if len(img_features) > 1 else 1
        for x in range(rows):
            features = img_features[x, :] if rows != 1 else img_features
            transformed_features = pca.transform(features.reshape(1, -1))
            result = model.predict(transformed_features)
            print("The %d's person in the photo is rated: %d/5.0" %
                  (x + 1, result))


featuresPath = ratingsPath = None
if len(sys.argv) == 2:
    img_path = sys.argv[1]
elif len(sys.argv) == 4:
    img_path, featuresPath, ratingsPath = sys.argv[1], sys.argv[2], sys.argv[3]
else:
    print("Give the path to the directory containing the facial images "
          "as the first argument and then the other dataset and labels.\n"
          "For example, if you are in the FaceRating folder then "
          "execute this program by running:\n"
          "    ./face_landmark_detection.py ../test_img/test.jpg[ "
          "../data/features_ALL.txt ../data/ratings.txt]\n")
    exit()
if featuresPath and ratingsPath:
    model = FaceRating(featuresPath, ratingsPath)
    model.train()
    model.rating(img_path)
else:
    model = FaceRating()
    model.rating(img_path)
