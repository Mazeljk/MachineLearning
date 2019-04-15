import sys
import numpy as np
import dlib
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
        self.score = 0

    def PCA(self, X_train, X_test):

        pca = decomposition.PCA(n_components=20)
        pca.fit(X_train)
        X_train, X_test = pca.transform(X_train), pca.transform(X_test)

        return X_train, X_test

    def train(self):

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
            if self.score < model_score:
                self.score = model_score
                self.best_model = model
        joblib.dump(self.best_model,
                    './model/face_rating_model.pkl', compress=1)
        # TODO
        # TODO
        # TODO
        # TODO
