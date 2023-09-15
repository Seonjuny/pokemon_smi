from data.inputs import Input
import tensorflow as tf
import tensorflow.keras as keras
from sklearn.model_selection import train_test_split
import os
import cv2
import glob
import numpy as np
from pathlib import Path
from tensorflow.keras.applications import vgg16, resnet50
from tensorflow.keras.models import Model
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


class DataLoad:
    def __init__(self, reshape_dim:int, test_size:float=0.2):
        self.reshape_dim = reshape_dim
        self.test_size = test_size
    def _load_data(self):
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        x_train = self._reshape(x_train)
        x_test = self._reshape(x_test)
        return (x_train, y_train), (x_test, y_test)
    def _reshape(self, data):
        return data.reshape(-1, self.reshape_dim, self.reshape_dim, 1)
    def _train_test_split(self, dataset):
        return train_test_split(*dataset, test_size=self.test_size)
    def _normalize(self, data):
        return data/255.0
    def load(self):
        train, (x_test, y_test) = self._load_data()
        x_train, x_val, y_train, y_val = self._train_test_split(train)
        # 노멀라이즈
        x_train = self._normalize(x_train)
        x_val = self._normalize(x_val)
        x_test = self._normalize(x_test)
        return (x_train, y_train), (x_val, y_val), (x_test, y_test)
    



    
    