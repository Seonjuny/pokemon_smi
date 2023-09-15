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


from extract import file_po

class im_model:
    
    def __init__(self):
        self.self = self
        self.X, self.y = file_po().poket_im()
    
    
        
    def df_model(self):

        X = self.X
        y = self.y

        print("#########------- 여기부터 시작합니다 -------########")
        base_model = vgg16.VGG16(weights='imagenet')
        vgg16_model = Model(inputs=base_model.input, outputs=base_model.get_layer("fc1").output )
        X_model = vgg16_model.predict(X)
        df = pd.DataFrame(X_model)
        df.index = y
        poket_cosine = cosine_similarity(df)
        poket_cosine = pd.DataFrame(data = poket_cosine, index = df.index, columns = df.index)
        
        return poket_cosine