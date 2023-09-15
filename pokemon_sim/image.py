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
from PIL import Image
import matplotlib.pyplot as plt

from model import im_model

class image_5:
    
    def __init__(self):
        self.self = self
    

    def get_similar_problem(self,item_id):
        
        
        
        poket_cosine = im_model().df_model()
        
        
        sim_df = pd.DataFrame(poket_cosine[item_id].sort_values(ascending=False).reset_index())
        sim_df.columns = ['name_cr', 'similarity']
        sim_df = sim_df[sim_df['name_cr'] != item_id][:5]
        name_list1 = sim_df['name_cr']
        for name in name_list1:
            
            try:
                m = f"./archive (1)/images/images/{name}.png"
                img_color = Image.open(m)

                plt.imshow(img_color)
                plt.show()
            except:
                m = f"./archive (1)/images/images/{name}.jpg"
                img_color = Image.open(m)

                plt.imshow(img_color)
                plt.show()