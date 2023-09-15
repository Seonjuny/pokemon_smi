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

class file_po:
    
    def __init__(self):
        self.self = self
        
    def poket_im(self):
    
        img_dir = "./archive (1)/images/images" # Dossier des images
        dim = 224 # 이미지 크기 지정
        X = [] # 이미지 데이터 리스트
        y = [] # 포켓몬 이름 리스트
        data_path = os.path.join(img_dir,'*g')
        files = glob.glob(data_path) # data_path에 해당한는 파일 목록을 가져옴
        n=0
        for f1 in files:
            try:
                n=n+1
                img = cv2.imread(f1) # 이미지 파일을 읽어드림
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # 이미지를 rgb값으로 변환(혹 rgb값이 아닌것이 있을까봐)
                img = cv2.resize(img, (dim,dim)) # 이미지 크기를 224x224으로 변경
                X.append(np.array(img)) # 이미지를 배열로 변환
                y.append(Path(f1).stem) # 이미지 파일 이름을 추가
            except :
                continue
        print(n,' images lues')
        X = np.array(X)
        y = np.array(y)

        return X,y