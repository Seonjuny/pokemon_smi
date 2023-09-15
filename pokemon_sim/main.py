#from load import DataLoad
from extract import file_po
from image import image_5
from model import im_model
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


from image import image_5


# 아래 image_name에서 원하는 포켓몬이름을 지정해주세요
image_name = 'zeraora'
image_5().get_similar_problem(image_name)

