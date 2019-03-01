import tensorflow as tf
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

#data variables
DATA_dir = 'C:\\Users\\yarne\\dropbox\\DeepLearning\\Data\\Petimages\\evaluate'
img_size = 50
n_img = 4

#create train_data
predict_data = []
def create_predict_data():
    path = DATA_dir
    for img in os.listdir(path):
        try:
            img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
            img_array = cv2.resize(img_array, (img_size, img_size))
            predict_data.append(img_array)
        except:
            pass

create_predict_data()
x = np.array(predict_data).reshape(-1, img_size, img_size, 1)
x = x/255.0
model = tf.keras.models.load_model('CNN_cat-dog.h5')
print(model.evaluate(x, [1 for i in range(n_img)]))
print((model.predict(x)-1)*100)
