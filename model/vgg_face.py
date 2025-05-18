from deepface.models.facial_recognition import VGGFace
from deepface.models.facial_recognition.VGGFace import VggFaceClient

from keras.src.applications.densenet import layers
from keras.src.layers import Dense, Dropout
from tensorflow.keras import Sequential
import tensorflow_addons as tfa
import tensorflow as tf

import numpy as np
import os
import cv2
class FacialRecognitionModel:
    def __init__(self,class_names=["ChauBui","Erik","HoaMinzy","KhoaPub","LamVlog","LanAnh","NguyenVietHoang","PhuongLy","SonTung","TranMinhHieu"]):
        self.class_name = class_names
        self.model = None


    def load_model(self):
        # Initialize VGGFace client and get base model
        client = VggFaceClient()
        base_model = client.model

        x = base_model.output
        x = Dense(2048, activation=None)(x)

        model = tf.keras.Model(inputs=base_model.input, outputs=x)

        # Freeze all layers except the last one
        for layer in model.layers[:-2]:
            layer.trainable = False


        # Compile model
        model.compile(optimizer='adam',
                      loss=tfa.losses.TripletSemiHardLoss(),
                      # metrics=['accuracy', tf.keras.metrics.Precision(thresholds=0.7)])
                      )

        self.model = model
        return self.model

    def get_model(self):
        if self.model is None:
            self.load_model()
        return self.model