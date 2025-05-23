
# import the necessary packages
import tensorflow as tf
import os
# path to training and testing data
TRAIN_DATASET = r"D:\Python plus\AI_For_CV\dataset\face_dataset\train"
VAL_DATASET = r"D:\Python plus\AI_For_CV\dataset\face_dataset\val"
CHECK_POINT = r"D:\Python plus\AI_For_CV\script\face_recognition\checkpoint\v1\model.h5"
# model input image size
IMAGE_SIZE = (224, 224)
# batch size and the buffer size
BATCH_SIZE = 256
BUFFER_SIZE = BATCH_SIZE * 2
# define autotune
AUTO = tf.data.AUTOTUNE
# define the training parameters
LEARNING_RATE = 0.0001
STEPS_PER_EPOCH = 50
VALIDATION_STEPS = 10
EPOCHS = 10
# Threshold for verification
THRESHOLD = 0.6

