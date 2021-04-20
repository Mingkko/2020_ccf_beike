import numpy as np
import config
import tensorflow as tf
import data_loader
import model
import pandas as pd
from tensorflow.keras.optimizers import Adam

#load data
TRAIN_PATH = './train/train.csv'
TEST_PATH = './test/test.csv'

df_train = pd.read_csv(TRAIN_PATH,sep=' ')
df_test = pd.read_csv(TEST_PATH,sep=' ')

train_data = data_loader.get_dataloader(df_train)
test_data = data_loader.get_dataloader(df_test)

#define loss and optimizer
loss_fn = tf.keras.losses.BinaryCrossentropy()
optimizer = Adam(learning_rate=config.LEARNING_RATE)

train_loss = tf.keras.metrics.Mean(name = 'train_loss')
train_accuracy = tf.keras.metrics.BinaryAccuracy(name='train_accuracy')

val_loss = tf.keras.metrics.Mean(name = 'val_loss')
val_accuracy = tf.keras.metrics.BinaryAccuracy(name='val_accuracy')


#load model
model = model.mymodel(conf=model.model_config)
print(model.summary())
#
# @tf.function
# def train_step(ids,masks,type_ids,label):
#     with tf.GradientTape() as tape:
#         predictions = model(ids= ids,masks = masks,token_type_ids = type_ids)
#         loss = loss_fn(y_true = label, y_pred=predictions)
#     gradients = tape.gradient(loss,model.trainable_variables)
#     optimizer.apply_gradients()