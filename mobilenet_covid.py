import os
from glob import glob
from keras.initializers import glorot_uniform
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from sklearn.metrics import classification_report
import functools
import keras
from keras.layers.normalization import BatchNormalization
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_curve
import pandas as pd
import numpy as np
import keras.backend as K
import tensorflow as tf
from sklearn.metrics import roc_auc_score
import numpy
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.callbacks import TensorBoard
from keras.preprocessing.image import ImageDataGenerator
from keras import layers
from keras import Model
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.applications import MobileNet
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping,ModelCheckpoint
import random
from keras.layers import Dense, Dropout, GlobalAveragePooling2D,GlobalMaxPooling2D,DepthwiseConv2D,Concatenate
import matplotlib.pyplot as plt
import tensorflow
from at import augmented_conv2d
from sklearn.metrics import confusion_matrix
tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0,
                          write_graph=True, write_images=False)

f=augmented_conv2d
print(f,type(f))
X_train = np.load("D:/polynomial/covid19/new_version/new_224_224_train.npy")
y_train = np.load("D:/polynomial/covid19/new_version/new_train_labels.npy")
X_val = np.load("D:/polynomial/covid19/new_version/new_covid19_test.npy")

y_val = np.load("D:/polynomial/covid19/new_version/new_covid19_test_labels.npy")

print(X_train.shape, X_val.shape)
print(y_train.shape, y_val.shape)

y_train = to_categorical(y_train)
y_val = to_categorical(y_val)
print(y_train.shape)
print(y_val.shape)
print(y_val)

'''
def custom_activation(x):
    return 0.6275*K.exp(-(K.square((x-1.274)/0.5787)))
    '''

def custom_activation(x):
    return 11.91*K.exp(-K.square((x/100-396.4)/138.6)) + 0.3829*K.exp(-K.square(((x/100-105.8)/40.98)))


pre_trained_model = MobileNet(input_shape=(224,224, 3), include_top=False, weights="imagenet")
 #imports the mobilenet model and discards the last 1000 neuron layer.
for layer in pre_trained_model.layers:
    #print(layer.name)
    layer.trainable = False

#print(len(pre_trained_model.layers))
last_layer = pre_trained_model.get_layer('conv_pw_13_relu')
#print('last layer output shape:', last_layer.output_shape)
last_output = last_layer.output
x=GlobalAveragePooling2D()(last_output)
#x=Dense(512,activation='relu')(x) #we add dense layers so that the model can learn more complex functions and classify for better results.
#x = Dropout(0.5)(x)
x = layers.Dense(512, activation='relu')(x)
# Add a dropout rate of 0.7
x = layers.Dropout(0.5)(x)

x = layers.Dense(256, activation='relu')(x)
# Add a dropout rate of 0.7
x = layers.Dropout(0.5)(x)


#x = layers.Dense(128, activation='relu')(x)
# Add a dropout rate of 0.7
#x = layers.Dropout(0.5)(x)


# Add a final sigmoid layer for classification
x = layers.Dense(3, activation='softmax',name='visualized_layer')(x)

model=Model(pre_trained_model.input, x)
# Configure and compile the model
'''
smooth = 1.
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)
'''
#top3_acc = functools.partial(keras.metrics.top_k_categorical_accuracy, k=3)

#top3_acc.__name__ = 'top3_acc'

optimizer = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True)
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy','top_k_categorical_accuracy'])

model.summary()


train_datagen = ImageDataGenerator(rotation_range=60, width_shift_range=0.2, height_shift_range=0.2,
                                   shear_range=0.2, zoom_range=0.2, fill_mode='nearest')

train_datagen.fit(X_train)

val_datagen = ImageDataGenerator()
val_datagen.fit(X_val)

batch_size = 32
epochs = 3
history = model.fit_generator(train_datagen.flow(X_train,y_train, batch_size=batch_size),
                              epochs = epochs, validation_data = val_datagen.flow(X_val, y_val),
                              verbose = 1, steps_per_epoch=(X_train.shape[0] // batch_size),
                              validation_steps=(X_val.shape[0] // batch_size))


def identity_block(X, f, filters):
    """
    Implementation of the identity block as defined in Figure 3

    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network

    Returns:
    X -- output of the identity block, tensor of shape (n_H, n_W, n_C)
    """

    # defining name basis
    #conv_name_base = 'res' + str(stage) + block + '_branch'
    #bn_name_base = 'bn' + str(stage) + block + '_branch'

    # Retrieve Filters
    F1, F2, F3 = filters

    # Save the input value. You'll need this later to add back to the main path.
    X_shortcut = X
    print(X_shortcut)

    # First component of main path
    X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(1, 1), padding='valid',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    #extra


    # Second component of main path (≈3 lines)
    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)



    # Third component of main path (≈2 lines)
    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3)(X)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X


def convolutional_block(X, f, filters, s=2):
    """
    Implementation of the convolutional block as defined in Figure 4

    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    s -- Integer, specifying the stride to be used

    Returns:
    X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)
    """

    # defining name basis
    #conv_name_base = 'res' + str(stage) + block + '_branch'
    #bn_name_base = 'bn' + str(stage) + block + '_branch'

    # Retrieve Filters
    F1, F2, F3 = filters

    # Save the input value
    X_shortcut = X

    ##### MAIN PATH #####
    # First component of main path
    X = Conv2D(F1, (1, 1), strides=(s, s),  kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)

    # Second component of main path (≈3 lines)
    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)



    # Third component of main path (≈2 lines)
    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3)(X)

    ##### SHORTCUT PATH #### (≈2 lines)
    X_shortcut = Conv2D(filters=F3, kernel_size=(1, 1), strides=(s, s), padding='valid',
                        kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis=3)(X_shortcut)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X
'''for layer in model.layers[:20]:
    layer.trainable = False

for layer in model.layers[20:]:
    layer.trainable = True'''
for layer in model.layers[:12]:
    layer.trainable=True
for layer in model.layers[12:13]:
    layer.trainable=DepthwiseConv2D(64,strides=(2, 2),dilation_rate=(1,1))
for layer in model.layers[13:25]:
    layer.trainable=True
for layer in model.layers[25:26]:
    layer.trainable=DepthwiseConv2D(128,strides=(2, 2),dilation_rate=(1,1))
for layer in model.layers[26:38]:
    layer.trainable=True
for layer in model.layers[38:39]:
    layer.trainable=DepthwiseConv2D(256,strides=(2, 2),dilation_rate=(2,2))

for layer in model.layers[39:75]:
    layer.trainable=True
    #layer.trainable=identity_block(layer.output,3,[64, 64, 256], stage=1, block='a')
for layer in model.layers[75:76]:
    layer.trainable=DepthwiseConv2D(512,strides=(2, 2),dilation_rate=(2,2))
'''
for layer in model.layers[76:81]:
    layer.trainable=identity_block(layer.output,3,[128, 128, 512], stage=2, block='a')
    #layer.trainable=True'''
for layer in model.layers[76:78]:
    ca1=convolutional_block(layer.output, f = 3, filters = [64, 64, 256],s=1)
    a1=identity_block(ca1, 3, [64, 64, 256])
    b1=identity_block(a1,3,[64, 64, 256])
    #ba1=BatchNormalization()(b1)
    ac1= augmented_conv2d(b1, 512)
    ca2 = convolutional_block(ac1, f=3, filters=[128, 128, 512], s=1)
    a2 = identity_block(ca2, 3, [128, 128, 512])
    b2 = identity_block(a2, 3, [128, 128, 512])
    c2 = identity_block(b2, 3, [128, 128, 512])
    #ba2=BatchNormalization()(c2)
    ac2=augmented_conv2d(c2,512)
    ca3= convolutional_block(ac2, f=3, filters=[256, 256, 1024], s=1)
    a3 = identity_block(ca3, 3, [1256, 256, 1024])
    b3 = identity_block(a3, 3, [256, 256, 1024])
    c3 = identity_block(b3, 3, [256, 256, 1024])




    #layer.trainable=Concatenate([ca,a,b])

    layer.trainable=c3
    #print('man')
    layer.trainable=True
for layer in model.layers[78:81]:
    layer.trainable=True
for layer in model.layers[81:82]:

    #res2=identity_block(res1,3,[128, 128, 512], stage=3, block='b')
    #res3=identity_block(res2,3,[128, 128, 512], stage=3, block='c')

    a=DepthwiseConv2D(1024, strides=(2, 2),dilation_rate=(4,4))
    b=DepthwiseConv2D(1024, strides=(2, 2),dilation_rate=(8,8))
    c=DepthwiseConv2D(1024, strides=(2, 2),dilation_rate=(16,16))
    d=Concatenate([a,b])
    layer.trainable=Concatenate([d,c])
for layer in model.layers[82:]:
    layer.trainable=True


def auroc(y_true, y_pred):
    return tf.py_func(roc_auc_score, (y_true, y_pred), tf.double)


optimizer = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['acc','top_k_categorical_accuracy'])

#save_best = ModelCheckpoint(filepath='D:/polynomial/covid19/inception_dl4_val_new/checkpoint-{val_acc:.4f}.h5',monitor='val_acc',mode='auto',save_best_only=True)
save_best=ModelCheckpoint('dilated_MobileNet.h5', verbose=1,monitor='val_acc',save_best_only=True, save_weights_only=True)
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', patience=5, verbose=1, factor=np.sqrt(0.1),
                                            min_lr=0.5e-6, cooldown=0,mode='auto')
callbacks=[learning_rate_reduction,tensorboard,save_best]

model.summary()

batch_size = 32
epochs = 22
history = model.fit_generator(train_datagen.flow(X_train,y_train, batch_size=batch_size),
                              epochs = epochs, validation_data = val_datagen.flow(X_val, y_val),
                              verbose = 1, steps_per_epoch=(X_train.shape[0] // batch_size),
                              validation_steps=(X_val.shape[0] // batch_size),
                              callbacks=callbacks)
model.load_weights('dilated_MobileNet.h5')
loss_val, acc_val,top5 = model.evaluate(X_val, y_val, verbose=1)
print("Validation: accuracy = %f  ;  loss_v = %f" % (acc_val, loss_val))
print('top5 validate accuracy: ',top5)
#print('top3 validate accuracy: ',top3)


X_test = np.load("D:/polynomial/covid19/new_version/new_covid19_test.npy")
y_test = np.load("D:/polynomial/covid19/new_version/new_covid19_test_labels.npy")
print("1st ytest: ",y_test)
#print("1st y_test",y_test)
target_names = ['Covid19', 'Normal','Pneumonia']
#print("1st y_test",y_test, y_test.shape)
y_pred = model.predict(X_test)
print("4th y_pred",y_pred.astype(float),y_pred.shape,type(y_pred))
np.set_printoptions(precision=3, suppress=True)
print("1: ",y_pred)
np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
print("2: ",y_pred)

y_classes = y_pred.argmax(axis=-1)
print("y classes1:",y_classes,y_classes.shape)

#print("5th y_pred",y_classes,y_classes.shape)
print(classification_report(y_test, y_classes, target_names=target_names))
y_test = to_categorical(y_test)
#print("2nd y_test",y_test)

print("2nd ytest: ",y_test)

loss_test, acc_test,top5 = model.evaluate(X_test, y_test, verbose=1)
print("Test: accuracy = %f  ;  loss = %f" % (acc_test, loss_test))
print("top5 test accuracy: ",top5)
#print("top3 test accuracy: ",top3)


pre=model.predict(X_test,verbose=1)
#print("pre",pre)
n_values=3
c = np.eye(n_values, dtype=int)[np.argmax(pre, axis=1)]
#print('one-hot-encoding',c)
cm = confusion_matrix(y_test.argmax(axis=1), c.argmax(axis=1))
print("confusion matrix",cm)
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print('new_cm',cm)
print('acc',cm.diagonal())

#model.save("InceptionV3.h5")

# Retrieve a list of accuracy results on training and test data
# sets for each training epoch
acc = history.history['acc']
val_acc = history.history['val_acc']

# Retrieve a list of list results on training and test data
# sets for each training epoch
loss = history.history['loss']
val_loss = history.history['val_loss']

# Get number of epochs
epochs = range(len(acc))
'''
# Plot training and validation accuracy per epoch
plt.plot(epochs, acc, label = "training")
plt.plot(epochs, val_acc, label = "validation")
plt.legend(loc="upper left")
plt.title('Training and validation accuracy')
plt.show()
plt.figure()

# Plot training and validation loss per epoch
plt.plot(epochs, loss, label = "training")
plt.plot(epochs, val_loss, label = "validation")
plt.legend(loc="upper right")
plt.title('Training and validation loss')
plt.show()
'''


#auc = roc_auc_score(y_test, y_pred)
#print('AUC: %.3f' % auc)

from vis.visualization import visualize_cam, overlay
from vis.utils import utils
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
from keras import activations

# Find the index of the to be visualized layer above
layer_index = utils.find_layer_idx(model, 'visualized_layer')

# Swap softmax with linear
model.layers[layer_index].activation = activations.linear
model = utils.apply_modifications(model)

# Numbers to visualize
indices_to_visualize = [7,12,17,20,21,22,33,34, 36,38,40,41,42,46, 83, 112,7,17,20,21,32,33,284,285,294,297]
path='D:/Covid19/gradcam/dense/'
# Visualize
for index_to_visualize in indices_to_visualize:
  # Get input
  input_image = X_test[index_to_visualize]
  input_class = np.argmax(y_test[index_to_visualize])
  # Matplotlib preparations
  fig, axes = plt.subplots(1, 2)
  # Generate visualization
  visualization = visualize_cam(model, layer_index, filter_indices=input_class, seed_input=input_image)
  axes[0].imshow(input_image[..., 0], cmap='gray')
  axes[0].set_title('Input')
  #axes[1].imshow(visualization)
  #axes[1].set_title('Grad-CAM')
  heatmap = np.uint8(cm.jet(visualization)[..., :3] * 255)
  original = np.uint8(cm.gray(input_image[..., 0])[..., :3] * 255)
  axes[1].imshow(overlay(heatmap, original))
  axes[1].set_title('Grad-CAM')
  fig.suptitle('RAM-Net = {}'.format(input_class))
  plt.savefig(os.path.join(path, '{}.png'.format(index_to_visualize)))
  #plt.show()

layer_index1 = utils.find_layer_idx(model, 'conv_pw_13_relu')

path1='D:/Covid19/gradcam/relu/'
# Visualize
for index_to_visualize in indices_to_visualize:
  # Get input
  input_image = X_test[index_to_visualize]
  input_class = np.argmax(y_test[index_to_visualize])
  # Matplotlib preparations
  fig, axes = plt.subplots(1, 2)
  # Generate visualization
  visualization = visualize_cam(model, layer_index1, filter_indices=input_class, seed_input=input_image)
  axes[0].imshow(input_image[..., 0], cmap='gray')
  axes[0].set_title('Input')
  # axes[1].imshow(visualization)
  # axes[1].set_title('Grad-CAM')
  heatmap = np.uint8(cm.jet(visualization)[..., :3] * 255)
  original = np.uint8(cm.gray(input_image[..., 0])[..., :3] * 255)
  axes[1].imshow(overlay(heatmap, original))
  axes[1].set_title('Grad-CAM')
  fig.suptitle('RAM-Net = {}'.format(input_class))
  plt.savefig(os.path.join(path1, '{}.png'.format(index_to_visualize)))
  #plt.show()

layer_index2 = utils.find_layer_idx(model, 'conv_pw_13')
path2='D:/Covid19/gradcam/conv/'
for index_to_visualize in indices_to_visualize:
  # Get input
  input_image = X_test[index_to_visualize]
  input_class = np.argmax(y_test[index_to_visualize])
  # Matplotlib preparations
  fig, axes = plt.subplots(1, 2)
  # Generate visualization
  visualization = visualize_cam(model, layer_index2, filter_indices=input_class, seed_input=input_image)
  axes[0].imshow(input_image[..., 0], cmap='gray')
  axes[0].set_title('Input')
  # axes[1].imshow(visualization)
  # axes[1].set_title('Grad-CAM')
  heatmap = np.uint8(cm.jet(visualization)[..., :3] * 255)
  original = np.uint8(cm.gray(input_image[..., 0])[..., :3] * 255)
  axes[1].imshow(overlay(heatmap, original))
  axes[1].set_title('Grad-CAM')
  fig.suptitle('RAM-Net = {}'.format(input_class))
  plt.savefig(os.path.join(path2, '{}.png'.format(index_to_visualize)))
  #plt.show()

layer_index3 = utils.find_layer_idx(model, 'conv_pw_13_bn')
path3='D:/Covid19/gradcam/conv_bn/'
for index_to_visualize in indices_to_visualize:
  # Get input
  input_image = X_test[index_to_visualize]
  input_class = np.argmax(y_test[index_to_visualize])
  # Matplotlib preparations
  fig, axes = plt.subplots(1, 2)
  # Generate visualization
  visualization = visualize_cam(model, layer_index3, filter_indices=input_class, seed_input=input_image)
  axes[0].imshow(input_image[..., 0], cmap='gray')
  axes[0].set_title('Input')
  # axes[1].imshow(visualization)
  # axes[1].set_title('Grad-CAM')
  heatmap = np.uint8(cm.jet(visualization)[..., :3] * 255)
  original = np.uint8(cm.gray(input_image[..., 0])[..., :3] * 255)
  axes[1].imshow(overlay(heatmap, original))
  axes[1].set_title('Grad-CAM')
  fig.suptitle('RAM-Net = {}'.format(input_class))
  plt.savefig(os.path.join(path3, '{}.png'.format(index_to_visualize)))
  #plt.show()

from vis.visualization import visualize_activation
from vis.utils import utils
import matplotlib.pyplot as plt
layer_index = utils.find_layer_idx(model, 'visualized_layer')


# Swap softmax with linear
model.layers[layer_index].activation = activations.linear
model = utils.apply_modifications(model)

# Classes to visualize
classes_to_visualize = [ 0, 1, 2 ]
classes = {
  0: 'COVID-19',
  1: 'Normal',
  2: 'Pneumonia'
}
path3='D:/Covid19/gradcam/layer1/'
path4='D:/Covid19/gradcam/layer2/'
path5='D:/Covid19/gradcam/layer3/'
path6='D:/Covid19/gradcam/layer4/'
# Visualize
for number_to_visualize in classes_to_visualize:
  visualization = visualize_activation(model, layer_index, filter_indices=number_to_visualize, input_range=(0., 1.))
  plt.imshow(visualization)
  plt.title('RAM-Net = {}'.format(classes[number_to_visualize]))
  plt.savefig(os.path.join(path3, '{}.png'.format(number_to_visualize)))
  #plt.show()
layer_index1=utils.find_layer_idx(model,'conv_pw_13_relu')
for number_to_visualize in classes_to_visualize:
  visualization = visualize_activation(model, layer_index1, filter_indices=number_to_visualize, input_range=(0., 1.))
  plt.imshow(visualization)
  plt.title('RAM-Net = {}'.format(classes[number_to_visualize]))
  plt.savefig(os.path.join(path4, '{}.png'.format(number_to_visualize)))
  #plt.show()

layer_index2=utils.find_layer_idx(model,'conv_pw_13')
for number_to_visualize in classes_to_visualize:
  visualization = visualize_activation(model, layer_index2, filter_indices=number_to_visualize, input_range=(0., 1.))
  plt.imshow(visualization)
  plt.title('RAM-Net = {}'.format(classes[number_to_visualize]))
  plt.savefig(os.path.join(path5, '{}.png'.format(number_to_visualize)))
  #plt.show()
'''
layer_index3=utils.find_layer_idx(model,'conv_pw_13_bn')
for number_to_visualize in classes_to_visualize:
  visualization = visualize_activation(model, layer_index3, filter_indices=number_to_visualize, input_range=(0., 1.))
  plt.imshow(visualization)
  plt.title('RAM-Net = {}'.format(classes[number_to_visualize]))
  plt.savefig(os.path.join(path6, '{}.png'.format(number_to_visualize)))
  #plt.show()
'''