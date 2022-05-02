
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
import keras.metrics as metrics
from keras.layers import AveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import BatchNormalization
from keras.layers import Dropout
from sklearn.metrics import classification_report
from keras.optimizers import Adam
import numpy as np
import tensorflow as tf
from keras import regularizers
from keras.layers.advanced_activations import LeakyReLU

path_drive = 'P://fm//py//images'

# Initialising the CNN
Classifier = Sequential()
# Step 1 - Convolution
Classifier.add(Convolution2D(32, 3, 3, input_shape = (64,64,3)))
Classifier.add(LeakyReLU(alpha=0.1))
# Classifier.add(BatchNormalization()) #no need

# Step 2 - Pooling
Classifier.add(MaxPooling2D(pool_size = (2,2)))
Classifier.add(Dropout(0.25))

# Classifier.add(Dense(64, activation= 'relu'))
# # second layer
Classifier.add(Convolution2D(64, 3, 3))
Classifier.add(LeakyReLU(alpha=0.1))
# Classifier.add(BatchNormalization())
Classifier.add(MaxPooling2D(pool_size = (2,2)))
# Classifier.add(Dropout(0.5))

# Classifier.add(Dense(64,input_dim = 64,
#                      kernel_regularizer = regularizers.l2(0.00001),
#                      activity_regularizer = regularizers.l1(0.00001)))

# #third layer
# Classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
# #Classifier.add(BatchNormalization())
# Classifier.add(MaxPooling2D(pool_size = (2,2)))
# Classifier.add(Dropout(0.25))

#forth layer
# Classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
# #Classifier.add(BatchNormalization())
# Classifier.add(MaxPooling2D(pool_size = (2,2)))


# Step 3 - Flattening
Classifier.add(Flatten())

# Step 4 - Full connection
#output_dim = 128

Classifier.add(Dense( 128, activation = 'linear'))
Classifier.add(Dropout(0.1,name='Dropout_Regularization')) #dropout=1 for fixing last epoch fluctuation
Classifier.add(Dense( 7, activation = 'softmax'))

# Compiling the CNN
opt = Adam(lr=0.0001)
# cross_entropy = tf.reduce_sum(labels*tf.log(tf.clip_by_value(logits,1e-10,1.0))kullback_leibler_divergence)
Classifier.compile(optimizer = opt, loss = 'categorical_crossentropy',
                   metrics=['accuracy'])

# [metrics.mae, metrics.categorical_accuracy,metrics.categorical_crossentropy, metrics.binary_accuracy,metrics.top_k_categorical_accuracy,

# Part 2 - Fitting the CNN to the images
 

train_datagen = ImageDataGenerator( featurewise_center=False)

# train_datagen = ImageDataGenerator(
#                                     featurewise_center=False,
#                                     samplewise_center=False,
#                                     featurewise_std_normalization=False,
#                                     samplewise_std_normalization=False,
#                                     zca_whitening=False,
#                                     zca_epsilon=1e-06,
#                                     rotation_range=1,
#                                     width_shift_range=0.0,
#                                     height_shift_range=0.0,
#                                     brightness_range=None,
#                                     shear_range=0.2,
#                                     zoom_range=0.2,
#                                     channel_shift_range=0.0,
#                                     fill_mode="nearest",
#                                     cval=0.0,
#                                     horizontal_flip=False,
#                                     vertical_flip=False,
#                                     rescale=1./255,
#                                     preprocessing_function=None,
#                                     data_format=None,
#                                     validation_split=0.0,
#                                     dtype=None)


test_datagen = ImageDataGenerator( featurewise_center=False)

# #test_datagen = ImageDataGenerator(
#                                     featurewise_center=False,
#                                     samplewise_center=False,
#                                     featurewise_std_normalization=False,
#                                     samplewise_std_normalization=False,
#                                     zca_whitening=False,
#                                     zca_epsilon=1e-06,
#                                     rotation_range=0,
#                                     width_shift_range=0.0,
#                                     height_shift_range=0.0,
#                                     brightness_range=None,
#                                     shear_range=0.2,
#                                     zoom_range=0.2,
#                                     channel_shift_range=0.0,
#                                     fill_mode="nearest",
#                                     cval=0.0,
#                                     horizontal_flip=False,
#                                     vertical_flip=False,
#                                     rescale=1.0/255,
#                                     preprocessing_function=None,
#                                     data_format=None,
#                                     validation_split=0.0,
#                                     dtype=None)


# train_set = train_datagen.flow_from_directory(path_drive + '/Train',
#                                               target_size = (64,64),
#                                               batch_size = 32,
#                                               class_mode = 'categorical', shuffle=True)


train_set = train_datagen.flow_from_directory(path_drive + '//Train',
                                              target_size=(64,64),
                                                  color_mode="rgb",
                                                  classes=['lie down','fall','bend', 'run', 'sitdown','standup','walk'],
                                                  class_mode="categorical",
                                                  batch_size=32,
                                                  shuffle=False,
                                                  seed=None,
                                                  save_to_dir=None,
                                                  save_prefix="",
                                                  save_format="jpg",
                                                  follow_links=False,
                                                  subset=None,
                                                  interpolation="nearest")



# test_set = test_datagen.flow_from_directory(path_drive + '/Test',
#                                             target_size =(64,64),
#                                             batch_size =32,
#                                             class_mode ='categorical')


test_set = test_datagen.flow_from_directory( path_drive + '//Test',
                                            target_size=(64,64),
                                            color_mode="rgb",
                                            classes=['lie down','fall','bend', 'run', 'sitdown','standup','walk'],
                                            class_mode="categorical",
                                            batch_size=32,
                                            shuffle=False,
                                            seed=None,
                                            save_to_dir=None,
                                            save_prefix="",
                                            save_format="jpg",
                                            follow_links=False,
                                            subset=None,
                                            interpolation="nearest")


from keras.callbacks import ModelCheckpoint

checkpoint = ModelCheckpoint("model_weights.h5",monitor='val_accuracy',verbose=1,save_best_only=True,mode='max')
callbacks_list=[checkpoint]

history = Classifier.fit_generator(train_set,
                                     epochs = 150,
                                     validation_data = test_set,
                                     callbacks=[checkpoint])


# history = Classifier.fit_generator(train_set,
#                                      samples_per_epoch = 250,
#                                      nb_epoch = 180,
#                                      validation_data = test_set,
#                                      nb_val_samples = 189,
#                                    callbacks=[checkpoint])


#Save and serialize model structure to JSON
Classifier_json = Classifier.to_json()
with open("Classifier.json","w") as json_file:
    json_file.write(Classifier_json)

# plot the evolution of Loss and Acuracy
import matplotlib.pyplot as plt

plt.figure(figsize=(20, 10))
plt.subplot(1, 2, 1)
plt.suptitle('Optimizer : Adam', fontsize=10)
plt.ylabel('Loss', fontsize=16)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend(loc='upper right')

plt.subplot(1,2,2)
plt.ylabel('Accuracy', fontsize=16)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend(loc='lower right')
plt.show()

# show the confusion matrix of our predictions

# compute predictions
predictions = Classifier.predict_generator(generator=test_set)
y_pred = [np.argmax(probas) for probas in predictions]
y_test = test_set.classes
class_names = test_set.class_indices.keys()

from sklearn.metrics import confusion_matrix
import itertools


def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues):
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(10,10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i,j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i,j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


# compute confusion matrix
cnf_matrix = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)

# plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, title='Normalized confusion matrix')
plt.show()

Classifier.summary()