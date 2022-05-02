
from numpy import mean
from numpy import std
from numpy import dstack
from pandas import read_csv
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import TimeDistributed
from keras.layers.convolutional import Conv1D
from keras.layers import LSTM
from keras.layers.convolutional import MaxPooling1D
from keras.layers import BatchNormalization
from keras.utils import to_categorical
import numpy as np 
import tensorflow as tf
import glob
import os
import csv
from keras.optimizers import Adam, Nadam, SGD
from keras import regularizers
from keras.layers.advanced_activations import LeakyReLU
from tensorflow_addons.optimizers import CyclicalLearningRate

class CSIModelConfig:
    """
    class for Human Activity Recognition ("lie down", "fall", "bend", "run", "sitdown", "standup", "walk")
    Using CSI (Channel State Information)
       I edited the code provided on https://github.com/ludlows.
    Args:
        win_len   :  integer (500 default) window length for batching sequence
        step      :  integer (200  default) sliding window by this step
        thrshd    :  float   (0.6  default) used to check if the activity is intensive inside a window
        downsample:  integer >=1 (2 default) downsample along the time axis
    """
    def __init__(self, win_len=300, step=50, thrshd=0.6):
        self._win_len = win_len
        self._step = step
        self._thrshd = thrshd
        self._labels = ("lie down", "fall", "bend", "run", "sitdown", "standup", "walk")
        # self._downsample = downsample

    def preprocessing(self, raw_folder, save=False):
        """
        Returns the Numpy Array for training within the format of (X_lable1, y_label1, ...., X_label7, y_label7)
        Args:
            raw_folder: the folder containing raw CSI 
            save      : choose if save the numpy array
        """
        numpy_tuple = extract_csi(raw_folder, self._labels, save, self._win_len, self._thrshd, self._step)
        # if self._downsample > 1:
        #     return tuple([v[:, ::self._downsample,...] if i%2 ==0 else v for i, v in enumerate(numpy_tuple)])
        return numpy_tuple
    
    def load_csi_data_from_files(self, np_files):
        """
        Returns the Numpy Array for training within the format of (X_lable1, y_label1, ...., X_label7, y_label7)
        Args:
            np_files: ('x_lie_down.npz', 'x_fall.npz', 'x_bend.npz', 'x_run.npz', 'x_sitdown.npz', 'x_standup.npz', 'x_walk.npz')
        """
        if len(np_files) != 7:
            raise ValueError('There should be 7 numpy files for lie down, fall, bend, run, sitdown, standup, walk.')
        x = [np.load(f)['arr_0'] for f in np_files]
        # if self._downsample > 1:
        #     x = [arr[:,::self._downsample, :] for arr in x]
        y = [np.zeros((arr.shape[0], len(self._labels))) for arr in x]
        numpy_list = []
        for i in range(len(self._labels)):
            y[i][:,i] = 1
            numpy_list.append(x[i])
            numpy_list.append(y[i])
        return tuple(numpy_list)







def extract_csi(raw_folder, labels, save=False, win_len=300, thrshd=0.6, step=50):
    """
    Return List of Array in the format of [X_label1, y_label1, X_label2, y_label2, .... X_Label7, y_label7]
    Args:
        raw_folder: the folder path of raw CSI csv files, input_* annotation_*
        labels    : all the labels existing in the folder
        save      : boolean, choose whether save the numpy array 
        win_len   :  integer, window length
        thrshd    :  float,  determine if an activity is strong enough inside a window
        step      :  integer, sliding window by step
    """
    ans = []
    for label in labels:
        feature_arr, label_arr = extract_csi_by_label(raw_folder, label, labels, save, win_len, thrshd, step)
        ans.append(feature_arr)
        ans.append(label_arr)
    return tuple(ans)




def extract_csi_by_label(raw_folder, label, labels, save=False, win_len=300, thrshd=0.6, step=50):
    """
    Returns all the samples (X,y) of "label" in the entire dataset
    Args:
        raw_folder: The path of Dataset folder
        label    : str, could be one of labels
        labels   : list of str, ['lie down', 'fall', 'bend', 'run', 'sitdown', 'standup', 'walk']
        save     : boolean, choose whether save the numpy array 
        win_len  :  integer, window length
        thrshd   :  float,  determine if an activity is strong enough inside a window
        step     :  integer, sliding window by step
    """
    print('Starting Extract CSI for Label {}'.format(label))
    label = label.lower()
    if not label in labels:
        raise ValueError("The label {} should be among 'lie down','fall','bend','run','sitdown','standup','walk'".format(labels))
    
    data_path_pattern = os.path.join(raw_folder,label, 'user_*' + label + '*.csv')
    input_csv_files = sorted(glob.glob(data_path_pattern))
    # annot_csv_files = [os.path.basename(fname).replace('user_', 'annotation_user') for fname in input_csv_files]
    # annot_csv_files = [os.path.join(raw_folder, label, fname) for fname in annot_csv_files]
    annot_csv_files = os.path.join(raw_folder,label, 'Annotation_user_*' + label + '*.csv')
    annot_csv_files = sorted(glob.glob(annot_csv_files))
    feature = []
    index = 0
    for csi_file, label_file in zip(input_csv_files, annot_csv_files):
        index += 1
        if not os.path.exists(label_file):
            print('Warning! Label File {} doesn\'t exist.'.format(label_file))
            continue
        feature.append(merge_csi_label(csi_file, label_file, win_len=win_len, thrshd=thrshd, step=step))
        print('Finished {:.2f}% for Label {}'.format(index / len(input_csv_files) * 100,label))
    
    feat_arr = np.concatenate(feature, axis=0)
    if save:
        np.savez_compressed("X_{}.npz".format(label), feat_arr)
    # one hot
    feat_label = np.zeros((feat_arr.shape[0], len(labels)))
    feat_label[:, labels.index(label)] = 1
    return feat_arr, feat_label


def train_valid_split(numpy_tuple, train_portion=0.8, seed=200):
    """
    Returns Train and Valid Datset with the format of (x_train, y_train, x_valid, y_valid),
    where x_train and y_train are shuffled randomly.

    Args:
        numpy_tuple  : tuple of numpy array: (x_lie_down, x_fall, x_bend, x_run, x_sitdown, x_standup, x_walk)
        train_portion: float, range (0,1)
        seed         : random seed
    """
    np.random.seed(seed=seed)
    x_train = []
    x_valid = []
    y_valid = []
    y_train = []

    for i, x_arr in enumerate(numpy_tuple):
        index = np.random.permutation([i for i in range(x_arr.shape[0])])
        split_len = int(train_portion * x_arr.shape[0])
        x_train.append(x_arr[index[:split_len], ...])
        tmpy = np.zeros((split_len,7))
        tmpy[:, i] = 1
        y_train.append(tmpy)
        x_valid.append(x_arr[index[split_len:],...])
        tmpy = np.zeros((x_arr.shape[0]-split_len,7))
        tmpy[:, i] = 1
        y_valid.append(tmpy)
    
    x_train = np.concatenate(x_train, axis=0)
    y_train = np.concatenate(y_train, axis=0)
    x_valid = np.concatenate(x_valid, axis=0)
    y_valid = np.concatenate(y_valid, axis=0)

    index = np.random.permutation([i for i in range(x_train.shape[0])])
    x_train = x_train[index, ...]
    y_train = y_train[index, ...]
    return x_train, y_train, x_valid, y_valid

def merge_csi_label(csifile, labelfile, win_len=300, thrshd=0.6, step=50):
    """
    Merge CSV files into a Numperrory Array  X,  csi amplitude feature
    Returns Numpy Array X, Shape(Num, Win_Len, 90)
    Args:
        csifile  :  str, csv file containing CSI data
        labelfile:  str, csv fiel with activity label 
        win_len  :  integer, window length
        thrshd   :  float,  determine if an activity is strong enough inside a window
        step     :  integer, sliding window by step
    """
    activity = []
    with open(labelfile, 'r') as labelf:
        reader = csv.reader(labelf)
        for line in reader:
            label  = line[0]
            if label == 'NoActivity':
                activity.append(0)
            else:
                activity.append(1)
    activity = np.array(activity)
    csi = []
    with open(csifile, 'r') as csif:
        reader = csv.reader(csif)
        for line in reader:
            line_array = np.array([float(v) for v in line])
            # extract the amplitude only
            line_array = line_array[0:52]
            csi.append(line_array[np.newaxis,...])
    csi = np.concatenate(csi, axis=0)
    assert(csi.shape[0] == activity.shape[0])
    # screen the data with a window
    index = 0
    feature = []
    while index + win_len <= csi.shape[0]:
        cur_activity = activity[index:index+win_len]
        if np.sum(cur_activity)  <  thrshd * win_len:
            index += step
            continue
        cur_feature = np.zeros((1, win_len, 52))
        cur_feature[0] = csi[index:index+win_len, :]
        feature.append(cur_feature)
        index += step
    return np.concatenate(feature, axis=0)

cfg = CSIModelConfig(win_len=300, step=50, thrshd=0.6)
numpy_tuple = cfg.preprocessing('CSV/', save=True)

x_lie_down, y_lie_down, x_fall, y_fall, x_bend, y_bend, x_run, y_run, x_sitdown, y_sitdown, x_standup, y_standup, x_walk, y_walk = numpy_tuple
x_train, y_train, x_valid, y_valid = train_valid_split((x_lie_down, x_fall, x_bend, x_run, x_sitdown, x_standup, x_walk),train_portion=0.8, seed=200)




# fit and evaluate a model

verbose, epochs, batch_size = 0, 300, 64
n_timesteps, n_features, n_outputs = x_train.shape[1], x_train.shape[2], y_train.shape[1]
model = Sequential()
model.add(LSTM(128, input_shape=(n_timesteps,n_features)))
# model.add(LSTM(150, input_shape=(n_timesteps,n_features),return_sequences=True)) use reurn sequence if u add more lstm
# model.add(LSTM(64))
# model.add(Dropout(0.25))
model.add(Dense(100, activation='relu'))
# model.add(Dropout(0.25))
model.add(Dense(n_outputs, activation='softmax'))
opt= Adam(lr=1e-4)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

#if you want to train conv1d
# model = Sequential()
# model.add(Conv1D(filters=32, kernel_size=4,padding='same', activation='relu', input_shape=(n_timesteps,n_features),kernel_initializer='random_normal'))
# model.add(MaxPooling1D(pool_size=2))
# model.add(Dropout(0.25))
# model.add(Conv1D(filters=64, kernel_size=4,padding='same', activation='relu' ,kernel_initializer='random_normal'))
# model.add(MaxPooling1D(pool_size=2))
# model.add(BatchNormalization())

# model.add(Flatten())
# model.add(Dense(128, activation='relu',kernel_regularizer = regularizers.l2(0.0001)))
# model.add(Dropout(0.5))
# model.add(Dense(n_outputs, activation='softmax', kernel_initializer='random_normal'))

# cyclical_learning_rate = CyclicalLearningRate(
#  initial_learning_rate=3e-6,
#  maximal_learning_rate=3e-4,
#  step_size=2360,
#  scale_fn=lambda x: 1 / (2.0 ** (x - 1)),
#  scale_mode='cycle') 

# opt = Adam(learning_rate=cyclical_learning_rate)
# opt= Adam(lr=4e-6)
# model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
# fit network
history=model.fit(
        x_train,
        y_train,verbose=1,
        batch_size=64, epochs=300,
        validation_split=0.2,
        steps_per_epoch=60,
        validation_data=(x_valid, y_valid),
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint('best_lstm.hdf5',
                                                monitor='val_accuracy',
                                                save_best_only=True,
                                                save_weights_only=False)
            ])
# evaluate model
_, accuracy = model.evaluate(x_valid, y_valid, batch_size=batch_size, verbose=0)



# summarize scores
model.summary()

    # load the best model
# model = cfg.load_model('best_conv.hdf5')
y_pred = model.predict(x_valid)

from sklearn.metrics import confusion_matrix, plot_confusion_matrix
cm=confusion_matrix(np.argmax(y_valid, axis=1), np.argmax(y_pred, axis=1), normalize='true')
print(cm)


   
    #plot curves
import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
    
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# If you need help, contact me p.fardmoshiri@gmail.com
