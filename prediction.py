import os
from os import listdir
from os.path import isfile, join
import pandas as pd
import librosa

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import numpy as np
from sklearn.model_selection import train_test_split
from timeit import default_timer as timer
from sklearn.metrics import classification_report, confusion_matrix
#import seaborn as sn
import matplotlib.pyplot as plt

import tensorflow as tf
import keras.models as models
from keras.models import Sequential, Model
from keras import regularizers, optimizers
from keras.layers import Dense, Conv1D, Flatten, Activation, MaxPooling1D, Dropout
from keras.utils import plot_model,to_categorical
from pathlib import Path

class Diagnosis():
  def __init__ (self, id, diagnosis, image_path):
    self.id = id
    self.diagnosis = diagnosis 
    self.image_path = image_path   

def get_wav_files():
  audio_path = 'very_large_data/autism_data_test/'
  files = [f for f in listdir(audio_path) if isfile(join(audio_path, f))]  #Gets all files in dir
  wav_files = [f for f in files if f.endswith('.wav')]  # Gets wav files 
  wav_files = sorted(wav_files)
  print(audio_path)
  return wav_files, audio_path

def diagnosis_data():
  diagnosis = pd.read_csv('labels_very_large_data/test_labels.csv')
  
  wav_files, audio_path = get_wav_files()
  #set up name of the first row
  diag_dict = {"id" : "truth"}

  diagnosis_list = []
  
  for index , row in diagnosis.iterrows():
    diag_dict[row[0]] = row[1]     

  c = 0
  for f in wav_files:
    diagnosis_list.append(Diagnosis(c, diag_dict[(f[:])], audio_path+f))  
    c+=1  

  return diagnosis_list

def audio_features(filename): 
  sound, sample_rate = librosa.load(filename)
  stft = np.abs(librosa.stft(sound))  
 
  mfccs = np.mean(librosa.feature.mfcc(y=sound, sr=sample_rate, n_mfcc=150),axis=1)
  chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate),axis=1)
  mel = np.mean(librosa.feature.melspectrogram(sound, sr=sample_rate),axis=1)
  contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate),axis=1)
  tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(sound), sr=sample_rate),axis=1)
    
  concat = np.concatenate((mfccs,chroma,mel,contrast,tonnetz))
  return concat

def data_points():
    image_file = Path('images_test.npy')
    labels_file = Path('labels_test.npy')

    if image_file.exists() & labels_file.exists():
        images = np.load('images_test.npy')
        labels = np.load('labels_test.npy')

    else:
        labels = []
        images = []

        to_hot_one = {"non_autistic":0, "autistic":1}

        count = 0
        for f in diagnosis_data():
            print(count)
            labels.append(to_hot_one[f.diagnosis]) 
            images.append(audio_features(f.image_path))
            count+=1
        np.save('labels_test.npy', labels)
        np.save('images_test.npy', images)

    return np.array(labels), np.array(images)

def preprocessing(labels, images):    

  # Split data
  #X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.0, random_state=10)

  # Hot one encode the labels
  #y_train = to_categorical(y_train)
  y_test = to_categorical(labels)
  X_test = images
  # Format new data
  #y_train = np.reshape(y_train, (y_train.shape[0], 2))
  #X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
  y_test = np.reshape(y_test, (y_test.shape[0], 2))
  X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1],  1))

  #return X_train, X_test, y_train, y_test
  return X_test, y_test
start = timer()

labels, images = data_points()
#X_train, X_test, y_train, y_test = preprocessing(labels, images)
X_test, y_test = preprocessing(labels, images)

print('Time taken: ', (timer() - start))

model = models.load_model('trained_model.h5')
matrix_index = ["non_autistic", "autistic"]

preds = model.predict(X_test)
#print(preds)

np.savetxt('pred.csv',preds,delimiter=',')

classpreds = np.argmax(preds, axis=1) # predicted classes 
y_testclass = np.argmax(y_test, axis=1) # true classes

np.savetxt('class_preds.csv', classpreds,delimiter=',')
np.savetxt('truth.csv',y_testclass,delimiter=',')

cm = confusion_matrix(y_testclass, classpreds)
print(classification_report(y_testclass, classpreds, target_names=matrix_index))

# Get percentage value for each element of the matrix
cm_sum = np.sum(cm, axis=1, keepdims=True)
cm_perc = cm / cm_sum.astype(float) * 100
annot = np.empty_like(cm).astype(str)
nrows, ncols = cm.shape
for i in range(nrows):
    for j in range(ncols):
        c = cm[i, j]
        p = cm_perc[i, j]
        if i == j:
            s = cm_sum[i]
            annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
        elif c == 0:
            annot[i, j] = ''
        else:
            annot[i, j] = '%.1f%%\n%d' % (p, c)

# Display confusion matrix 
df_cm = pd.DataFrame(cm, index = matrix_index, columns = matrix_index)
df_cm.index.name = 'Actual'
df_cm.columns.name = 'Predicted'
fig, ax = plt.subplots(figsize=(10,7))
#sn.heatmap(df_cm, annot=annot, fmt='')
#plt.savefig('confusion_test.png')
plt.close()
#os.system('python pred_merger.py')
