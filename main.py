#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 13:20:35 2022

@author: zeeshan
"""
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"] = "3";


from keras.layers import Dense, Dropout 
from keras.layers.recurrent import LSTM, GRU 
from keras.models import Sequential 
import pandas as pd 
import numpy as np 
from keras.layers import Flatten 
from keras.layers import GRU,Bidirectional
from keras.layers import Conv1D, MaxPooling1D, Input, LeakyReLU, Activation,TimeDistributed, BatchNormalization,concatenate
import matplotlib.pyplot as plt 
from sklearn.preprocessing import scale 
from sklearn.metrics import roc_curve, auc, confusion_matrix, roc_auc_score 
from sklearn.model_selection import StratifiedKFold, KFold
import keras.utils as utils #utils.tools as utils
import seaborn as sn
from sklearn.model_selection import train_test_split
import random
import math
import tensorflow as tf
from keras import losses
from keras.models import Model
from keras.metrics import binary_accuracy
from tensorflow.keras.optimizers import SGD,Adam,Adamax, Nadam, RMSprop, Adadelta
import keras
from keras import regularizers
from sklearn.metrics import matthews_corrcoef
# import utils.tools as utils

from keras import Sequential
import pickle 
 

data_=pd.read_csv('/home/zeeshan/Rice_data_test/test_data_EN.csv') 

data_io1=np.array(data_)  
 
data_only=data_io1[:,1:-1] 
label=data_io1[:,-1] 
[m1,n1]=np.shape(data_only)

shu=scale(data_only)
X1=shu 
y=label 
X=np.reshape(X1,(-1,1,n1))  


data_io = X
labels = y 



def calculateScore(X, y, model, folds):
    
    score = model.evaluate(X, y) # Gives loss and accuracy
    pred_y = model.predict(X)

    accuracy = score[1];

    tempLabel = np.zeros(shape = y.shape, dtype=np.int32)

    for i in range(len(y)):
        if pred_y[i] < 0.5:
            tempLabel[i] = 0;
        else:
            tempLabel[i] = 1;
    # print(tempLabel)        
    
    confusion = confusion_matrix(y, tempLabel)
    TN, FP, FN, TP = confusion.ravel()
    # print(confusion)
    

    # **** Confusion Matrix Plot ****

    confusion_norm = confusion / confusion.astype(np.float).sum(axis=1) # Normalize confusion matrix
    sn.heatmap(confusion_norm, annot=True, cmap='Blues')
    # sn.heatmap(confusion, annot=True, cmap='Blues')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
        

 
    sensitivity = TP / float(TP + FN)
    specificity = TN / float(TN + FP)
    MCC = matthews_corrcoef(y, tempLabel)
    # MCC = (TP*TN-FP*FN)/math.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))

    F1Score = (2 * TP) / float(2 * TP + FP + FN)
    precision = TP / float(TP + FP)

    pred_y = pred_y.reshape((-1, ))

    ROCArea = roc_auc_score(y, pred_y)
    fpr, tpr, thresholds = roc_curve(y, pred_y)
    lossValue = None;

    print(y.shape)
    print(pred_y.shape)

    y_true = tf.convert_to_tensor(y, np.float32)
    y_pred = tf.convert_to_tensor(pred_y, np.float32)
    
    plt.show() 
    
    lossValue = losses.binary_crossentropy(y_true, y_pred)#.eval()

    return {'sn' : sensitivity, 'sp' : specificity, 'acc' : accuracy, 'MCC' : MCC, 'AUC' : ROCArea, 'precision' : precision, 'F1' : F1Score, 'fpr' : fpr, 'tpr' : tpr, 'thresholds' : thresholds, 'lossValue' : lossValue}

# ************************ RESULTS *****************************
# ******** Performance Calculation and ROC Curve ***************

def analyze(temp, OutputDir):

    trainning_result, validation_result, testing_result = temp;

    file = open(OutputDir + '/performance.txt', 'w')

    index = 0
    for x in [trainning_result, validation_result, testing_result]:


        title = ''

        if index == 0:
            title = 'training_'
        if index == 1:
            title = 'validation_'
        if index == 2:
            title = 'testing_'

        index += 1;

        file.write(title +  'results\n')

        for j in ['sn', 'sp', 'acc', 'MCC', 'AUC', 'precision', 'F1', 'lossValue']:

            total = []

            for val in x:
                total.append(val[j])

            file.write(j + ' : mean : ' + str(np.mean(total)) + ' std : ' + str(np.std(total))  + '\n')

        file.write('\n\n______________________________\n')
    file.close();

# **** ROC Curve ****

    index = 0

    for x in [trainning_result, validation_result, testing_result]:

        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)

        i = 0

        for val in x:
            tpr = val['tpr']
            fpr = val['fpr']
            tprs.append(np.interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)
            plt.plot(fpr, tpr, lw=1, alpha=0.3,label='ROC fold %d (AUC = %0.2f)' % (i+1, roc_auc))

            i += 1

        print;

        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',label='Random', alpha=.8)

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        plt.plot(mean_fpr, mean_tpr, color='b',
                 label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
                 lw=2, alpha=.8)

        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                         label=r'$\pm$ 1 std. dev.')

        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic curve')
        plt.legend(loc="lower right")

        title = ''

        if index == 0:
            title = 'training_'
        if index == 1:
            title = 'validation_'
        if index == 2:
            title = 'testing_'

        plt.savefig( OutputDir + '/' + title +'ROC.png')
        plt.close('all');

        index += 1;



# obj = Object()

# ************** FINALIZED MODEL **************************
def model_cnn():
    
    n1=173 
    
    input_shape = (1,n1)
    inputs = Input(shape = input_shape)
    
    
    conv0 = Conv1D(filters=64, kernel_size=7,strides=1, padding = 'same')(inputs) #64
    normLayer0 = BatchNormalization(momentum=0.8, epsilon=1e-5)(conv0);
    act0 = Activation(activation='elu')(normLayer0)
    x0 = Flatten()(act0)
    
    dense1 = Dense(16, activation= 'elu')(x0)
    dense2 = Dense(8, activation= 'elu')(dense1)
    output = Dense(1, activation= 'sigmoid')(dense2)
    
    
    model = Model(inputs = inputs, outputs = output)
    
    opt=SGD(learning_rate=0.001, momentum = 0.9)
    model.compile(loss='binary_crossentropy', optimizer= opt, metrics=[binary_accuracy]);
    
    
    
    return model


# ************** K_Fold  **************************

folds=5
kf = KFold(n_splits=folds, shuffle=True, random_state=4) 

for train_index, test_index in kf.split(data_io,labels): 
    # print("TRAIN:", train_index, "TEST:", test_index)
    X_train0, X_test = data_io[train_index], data_io[test_index]
    y_train0, y_test = labels[train_index], labels[test_index]

    X_train, X_validation, y_train, y_validation = train_test_split(X_train0, y_train0, test_size=0.1, random_state=92, shuffle=True)

   
# ************** Train Model ***********************
    
trainning_result = []
validation_result = []
testing_result = []

for test_index in range(folds):    
    
    model = model_cnn();

    callback = tf.keras.callbacks.EarlyStopping(monitor='val_binary_accuracy', patience= 10, restore_best_weights=True)
    history = model.fit(X_train, y_train, batch_size = 32, epochs=50, validation_data = (X_validation, y_validation),callbacks=[callback]);
    
    model.save('/home/zeeshan/Rice_data_test/Results/model'+str(test_index+1)+'.h5')
    
    #**************** Plot graphs **************
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    
  
    trainning_result.append(calculateScore(X_train, y_train, model, folds))
    validation_result.append(calculateScore(X_validation, y_validation, model, folds));
    testing_result.append(calculateScore(X_test, y_test, model, folds));
        
temp_dict = (trainning_result, validation_result, testing_result)

OutputDir = '/home/zeeshan/Rice_data_test/Results/'

analyze(temp_dict,OutputDir)


