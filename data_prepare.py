#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 14:39:18 2022

@author: zeeshan
"""


# Apply onehot/ncp/eiip

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"] = "4";

# ************ Libraries **********************
# from tensorflow.keras import Sequential
import numpy as np
from Bio import SeqIO

def dataProcessing(path,fileformat):
    all_seq_data = []
    all_seq_data2 = []
    all_seq_data3 = []

    for record in SeqIO.parse(path,fileformat):
        sequences = record.seq # All sequences in dataset
    
        # print(record.seq)
        # print(sequences)
        seq_data=[]
        seq_data2=[]
        seq_data3=[]
       
        for i in range(len(sequences)):
            if sequences[i] == 'A':
                seq_data.append([1,0,0,0])
            if sequences[i] == 'T':
                seq_data.append([0,1,0,0])
            if sequences[i] == 'U':
                seq_data.append([0,1,0,0])                
            if sequences[i] == 'C':
                seq_data.append([0,0,1,0])
            if sequences[i] == 'G':
                seq_data.append([0,0,0,1])
            if sequences[i] == 'N':
                seq_data.append([0,0,0,0])
        all_seq_data.append(seq_data)
        # print(all_seq_data)
        
        for i in range(len(sequences)):
            if sequences[i] == 'A':
                seq_data2.append([1,1,1])
            if sequences[i] == 'T':
                seq_data2.append([0,1,0])
            if sequences[i] == 'U':
                seq_data2.append([0,1,0])
            if sequences[i] == 'C':
                seq_data2.append([0,0,1])
            if sequences[i] == 'G':
                seq_data2.append([1,0,0])
            if sequences[i] == 'N':
                seq_data2.append([0,0,0])        
        all_seq_data2.append(seq_data2)
        
        for i in range(len(sequences)):
            if sequences[i] == 'A':
                seq_data3.append([0.1260])
            if sequences[i] == 'T':
                seq_data3.append([0.1335])
            if sequences[i] == 'U':
                seq_data3.append([0.1335])
            if sequences[i] == 'C':
                seq_data3.append([0.1340])
            if sequences[i] == 'G':
                seq_data3.append([0.0806])
            if sequences[i] == 'N':
                seq_data3.append([0.0])        
        all_seq_data3.append(seq_data3)
        
        
    all_seq_data = np.array(all_seq_data);
    all_seq_data2 = np.array(all_seq_data2);
    all_seq_data3 = np.array(all_seq_data3);
    
    comb = np.concatenate((all_seq_data, all_seq_data2,all_seq_data3), axis=2)
    
    return comb,all_seq_data,all_seq_data2,all_seq_data3
    
data_io = dataProcessing("/home/zeeshan/Rice_data_test/test_data.fasta", "fasta")

# ********** DATA LABELLING *********************

len_data = len(data_io[0])
pos_lab = np.ones(int(len_data/2));
neg_lab = np.zeros(int(len_data/2));
labels = np.concatenate((pos_lab,neg_lab),axis=0)

# ***Saving CSV # Reshaping and labeling data
a = data_io[0]
a = a.reshape(len_data,-1)
labels = labels.reshape(len_data,-1)
data = np.concatenate((a, labels),axis=1)
np.savetxt("test.csv", data, delimiter=",")
