#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 14:40:06 2022

@author: zeeshan
"""

import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
from sklearn.preprocessing import scale,StandardScaler
from sklearn.linear_model import ElasticNet,ElasticNetCV



def elasticNet(data,label,alpha =np.array([0.01])):
    enet=ElasticNet(alpha=alpha, l1_ratio=0.01)
    enet.fit(data,label)
    mask = enet.coef_ != 0
    new_data = data[:,mask]
    return new_data,mask
    
    
data_=pd.read_csv('/home/zeeshan/Rice_data_test/test_data.csv')
dataa=np.array(data_)
# data=data[:,2:]
dataa=dataa[:,:-1]
[m1,n1]=np.shape(dataa)
label1=np.ones((int(m1/2+1),1))
label2=np.zeros((int(m1/2),1)) # +1 added
label=np.append(label1,label2)
# label = dataa[:,-1]
shu=scale(dataa)
data_2,index=elasticNet(shu,label)
shu=data_2
data_csv = pd.DataFrame(data=shu)

l = label.reshape(-1,1)
labelled_data = np.concatenate((data_csv,l),axis=1)
labelled_data = pd.DataFrame(data=labelled_data)
labelled_data.to_csv('test_EN.csv')


# Saving indexes
a = pd.DataFrame(index)
a.to_csv("test_data_EN_INDEXES.csv")




#%%

########## Preparing ind test data using ElasticNet indexes ############

import pandas as pd
import numpy as np
from sklearn.preprocessing import scale 
a = pd.read_csv('/home/zeeshan/Rice_data_test/indep_data.csv', header=None)
a=np.array(a)
aa = a[:,:-1]
aa=scale(aa)
l = a[:,-1]
ll=l.reshape(len(aa),-1)
ind = pd.read_csv('/home/zeeshan/Rice_data_test/test_data_EN_INDEXES.csv')
ind = np.array(ind)
ind=ind[:,1]
ind = ind.astype(bool)
tr_data = aa[:,ind]
tr_data2 = np.concatenate((tr_data,ll),axis=1)
tr_data2 = pd.DataFrame(data=tr_data2)
tr_data2.to_csv('indep_data_EN.csv')

