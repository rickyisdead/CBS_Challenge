#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import TruncatedSVD

#list of selected features
features = ['computerexperience',
'nativespeaker',
'edlevel3',
'monthlyincpr',
'yearlyincpr',
'v89',
'v127',
'v239',
'v224',
'v71',
'v105',
'yrsqual',
'yrsqual_t',
'yrsget',
'vet',
'ctryqual',
'nativelang',
'ageg10lfs',
'edcat8',
'leaver1624',
'leavedu',
'fe12',
'aetpop',
'edwork',
'neet',
'nopaidworkever',
'paidwork12',
'paidwork5',
'earnmthallppp',
'learnatwork_wle_ca',
'readytolearn_wle_ca',
'icthome_wle_ca',
'ictwork_wle_ca',
'influence_wle_ca',
'planning_wle_ca',
'readhome_wle_ca',
'readwork_wle_ca',
'taskdisc_wle_ca',
'writhome_wle_ca',
'writwork_wle_ca',
'v34',
'v42',
'aetpop',
'v31',
'v82',
'v289',
'v107',
'v60',
'iscoskil4',
'job_performance']





# In[2]:


#reads selected features into dataframe
#explicitly casts numerical data representing categories instead of values into strings
df = pd.read_csv('hw5-trainingset-rp2876.csv',header=0,low_memory=False,dtype = {'v239':str, 'v224':str, 'v105':str},usecols=features)

#reads the job_performance column into an array and drops it from the dataframe
labels = df.get('job_performance').to_numpy()
df = df.drop('job_performance', axis=1)

#separates data into numerical and categorical data
category_data = df.select_dtypes(include=['object'])
numerical_data = df.select_dtypes(exclude=['object'])

num_size = len(numerical_data.columns)

#encodes categorical data with one hot key
onehot = pd.get_dummies(category_data,dummy_na=True)

df = numerical_data.join(onehot)

#convert dataframe to numpy ndarray
darray = df.to_numpy(dtype = 'float32')

#imputes missing values for numerical data inputs
col_mean = np.nanmean(darray[:,:num_size], axis=0)
ind = np.where(np.isnan(darray[:,:num_size]))
darray[ind] = np.take(col_mean, ind[True])


# In[3]:


#transform the data using truncated SVD and scale
svd = TruncatedSVD(n_components = 300)
X_trans = svd.fit_transform(darray)
scaler = StandardScaler()
X_trans = scaler.fit_transform(X_trans)


# In[4]:


#lasso regression
clf = linear_model.Lasso(alpha = 2.0)
clf.fit(X_trans,labels)


# In[5]:


#load testing set
features.remove('job_performance')
dff = pd.read_csv('hw5-testset-rp2876.csv',header=0,low_memory=False,dtype = {'v239':str, 'v224':str, 'v105':str},usecols=features)

#separates data into numerical and categorical data
category_data = dff.select_dtypes(include=['object'])
numerical_data = dff.select_dtypes(exclude=['object'])

num_size = len(numerical_data.columns)

#encodes categorical data with one hot key
onehot = pd.get_dummies(category_data,dummy_na=True)

dff = numerical_data.join(onehot)

#convert dataframe to numpy ndarray
test = dff.to_numpy(dtype = 'float32')

#imputes missing values for numerical data inputs
col_mean = np.nanmean(test[:,:num_size], axis=0)
ind = np.where(np.isnan(test[:,:num_size]))
test[ind] = np.take(col_mean, ind[True])

test = svd.fit_transform(test)
test = scaler.fit_transform(test)


# In[7]:


#use trained classifier to predict on testing set
predicted = clf.predict(test)
print(predicted)
writeup = pd.DataFrame(data=predicted, columns=['job_performance'])

#export to empty csv file
filedata = pd.read_csv('hw5-testset-rp2876.csv',header=0,low_memory=False)
filedata['job_performance'] = writeup['job_performance']
filedata.to_csv('hw5-testset-rp2876-submission.csv',index=False)

