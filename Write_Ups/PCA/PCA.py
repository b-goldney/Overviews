#!/usr/bin/env python
# coding: utf-8

# # Homework 3: PCA

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt # needed to play video


# In order to transfer the data files from Matlab to Python, I had to write each camera's data to csv files. It's not possible to write a 3D (or greater) matrix to a csv file. In Matlab, to write the data to csv files, I have to reshape it.  I reshaped each file into a (n,1) matrix.  After loading the data into Python, I'll reshape the data into the proper 4D shape. 

# ### Test 1: Ideal Case
# - Consider a small displacement of the mass in the z direction and the ensuing oscillations. In this case, the entire motion is in the z directions with simple harmonic motion being observed
# - The dimensions of each file are as follows:  
#  - vidFrames1_1: 480,640,3,226  
#  - vidFrames2_1: 480,640,3,284  
#  - vidFrames3_1: 480,640,3,232  

# Read files for the first video:
# - camN_1.mat where N=1,2,3

# In[1136]:


files = []
for i in range(1,4):
        files.append(pd.read_csv('cam'+ str(i) + '_1.csv',header=None))


# In[1137]:


cam1_1 = files[0].to_numpy()
cam2_1 = files[1].to_numpy()
cam3_1 = files[2].to_numpy()

cam1_1 = cam1_1.reshape((480,640,3,226), order='F') # Convert to ndarray, order='F' is to reshape using
# Fortran style which is first index first, whic is what Matlab uses
cam2_1 = cam2_1.reshape((480,640,3,284), order='F')
cam3_1 = cam3_1.reshape((480,640,3,232), order='F')


# In[1138]:


[a11,b11,c11,d11] = cam1_1.shape
[a21,b21,c21,d21] = cam2_1.shape
[a31,b31,c31,d31] = cam3_1.shape


# **Show sequence of images to understand the trajectory of the paint can**

# In[1139]:


for i in range(0,50,15):
    plt.imshow(cam1_1[:,:,:,i])
    plt.show()


# In[1140]:


window = 20
x_axis = [322]
y_axis = [228]

test = cam1_1[:,:,:,0]

test[0:y_axis[-1]-window,:] = 0 # for some reason this changes the y axis
test[window+y_axis[-1]:,:] = 0
test[:,0:x_axis[-1]-window] = 0 # and this changes the x axis
test[:,window+x_axis[-1]:] = 0

plt.imshow(test)


# From the pictures above we can see the paint can stays between the values of 300 and 400 on the x-axis.  In order to save memory and to ensure only the flash light is captured, we'll assign all other pixels a value of 0. Additionally, we'll use a window to further reduce the search area (i.e the area from which the max value will be pulled).  We'll determine the coordinates of the flashlight on the first image and then only search an area that's slightly above/below that window to pull the max value from. 

# **Test 1: First Camera**

# To acquire the mouse coordinates I used ginput in Matlab.  Matlab's seems to have better built-in functionality for this task.  I used the following code in Matlab:
# 
# imshow(vidFrames1_1(:,:,:,1)))  
# [x1, y1] = ginput(1);
# 
# The mouse coordinates of the first image are [322,228]

# In[1141]:


from skimage.color import rgb2gray

x_axis1_1 = [322]
y_axis1_1 = [228]
window = 20

for i in range(1,d11):
    img = rgb2gray(cam1_1[:,:,:,i])
    img[0:y_axis1_1[-1]-window,:] = 0 # for some reason this changes the y axis
    img[window+y_axis1_1[-1]:,:] = 0
    img[:,0:x_axis1_1[-1]-window] = 0 # and this changes the x axis
    img[:,window+x_axis1_1[-1]:] = 0    #img[:,227+window:] = 0
    [y_axis,x_axis] = np.unravel_index(np.argmax(img, axis=None), img.shape)

    y_axis1_1.append(y_axis)
    x_axis1_1.append(x_axis)
    #print(y_axis2,x_axis2)


# **Test 1: Second Camera**

# In[1142]:


x_axis2_1 = [278]
y_axis2_1 = [272]
window = 20

for i in range(1,d21):
    img = rgb2gray(cam2_1[:,:,:,i])
    img[0:y_axis2_1[-1]-window,:] = 0 # for some reason this changes the y axis
    img[window+y_axis2_1[-1]:,:] = 0
    img[:,0:x_axis2_1[-1]-window] = 0 # and this changes the x axis
    img[:,window+x_axis2_1[-1]:] = 0    #img[:,227+window:] = 0
    [y_axis,x_axis] = np.unravel_index(np.argmax(img, axis=None), img.shape)

    y_axis2_1.append(y_axis)
    x_axis2_1.append(x_axis)
    #print(y_axis2,x_axis2)


# **Test 1: Third Camera**

# In[1143]:


x_axis3_1 = [320]
y_axis3_1 = [270]
window = 20

for i in range(1,d31):
    img = rgb2gray(cam3_1[:,:,:,i])
    img[0:y_axis3_1[-1]-window,:] = 0 # for some reason this changes the y axis
    img[window+y_axis3_1[-1]:,:] = 0
    img[:,0:x_axis3_1[-1]-window] = 0 # and this changes the x axis
    img[:,window+x_axis3_1[-1]:] = 0    #img[:,227+window:] = 0
    [y_axis,x_axis] = np.unravel_index(np.argmax(img, axis=None), img.shape)

    y_axis3_1.append(y_axis)
    x_axis3_1.append(x_axis)
    #print(y_axis2,x_axis2)


# **To conserve memory we'll delete the three numpy ndarrays**

# In[1144]:


del cam1_1, cam2_1, cam3_1


# **Ensure all the data has the same length**

# In[1145]:


test1_ymin = min(len(y_axis1_1),len(y_axis2_1),len(y_axis3_1))
#test1_xmin = min(len(x_axis1_1),len(x_axis2_1),len(x_axis3_1))


y_axis1_1 = y_axis1_1[0:test1_ymin]
y_axis2_1 = y_axis2_1[0:test1_ymin]
y_axis3_1 = y_axis3_1[0:test1_ymin]

x_axis1_1 = x_axis1_1[0:test1_ymin] # since x-axis and y-axis have the same length it doesn't matter which is used
x_axis2_1 = x_axis2_1[0:test1_ymin]
x_axis3_1 = x_axis3_1[0:test1_ymin]


# **Normalize the data for the x and y axis**

# In[1146]:


# Normalize y coordinates for the flashlight
yaxis1_1_norm = y_axis1_1 / max(y_axis1_1)
yaxis2_1_norm = y_axis2_1 / max(y_axis2_1)
yaxis3_1_norm = y_axis3_1 / max(y_axis3_1)

# Normalize x coordinates for the flashlight
xaxis1_1_norm = x_axis1_1 / max(x_axis1_1)
xaxis2_1_norm = x_axis2_1 / max(x_axis2_1)
xaxis3_1_norm = x_axis3_1 / max(x_axis3_1)


# In[1147]:


fig,(ax1, ax2, ax3) =  plt.subplots(1,3,figsize=(15,4))

ax1.plot(np.linspace(1,test1_ymin,test1_ymin) /test1_ymin,yaxis1_1_norm)
ax2.plot(np.linspace(1,test1_ymin,test1_ymin) /test1_ymin,yaxis2_1_norm)
ax3.plot(np.linspace(1,test1_ymin,test1_ymin) /test1_ymin,yaxis3_1_norm)

ax1.set_title('Camera 1', fontsize=10)
ax2.set_title('Camera 2', fontsize=10)
ax3.set_title('Camera 3', fontsize=10)

ax1.set_ylim([0, 1])
ax2.set_ylim([0, 1])
ax3.set_ylim([0, 1])

fig.suptitle('Test 1: y-axis')

plt.show()


# In[1167]:


fig,(ax1, ax2, ax3) =  plt.subplots(1,3,figsize=(15,4))

ax1.plot(np.linspace(1,test1_ymin,test1_ymin) ,xaxis1_1_norm)
ax2.plot(np.linspace(1,test1_ymin,test1_ymin) ,xaxis2_1_norm)
ax3.plot(np.linspace(1,test1_ymin,test1_ymin) ,xaxis3_1_norm)

ax1.set_title('Camera 1', fontsize=10)
ax2.set_title('Camera 2', fontsize=10)
ax3.set_title('Camera 3', fontsize=10)

ax1.set_ylim([0, 1])
ax2.set_ylim([0, 1])
ax3.set_ylim([0, 1])

fig.suptitle('Test 1: x-axis')

plt.show()


# **Apply PCA**

# In[1149]:


from numpy import linalg as LA

# Create 2 dataframes: one for x coordinates and one for y coordinates
test1_x = pd.DataFrame({'Camera 1':x_axis1_1, 'Camera 2': x_axis2_1,'Camera 3': x_axis3_1}).T
test1_y = pd.DataFrame({'Camera 1':y_axis1_1, 'Camera 2': y_axis2_1,'Camera 3': y_axis3_1}).T

# Subtract mean from each row
test1_x.iloc[0,:] = test1_x.iloc[0,:] - test1_x.iloc[0,:].mean()
test1_x.iloc[1,:] = test1_x.iloc[1,:] - test1_x.iloc[1,:].mean()
test1_x.iloc[2,:] = test1_x.iloc[2,:] - test1_x.iloc[2,:].mean()

test1_y.iloc[0,:] = test1_y.iloc[0,:] - test1_y.iloc[0,:].mean()
test1_y.iloc[1,:] = test1_y.iloc[1,:] - test1_y.iloc[1,:].mean()
test1_y.iloc[2,:] = test1_y.iloc[2,:] - test1_y.iloc[2,:].mean()


# Divide by standard deviation
test1_x.iloc[0,:] = test1_x.iloc[0,:] / np.std(test1_x.iloc[0,:])
test1_x.iloc[1,:] = test1_x.iloc[1,:] / np.std(test1_x.iloc[1,:])
test1_x.iloc[2,:] = test1_x.iloc[2,:] / np.std(test1_x.iloc[2,:])

test1_y.iloc[0,:] = test1_y.iloc[0,:] / np.std(test1_y.iloc[0,:])
test1_y.iloc[1,:] = test1_y.iloc[1,:] / np.std(test1_y.iloc[1,:])
test1_y.iloc[2,:] = test1_y.iloc[2,:] / np.std(test1_y.iloc[2,:])

# Append three rows of x values to three rows of y values
test1 = test1_x.append(test1_y)

# calculate covariance matrix
#covar = (1 / (len(test1.columns)) - 1) * np.cov(test1, rowvar=True)
covar = np.cov(test1, rowvar=True)

# Ensure covariance matrix is a square
print(covar.shape, 'notice covar is a square matrix')


# In[1150]:


# Factor covariance matrix using SVD
[U,S,V] = np.linalg.svd(test1 / np.sqrt(226-1), full_matrices=False)


# In[1151]:


from scipy.linalg import svd

rho1 = (S*S) / (S*S).sum()

threshold = 0.95

plt.figure()
plt.plot(range(1,len(rho1)+1),rho,'x-')
plt.plot(range(1,len(rho1)+1),np.cumsum(rho1),'o-')
plt.plot([1,len(rho1)],[threshold, threshold],'k--')
plt.title('Variance explained by principal components');
plt.xlabel('Principal component');
plt.ylabel('Variance explained');
plt.legend(['Individual','Cumulative','Threshold'])
#plt.grid()
plt.show()


# In[1152]:


rho1


# **Plot Projections of Data onto Left Singular Vectors**

# In[1157]:


modes = np.(U.T,test1)
modes.iloc[0,:]

plt.plot(range(0,226), modes.iloc[0,:])
plt.plot(range(0,226), modes.iloc[1,:])
plt.plot(range(0,226), modes.iloc[2,:])

plt.title('Test 1: Projections of Data onto Left Singular Vectors')


# **Confirm the values from the manual calculation are close to those from sklearn**

# In[1158]:


from sklearn import decomposition
from sklearn import datasets

from sklearn.decomposition import PCA
pca = PCA(n_components=6, svd_solver='full')
principalComponents = pca.fit_transform(test1.T)
#pca.explained_variance_ratio_

# Check if components match using explained_variance_ from sklear
print(pca.explained_variance_[0] / sum(pca.explained_variance_), 'sklearn: the first component matches')
print(pca.explained_variance_[1] / sum(pca.explained_variance_), 'sklearn: the second component matches')


# In[1155]:


# Check if components match using eigenvalue decomposition
eig_val, eig_vec = np.linalg.eig(covar)
print(eig_val[0] / sum(eig_val), 'Evalue decomp: the first component matches')
print(eig_val[1] / sum(eig_val), 'Evalue decomp: the second component matches')


# In[1156]:


print(rho1[0], 'SVD: the first component matches')
print(rho1[1], 'SVD: the second component matches')


# ### Test 2: Noisy Case
# - Introduce camera shake into the video recording

# In[1049]:


files = []
for i in range(1,4):
        files.append(pd.read_csv('cam'+ str(i) + '_2.csv',header=None))


# - The dimensions of each file are as follows:  
#  - vidFrames1_2: 480,640,3,314  
#  - vidFrames2_2: 480,640,3,356
#  - vidFrames3_2: 480,640,3,327

# In[1050]:


cam1_2 = files[0].to_numpy()
cam2_2 = files[1].to_numpy()
cam3_2 = files[2].to_numpy()

cam1_2 = cam1_2.reshape((480,640,3,314), order='F') # Convert to ndarray, order='F' is to reshape using
# Fortran style which is first index first, which is what Matlab uses
cam2_2 = cam2_2.reshape((480,640,3,356), order='F')
cam3_2 = cam3_2.reshape((480,640,3,327), order='F')


# In[1051]:


[a12,b12,c12,d12] = cam1_2.shape 
[a22,b22,c22,d22] = cam2_2.shape
[a32,b32,c32,d32] = cam3_2.shape


# **Test 2: First Camera**

# In[1052]:


x_axis1_2 = [326]
y_axis1_2 = [304]
window = 30

for i in range(1,d12):
    img = rgb2gray(cam1_2[:,:,:,i])
    img[0:y_axis1_2[-1]-window,:] = 0 # for some reason this changes the y axis
    img[window+y_axis1_2[-1]:,:] = 0
    img[:,0:x_axis1_2[-1]-window] = 0 # and this changes the x axis
    img[:,window+x_axis1_2[-1]:] = 0    #img[:,227+window:] = 0
    [y_axis,x_axis] = np.unravel_index(np.argmax(img, axis=None), img.shape)

    y_axis1_2.append(y_axis)
    x_axis1_2.append(x_axis)
    #print(y_axis2,x_axis2)


# **Test 2: Second Camera**

# In[1053]:


x_axis2_2 = [326]
y_axis2_2 = [304]
window = 30

for i in range(1,d22):
    img = rgb2gray(cam2_2[:,:,:,i])
    img[0:y_axis2_2[-1]-window,:] = 0 # for some reason this changes the y axis
    img[window+y_axis2_2[-1]:,:] = 0
    img[:,0:x_axis2_2[-1]-window] = 0 # and this changes the x axis
    img[:,window+x_axis2_2[-1]:] = 0    #img[:,227+window:] = 0
    [y_axis,x_axis] = np.unravel_index(np.argmax(img, axis=None), img.shape)

    y_axis2_2.append(y_axis)
    x_axis2_2.append(x_axis)
    #print(y_axis2,x_axis2)


# **Test 2: Third Camera**

# In[1054]:


x_axis3_2 = [348]
y_axis3_2 = [246]
window = 20

for i in range(1,d32):
    img = rgb2gray(cam3_2[:,:,:,i])
    img[0:y_axis3_2[-1]-window,:] = 0 # for some reason this changes the y axis
    img[window+y_axis3_2[-1]:,:] = 0
    img[:,0:x_axis3_2[-1]-window] = 0 # and this changes the x axis
    img[:,window+x_axis3_2[-1]:] = 0    #img[:,227+window:] = 0
    [y_axis,x_axis] = np.unravel_index(np.argmax(img, axis=None), img.shape)

    y_axis3_2.append(y_axis)
    x_axis3_2.append(x_axis)
    #print(y_axis2,x_axis2)


# To conserve memory we'll delete the three numpy ndarrays

# In[777]:


del cam1_2, cam2_2, cam3_2


# **Ensure all the data has the same length**

# In[1055]:


test2_min = min(len(y_axis1_2),len(y_axis2_2),len(y_axis3_2))
#test1_xmin = min(len(x_axis1_1),len(x_axis2_1),len(x_axis3_1))

y_axis1_2 = y_axis1_2[0:test2_min]
y_axis2_2 = y_axis2_2[0:test2_min]
y_axis3_2 = y_axis3_2[0:test2_min]

x_axis1_2 = x_axis1_2[0:test2_min] # since x-axis and y-axis have the same length it doesn't matter which is used
x_axis2_2 = x_axis2_2[0:test2_min]
x_axis3_2 = x_axis3_2[0:test2_min]


# **Normalize the data**

# In[1010]:


yaxis1_2_norm = y_axis1_2 / max(y_axis1_2)
yaxis2_2_norm = y_axis2_2 / max(y_axis2_2)
yaxis3_2_norm = y_axis3_2 / max(y_axis3_2)

xaxis1_2_norm = x_axis1_2 / max(x_axis1_2)
xaxis2_2_norm = x_axis2_2 / max(x_axis2_2)
xaxis3_2_norm = x_axis3_2 / max(x_axis3_2)


# In[1166]:


fig,(ax1, ax2, ax3) =  plt.subplots(1,3,figsize=(15,4))

ax1.plot(range(0,test2_min),yaxis1_2_norm)
ax2.plot(range(0,test2_min),yaxis2_2_norm)
ax3.plot(range(0,test2_min),yaxis3_2_norm)

ax1.set_title('Camera 1', fontsize=10)
ax2.set_title('Camera 2', fontsize=10)
ax3.set_title('Camera 3', fontsize=10)

ax1.set_ylim([0, 1])
ax2.set_ylim([0, 1])
ax3.set_ylim([0, 1])

fig.suptitle('Test 2: y-Axis')

plt.show()


# In[1012]:


fig,(ax1, ax2, ax3) =  plt.subplots(1,3,figsize=(15,4))

ax1.plot(range(0,test2_min),xaxis1_2_norm)
ax2.plot(range(0,test2_min),xaxis2_2_norm)
ax3.plot(range(0,test2_min),xaxis3_2_norm)

ax1.set_title('Camera 1', fontsize=10)
ax2.set_title('Camera 2', fontsize=10)
ax3.set_title('Camera 3', fontsize=10)

ax1.set_ylim((0,1))
ax2.set_ylim((0,1))
ax3.set_ylim((0,1))

fig.suptitle('Test 2: x-Axis')

plt.show()


# **Apply PCA**

# In[1056]:


from numpy import linalg as LA

# Create 2 dataframes: one for x coordinates and one for y coordinates
test2_x = pd.DataFrame({'Camera 1':x_axis1_2, 'Camera 2': x_axis2_2,'Camera 3': x_axis3_2}).T
test2_y = pd.DataFrame({'Camera 1':y_axis1_2, 'Camera 2': y_axis2_2,'Camera 3': y_axis3_2}).T

# Subtract mean from each row
test2_x.iloc[0,:] = test2_x.iloc[0,:] - test2_x.iloc[0,:].mean()
test2_x.iloc[1,:] = test2_x.iloc[1,:] - test2_x.iloc[1,:].mean()
test2_x.iloc[2,:] = test2_x.iloc[2,:] - test2_x.iloc[2,:].mean()

test2_y.iloc[0,:] = test2_y.iloc[0,:] - test2_y.iloc[0,:].mean()
test2_y.iloc[1,:] = test2_y.iloc[1,:] - test2_y.iloc[1,:].mean()
test2_y.iloc[2,:] = test2_y.iloc[2,:] - test2_y.iloc[2,:].mean()

# Divide by standard deviation
test2_x.iloc[0,:] = test2_x.iloc[0,:] / np.std(test2_x.iloc[0,:])
test2_x.iloc[1,:] = test2_x.iloc[1,:] / np.std(test2_x.iloc[1,:])
test2_x.iloc[2,:] = test2_x.iloc[2,:] / np.std(test2_x.iloc[2,:])

test2_y.iloc[0,:] = test2_y.iloc[0,:] / np.std(test2_y.iloc[0,:])
test2_y.iloc[1,:] = test2_y.iloc[1,:] / np.std(test2_y.iloc[1,:])
test2_y.iloc[2,:] = test2_y.iloc[2,:] / np.std(test2_y.iloc[2,:])

# Append three rows of x values to three rows of y values
test2 = test2_x.append(test2_y)

# calculate covariance matrix
#covar = (1 / (len(test1.columns)) - 1) * np.cov(test1, rowvar=True)
covar = np.cov(test2, rowvar=True)

# Ensure covariance matrix is a square
print(covar.shape, 'notice covar is a square matrix')


# In[1064]:


# Factor covariance matrix using SVD
[U,S,V] = np.linalg.svd(test2 / np.sqrt(len(test2.columns)-1), full_matrices=False)


# In[1065]:


from scipy.linalg import svd

rho2 = (S*S) / (S*S).sum()

threshold = 0.95

plt.figure()
plt.plot(range(1,len(rho2)+1),rho2,'x-')
plt.plot(range(1,len(rho2)+1),np.cumsum(rho2),'o-')
plt.plot([1,len(rho2)],[threshold, threshold],'k--')
plt.title('Test 2: Variance explained by principal components');
plt.xlabel('Principal component');
plt.ylabel('Variance explained');
plt.legend(['Individual','Cumulative','Threshold'])
#plt.grid()
plt.show()


# In[1066]:


rho2


# **Plot Projections of Data onto Left Singular Vectors**

# In[1067]:


modes = np.matmul(U.T,test2)
modes.iloc[0,:]

plt.plot(range(0,len(test2.columns)), modes.iloc[0,:])
plt.plot(range(0,len(test2.columns)), modes.iloc[1,:])
plt.plot(range(0,len(test2.columns)), modes.iloc[2,:])

plt.title('Test 2: Projections of Data onto Left Singular Vectors')


# **Confirm the values from the manual calculation are close to those from sklearn**

# In[1070]:


from sklearn import decomposition
from sklearn import datasets

from sklearn.decomposition import PCA
pca = PCA(n_components=6, svd_solver='full')
principalComponents = pca.fit_transform(test2.T)
#pca.explained_variance_ratio_

# Check if components match using explained_variance_ from sklear
print(pca.explained_variance_[0] / sum(pca.explained_variance_), 'sklearn: the first component matches')
print(pca.explained_variance_[1] / sum(pca.explained_variance_), 'sklearn: the second component matches')


# In[1095]:


# Check if components match using eigenvalue decomposition
eig_val, eig_vec = np.linalg.eig(covar)
print(eig_val[0] / sum(eig_val), 'Evalue decomp: the first component matches')
print(eig_val[1] / sum(eig_val), 'Evalue decomp: the second no match (check instability of Eig Decomp)')


# In[1072]:


print(rho2[0], 'SVD: the first component matches')
print(rho2[1], 'SVD: the second component matches')


# ### Test 3:  horizontal displacement

# - The dimensions of each file are as follows:  
#  - vidFrames1_2: 480,640,3,239
#  - vidFrames2_2: 480,640,3,281
#  - vidFrames3_2: 480,640,3,237

# In[792]:


files = []
for i in range(1,4):
        files.append(pd.read_csv('cam'+ str(i) + '_3.csv',header=None))


# In[793]:


cam1_3 = files[0].to_numpy()
cam2_3 = files[1].to_numpy()
cam3_3 = files[2].to_numpy()

cam1_3 = cam1_3.reshape((480,640,3,239), order='F') # Convert to ndarray, order='F' is to reshape using
# Fortran style which is first index first, whic is what Matlab uses
cam2_3 = cam2_3.reshape((480,640,3,281), order='F')
cam3_3 = cam3_3.reshape((480,640,3,237), order='F')


# In[794]:


[a13,b13,c13,d13] = cam1_3.shape 
[a23,b23,c23,d23] = cam2_3.shape
[a33,b33,c33,d33] = cam3_3.shape


# **Test 3: First Camera**

# In[795]:


x_axis1_3 = [326]
y_axis1_3 = [284]
window = 20

for i in range(1,d13):
    img = rgb2gray(cam1_3[:,:,:,i])
    img[0:y_axis1_3[-1]-window,:] = 0 # for some reason this changes the y axis
    img[window+y_axis1_3[-1]:,:] = 0
    img[:,0:x_axis1_3[-1]-window] = 0 # and this changes the x axis
    img[:,window+x_axis1_3[-1]:] = 0    #img[:,227+window:] = 0
    [y_axis,x_axis] = np.unravel_index(np.argmax(img, axis=None), img.shape)

    y_axis1_3.append(y_axis)
    x_axis1_3.append(x_axis)
    #print(y_axis2,x_axis2)


# **Test 3: Second Camera**

# In[796]:


x_axis2_3 = [244]
y_axis2_3 = [292]
window = 20

for i in range(1,d23):
    img = rgb2gray(cam2_3[:,:,:,i])
    img[0:y_axis2_3[-1]-window,:] = 0 # for some reason this changes the y axis
    img[window+y_axis2_3[-1]:,:] = 0
    img[:,0:x_axis2_3[-1]-window] = 0 # and this changes the x axis
    img[:,window+x_axis2_3[-1]:] = 0    #img[:,227+window:] = 0
    [y_axis,x_axis] = np.unravel_index(np.argmax(img, axis=None), img.shape)

    y_axis2_3.append(y_axis)
    x_axis2_3.append(x_axis)
    #print(y_axis2,x_axis2)


# **Test 3: Third Camera**

# In[797]:


x_axis3_3 = [356]
y_axis3_3 = [230]
window = 20

for i in range(1,d33):
    img = rgb2gray(cam3_3[:,:,:,i])
    img[0:y_axis3_3[-1]-window,:] = 0 # for some reason this changes the y axis
    img[window+y_axis3_3[-1]:,:] = 0
    img[:,0:x_axis3_3[-1]-window] = 0 # and this changes the x axis
    img[:,window+x_axis3_3[-1]:] = 0    #img[:,227+window:] = 0
    [y_axis,x_axis] = np.unravel_index(np.argmax(img, axis=None), img.shape)

    y_axis3_3.append(y_axis)
    x_axis3_3.append(x_axis)
    #print(y_axis2,x_axis2)


# In[798]:


del cam1_3, cam2_3, cam3_3


# **Ensure all the data has the same length**

# In[799]:


test3_min = min(len(y_axis1_3),len(y_axis2_3),len(y_axis3_3))
#test1_xmin = min(len(x_axis1_1),len(x_axis2_1),len(x_axis3_1))

y_axis1_3 = y_axis1_3[0:test3_min]
y_axis2_3 = y_axis2_3[0:test3_min]
y_axis3_3 = y_axis3_3[0:test3_min]

x_axis1_3 = x_axis1_3[0:test3_min] # since x-axis and y-axis have the same length it doesn't matter which is used
x_axis2_3 = x_axis2_3[0:test3_min]
x_axis3_3 = x_axis3_3[0:test3_min]


# **Normalize the data**

# In[1077]:


yaxis1_3_norm = y_axis1_3 / max(y_axis1_3)
yaxis2_3_norm = y_axis2_3 / max(y_axis2_3)
yaxis3_3_norm = y_axis3_3 / max(y_axis3_3)

xaxis1_3_norm = x_axis1_3 / max(x_axis1_3)
xaxis2_3_norm = x_axis2_3 / max(x_axis2_3)
xaxis3_3_norm = x_axis3_3 / max(x_axis3_3)


# In[1078]:


fig,(ax1, ax2, ax3) =  plt.subplots(1,3,figsize=(15,4))

ax1.plot(range(0,test3_min),yaxis1_3_norm)
ax2.plot(range(0,test3_min),yaxis2_3_norm)
ax3.plot(range(0,test3_min),yaxis3_3_norm)

ax1.set_title('Camera 1', fontsize=10)
ax2.set_title('Camera 2', fontsize=10)
ax3.set_title('Camera 3', fontsize=10)

ax1.set_ylim([0, 1])
ax2.set_ylim([0, 1])
ax3.set_ylim([0, 1])

fig.suptitle('Test 3: y-Axis')

plt.show()


# In[1079]:


fig,(ax1, ax2, ax3) =  plt.subplots(1,3,figsize=(15,4))

ax1.plot(range(0,test3_min),xaxis1_3_norm)
ax2.plot(range(0,test3_min),xaxis2_3_norm)
ax3.plot(range(0,test3_min),xaxis3_3_norm)

ax1.set_title('Camera 1', fontsize=10)
ax2.set_title('Camera 2', fontsize=10)
ax3.set_title('Camera 3', fontsize=10)

ax1.set_ylim([0, 1])
ax2.set_ylim([0, 1])
ax3.set_ylim([0, 1])

fig.suptitle('Test 3: x-Axis')

plt.show()


# **Apply PCA**

# In[1080]:


from numpy import linalg as LA

# Create 2 dataframes: one for x coordinates and one for y coordinates
test3_x = pd.DataFrame({'Camera 1':x_axis1_3, 'Camera 2': x_axis2_3,'Camera 3': x_axis3_3}).T
test3_y = pd.DataFrame({'Camera 1':y_axis1_3, 'Camera 2': y_axis2_3,'Camera 3': y_axis3_3}).T

# Subtract mean from each row
test3_x.iloc[0,:] = test3_x.iloc[0,:] - test3_x.iloc[0,:].mean()
test3_x.iloc[1,:] = test3_x.iloc[1,:] - test3_x.iloc[1,:].mean()
test3_x.iloc[2,:] = test3_x.iloc[2,:] - test3_x.iloc[2,:].mean()

test3_y.iloc[0,:] = test3_y.iloc[0,:] - test3_y.iloc[0,:].mean()
test3_y.iloc[1,:] = test3_y.iloc[1,:] - test3_y.iloc[1,:].mean()
test3_y.iloc[2,:] = test3_y.iloc[2,:] - test3_y.iloc[2,:].mean()

# Divide by standard deviation
test3_x.iloc[0,:] = test3_x.iloc[0,:] / np.std(test3_x.iloc[0,:])
test3_x.iloc[1,:] = test3_x.iloc[1,:] / np.std(test3_x.iloc[1,:])
test3_x.iloc[2,:] = test3_x.iloc[2,:] / np.std(test3_x.iloc[2,:])

test3_y.iloc[0,:] = test3_y.iloc[0,:] / np.std(test3_y.iloc[0,:])
test3_y.iloc[1,:] = test3_y.iloc[1,:] / np.std(test3_y.iloc[1,:])
test3_y.iloc[2,:] = test3_y.iloc[2,:] / np.std(test3_y.iloc[2,:])


# Append three rows of x values to three rows of y values
test3 = test3_x.append(test3_y)

# calculate covariance matrix
covar = (1 / (len(test3.columns)) - 1) * np.cov(test3, rowvar=True)

# Ensure covariance matrix is a square
print(covar.shape, 'notice covar is a square matrix')


# In[1087]:


# Factor covariance matrix using SVD
[U,S,V] = np.linalg.svd(test3 / np.sqrt(len(test3.columns)-1),full_matrices=False)

lambda_ = np.diag(S) **2


# In[1088]:


from scipy.linalg import svd

rho3 = (S*S) / (S*S).sum()

threshold = 0.95

plt.figure()
plt.plot(range(1,len(rho3)+1),rho3,'x-')
plt.plot(range(1,len(rho3)+1),np.cumsum(rho3),'o-')
plt.plot([1,len(rho3)],[threshold, threshold],'k--')
plt.title('Test 3: Variance explained by principal components');
plt.xlabel('Principal component');
plt.ylabel('Variance explained');
plt.legend(['Individual','Cumulative','Threshold'])
#plt.grid()
plt.show()


# **Plot Projections of Data onto Left Singular Vectors**

# In[1089]:


modes = np.matmul(U.T,test3)
modes.iloc[0,:]

plt.plot(range(0,len(test3.columns)), modes.iloc[0,:])
plt.plot(range(0,len(test3.columns)), modes.iloc[1,:])
plt.plot(range(0,len(test3.columns)), modes.iloc[2,:])

plt.title('Test 3: Projections of Data onto Left Singular Vectors')


# **Confirm the values from the manual calculation are close to those from sklearn**

# In[1090]:


from sklearn import decomposition
from sklearn import datasets

from sklearn.decomposition import PCA
pca = PCA(n_components=5, svd_solver='full')
principalComponents = pca.fit_transform(test3.T)
#pca.explained_variance_ratio_

# Check if components match using explained_variance_ from sklear
print(pca.explained_variance_[0] / sum(pca.explained_variance_), 'sklearn: the first component matches')
print(pca.explained_variance_[1] / sum(pca.explained_variance_), 'sklearn: the second component matches')


# In[1096]:


# Check if components match using eigenvalue decomposition
eig_val, eig_vec = np.linalg.eig(covar)
print(eig_val[0] / sum(eig_val), 'Evalue decomp: the first component matches')
print(eig_val[1] / sum(eig_val), 'Evalue decomp: no match (check instability of Eig Decomp)')


# In[1092]:


print(rho3[0], 'SVD: the first component matches')
print(rho3[1], 'SVD: the second component matches')


# ### Test 4: horizontal displacement and rotation

# - The dimensions of each file are as follows:  
#  - vidFrames1_2: 480,640,3,392
#  - vidFrames2_2: 480,640,3,405
#  - vidFrames3_2: 480,640,3,394

# In[810]:


files = []
for i in range(1,4):
        files.append(pd.read_csv('cam'+ str(i) + '_4.csv',header=None))


# In[811]:


cam1_4 = files[0].to_numpy()
cam2_4 = files[1].to_numpy()
cam3_4 = files[2].to_numpy()

cam1_4 = cam1_4.reshape((480,640,3,392), order='F') # Convert to ndarray, order='F' is to reshape using
# Fortran style which is first index first, whic is what Matlab uses
cam2_4 = cam2_4.reshape((480,640,3,405), order='F')
cam3_4 = cam3_4.reshape((480,640,3,394), order='F')


# In[812]:


[a14,b14,c14,d14] = cam1_4.shape 
[a24,b24,c24,d24] = cam2_4.shape
[a34,b34,c34,d34] = cam3_4.shape


# **Test 4: First Camera**

# In[813]:


x_axis1_4 = [412]
y_axis1_4 = [268]
window = 40

for i in range(1,d14):
    img = rgb2gray(cam1_4[:,:,:,i])
    img[0:y_axis1_4[-1]-window,:] = 0 # for some reason this changes the y axis
    img[window+y_axis1_4[-1]:,:] = 0
    img[:,0:x_axis1_4[-1]-window] = 0 # and this changes the x axis
    img[:,window+x_axis1_4[-1]:] = 0    #img[:,227+window:] = 0
    [y_axis,x_axis] = np.unravel_index(np.argmax(img, axis=None), img.shape)

    y_axis1_4.append(y_axis)
    x_axis1_4.append(x_axis)
    #print(y_axis2,x_axis2)


# **Test 4: Second Camera**

# In[814]:


x_axis2_4 = [278]
y_axis2_4 = [248]
window = 20

for i in range(1,d24):
    img = rgb2gray(cam2_4[:,:,:,i])
    img[0:y_axis2_4[-1]-window,:] = 0 # for some reason this changes the y axis
    img[window+y_axis2_4[-1]:,:] = 0
    img[:,0:x_axis2_4[-1]-window] = 0 # and this changes the x axis
    img[:,window+x_axis2_4[-1]:] = 0    #img[:,227+window:] = 0
    [y_axis,x_axis] = np.unravel_index(np.argmax(img, axis=None), img.shape)

    y_axis2_4.append(y_axis)
    x_axis2_4.append(x_axis)
    #print(y_axis2,x_axis2)


# **Test 4: Third Camera**

# In[815]:


x_axis3_4 = [368]
y_axis3_4 = [230]
window = 20

for i in range(1,d34):
    img = rgb2gray(cam3_4[:,:,:,i])
    img[0:y_axis3_4[-1]-window,:] = 0 # for some reason this changes the y axis
    img[window+y_axis3_4[-1]:,:] = 0
    img[:,0:x_axis3_4[-1]-window] = 0 # and this changes the x axis
    img[:,window+x_axis3_4[-1]:] = 0    #img[:,227+window:] = 0
    [y_axis,x_axis] = np.unravel_index(np.argmax(img, axis=None), img.shape)

    y_axis3_4.append(y_axis)
    x_axis3_4.append(x_axis)


# In[816]:


del cam1_4, cam2_4, cam3_4


# **Ensure all the data has the same length**

# In[817]:


test4_min = min(len(y_axis1_4),len(y_axis2_4),len(y_axis3_4))
#test1_xmin = min(len(x_axis1_1),len(x_axis2_1),len(x_axis3_1))


y_axis1_4 = y_axis1_4[0:test4_min]
y_axis2_4 = y_axis2_4[0:test4_min]
y_axis3_4 = y_axis3_4[0:test4_min]

x_axis1_4 = x_axis1_4[0:test4_min] # since x-axis and y-axis have the same length it doesn't matter which is used
x_axis2_4 = x_axis2_4[0:test4_min]
x_axis3_4 = x_axis3_4[0:test4_min]


# **Normalize the data**

# In[1097]:


yaxis1_4_norm = y_axis1_4 / max(y_axis1_4)
yaxis2_4_norm = y_axis2_4 / max(y_axis2_4)
yaxis3_4_norm = y_axis3_4 / max(y_axis3_4)

xaxis1_4_norm = x_axis1_4 / max(x_axis1_4)
xaxis2_4_norm = x_axis2_4 / max(x_axis2_4)
xaxis3_4_norm = x_axis3_4 / max(x_axis3_4)


# In[1098]:


fig,(ax1, ax2, ax3) =  plt.subplots(1,3,figsize=(15,4))

ax1.plot(range(0,test4_min),yaxis1_4_norm)
ax2.plot(range(0,test4_min),yaxis2_4_norm)
ax3.plot(range(0,test4_min),yaxis3_4_norm)

ax1.set_title('Camera 1', fontsize=10)
ax2.set_title('Camera 2', fontsize=10)
ax3.set_title('Camera 3', fontsize=10)

ax1.set_ylim([0, 1])
ax2.set_ylim([0, 1])
ax3.set_ylim([0, 1])

fig.suptitle('Test 4: y-Axis')

plt.show()


# In[1118]:


fig,(ax1, ax2, ax3) =  plt.subplots(1,3,figsize=(15,4))

ax1.plot(range(0,test4_min),xaxis1_4_norm)
ax2.plot(range(0,test4_min),xaxis2_4_norm)
ax3.plot(range(0,test4_min),xaxis3_4_norm)

ax1.set_title('Camera 1', fontsize=10)
ax2.set_title('Camera 2', fontsize=10)
ax3.set_title('Camera 3', fontsize=10)

ax1.set_ylim([0, 1])
ax2.set_ylim([0, 1])
ax3.set_ylim([0, 1])

fig.suptitle('Test 4: x-Axis')

fig.show()


# **Apply PCA**

# In[1100]:


from numpy import linalg as LA

# Create 2 dataframes: one for x coordinates and one for y coordinates
test4_x = pd.DataFrame({'Camera 1':x_axis1_4, 'Camera 2': x_axis2_4,'Camera 3': x_axis3_4}).T
test4_y = pd.DataFrame({'Camera 1':y_axis1_4, 'Camera 2': y_axis2_4,'Camera 3': y_axis3_4}).T

# Subtract mean from each row
test4_x.iloc[0,:] = test4_x.iloc[0,:] - test4_x.iloc[0,:].mean()
test4_x.iloc[1,:] = test4_x.iloc[1,:] - test4_x.iloc[1,:].mean()
test4_x.iloc[2,:] = test4_x.iloc[2,:] - test4_x.iloc[2,:].mean()

test4_y.iloc[0,:] = test4_y.iloc[0,:] - test4_y.iloc[0,:].mean()
test4_y.iloc[1,:] = test4_y.iloc[1,:] - test4_y.iloc[1,:].mean()
test4_y.iloc[2,:] = test4_y.iloc[2,:] - test4_y.iloc[2,:].mean()

# Divide by standard deviation
test4_x.iloc[0,:] = test4_x.iloc[0,:] / np.std(test4_x.iloc[0,:])
test4_x.iloc[1,:] = test4_x.iloc[1,:] / np.std(test4_x.iloc[1,:])
test4_x.iloc[2,:] = test4_x.iloc[2,:] / np.std(test4_x.iloc[2,:])

test4_y.iloc[0,:] = test4_y.iloc[0,:] / np.std(test4_y.iloc[0,:])
test4_y.iloc[1,:] = test4_y.iloc[1,:] / np.std(test4_y.iloc[1,:])
test4_y.iloc[2,:] = test4_y.iloc[2,:] / np.std(test4_y.iloc[2,:])

# Append three rows of x values to three rows of y values
test4 = test4_x.append(test4_y)

# calculate covariance matrix
covar = (1 / (len(test4.columns)) - 1) * np.cov(test4, rowvar=True)

# Ensure covariance matrix is a square
print(covar.shape, 'notice covar is a square matrix')


# In[1101]:


# Factor covariance matrix using SVD
[U,S,V] = np.linalg.svd(test4 / np.sqrt(len(test4.columns)-1))

lambda_ = np.diag(S) **2


# In[1102]:


from scipy.linalg import svd

rho4 = (S*S) / (S*S).sum()

threshold = 0.95

plt.figure()
plt.plot(range(1,len(rho4)+1),rho4,'x-')
plt.plot(range(1,len(rho4)+1),np.cumsum(rho4),'o-')
plt.plot([1,len(rho4)],[threshold, threshold],'k--')
plt.title('Test 4: Variance explained by principal components');
plt.xlabel('Principal component');
plt.ylabel('Variance explained');
plt.legend(['Individual','Cumulative','Threshold'])
#plt.grid()
plt.show()


# **Plot Projections of Data onto Left Singular Vectors**

# In[1104]:


modes = np.matmul(U.T,test4)
modes.iloc[0,:]

plt.plot(range(0,len(test4.columns)), modes.iloc[0,:])
plt.plot(range(0,len(test4.columns)), modes.iloc[1,:])
plt.plot(range(0,len(test4.columns)), modes.iloc[2,:])

plt.title('Test 4: Projections of Data onto Left Singular Vectors')


# **Confirm the values from the manual calculation are close to those from sklearn**

# In[1105]:


from sklearn import decomposition
from sklearn import datasets

from sklearn.decomposition import PCA
pca = PCA(n_components=6, svd_solver='full')
principalComponents = pca.fit_transform(test4.T)
#pca.explained_variance_ratio_

# Check if components match using explained_variance_ from sklear
print(pca.explained_variance_[0] / sum(pca.explained_variance_), 'sklearn: the first component matches')
print(pca.explained_variance_[1] / sum(pca.explained_variance_), 'sklearn: the second component matches')


# In[1106]:


# Check if components match using eigenvalue decomposition
eig_val, eig_vec = np.linalg.eig(covar)
print(eig_val[0] / sum(eig_val), 'Evalue decomp: the first component matches')
print(eig_val[1] / sum(eig_val), 'Evalue decomp: no match (check instability of Eig Decomp)')


# In[1107]:


print(rho4[0], 'SVD: the first component matches')
print(rho4[1], 'SVD: the second component matches')


# **Recap the explained variance for each test**

# In[1134]:


fig,[[ax1, ax2], [ax3, ax4]] =  plt.subplots(2,2,figsize=(10,10))

# Test 1
ax1.plot(range(1,len(rho1)+1),rho1,'x-')
ax1.plot(range(1,len(rho1)+1),np.cumsum(rho1),'o-')
ax1.plot([1,len(rho1)],[threshold, threshold],'k--')
ax1.set_title('Test 1: Var explained by PCs');
ax1.set_xlabel('Principal component');
ax1.set_ylabel('Variance explained');#
ax1.legend(['Individual','Cumulative','Threshold'])

# Test 2
ax2.plot(range(1,len(rho2)+1),rho2,'x-')
ax2.plot(range(1,len(rho2)+1),np.cumsum(rho2),'o-')
ax2.plot([1,len(rho2)],[threshold, threshold],'k--')
ax2.set_title('Test 2: Var explained by PCs');
ax2.set_xlabel('Principal component');
ax2.set_ylabel('Variance explained');
ax2.legend(['Individual','Cumulative','Threshold'])

# Test 3
ax3.plot(range(1,len(rho3)+1),rho3,'x-')
ax3.plot(range(1,len(rho3)+1),np.cumsum(rho3),'o-')
ax3.plot([1,len(rho3)],[threshold, threshold],'k--')
ax3.set_title('Test 3: Var explained by PCs');
ax3.set_xlabel('Principal component');
ax3.set_ylabel('Variance explained');
ax3.legend(['Individual','Cumulative','Threshold'])

# Test 4
ax4.plot(range(1,len(rho4)+1),rho4,'x-')
ax4.plot(range(1,len(rho4)+1),np.cumsum(rho4),'o-')
ax4.plot([1,len(rho4)],[threshold, threshold],'k--')
ax4.set_title('Test 4: Var explained by PCs');
ax4.set_xlabel('Principal component');
ax4.set_ylabel('Variance explained');
ax4.legend(['Individual','Cumulative','Threshold'])

fig.show()


# In[ ]:




