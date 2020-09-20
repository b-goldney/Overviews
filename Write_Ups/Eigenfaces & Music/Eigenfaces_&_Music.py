#!/usr/bin/env python
# coding: utf-8

# ### Homework 4: Eigenfaces & Music Genre Identification
# Github: https://github.com/b-goldney/AMATH_582

# **Load data**

# In[132]:


import glob
path = '/Users/brandongoldney/Documents/U_Washington/AMATH_582/HW_4/CroppedYale/'

files = [f for f in glob.glob(path + "**/*.pgm", recursive=True)]

for f in files:
    print(f)


# In[133]:


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage.color import rgb2gray
from numpy import asarray
from PIL import Image
import scipy.misc

yale_faces = np.zeros((32256,len(files)))
#np.ndarray(shape=(32256,10), dtype=float, order='F')
for f in range(0,len(files)):
    data = Image.open(files[f])
    yale_faces[:,f] = np.reshape(data,((192*168)))


# **Show the "Average" Face**

# In[160]:


mean_image = np.mean(yale_faces, axis=1)
plt.imshow(np.reshape(mean_image,[192,168]), cmap = plt.get_cmap("gray"))
plt.xlabel('x axis pixel range')
plt.ylabel('y axis pixel range')
plt.title('The Average Face')


# **Normalize the Faces**

# In[124]:


yale_norm = np.zeros((32256,len(files)))
for i in range(0,len(files)):
    yale_norm[:,i] = yale_faces[:,i] - mean_image


# **PCA**

# In[137]:


from sklearn.decomposition.pca import PCA

pca = PCA()
pca.fit(yale_norm)


# In[155]:


# Let's make a scree plot
pve = pca.explained_variance_ratio_
pve.shape
plt.plot(range(len(pve)), pve)
plt.title("Eigenfaces Scree Plot")
plt.ylabel("Variance Explained")
plt.xlabel("Principal Components")
plt.xlim(0,50)


# In[147]:


# eigenfaces
eigenfaces = pca.components_
#plt.imshow(np.reshape(eigenfaces,[192,168]), cmap = plt.get_cmap("gray"),dpi=300)


# In[161]:


img_idx = yale_faces[0]
loadings = pca.components_
n_components = loadings.shape[0]
scores = np.dot(yale_norm[:,:], loadings[:,:].T)

img_proj = []
for n in range(n_components):
    proj = np.dot(scores[img_idx, n], loadings[n,:])
    img_proj.append(proj)
len(img_proj)


# **Music**

# In[ ]:


# read in triples of user/artist/playcount from the input dataset
data = pandas.read_table("usersha1-artmbid-artname-plays.tsv", 
                         usecols=[0, 2, 3], 
                         names=['user', 'artist', 'plays'])

# map each artist and user to a unique numeric value
data['user'] = data['user'].astype("category")
data['artist'] = data['artist'].astype("category")

# create a sparse matrix of all the artist/user/play triples
plays = coo_matrix((data['plays'].astype(float), 
                   (data['artist'].cat.codes, 
                    data['user'].cat.codes)))


# In[ ]:


class TopRelated(object):
    def __init__(self, artist_factors):
        # fully normalize artist_factors, so can compare with only the dot product
        norms = numpy.linalg.norm(artist_factors, axis=-1)
        self.factors = artist_factors / norms[:, numpy.newaxis]

    def get_related(self, artistid, N=10):
        scores = self.factors.dot(self.factors[artistid])
        best = numpy.argpartition(scores, -N)[-N:]
        return sorted(zip(best, scores[best]), key=lambda x: -x[1])


# In[ ]:


def alternating_least_squares(Cui, factors, regularization, iterations=20):
    users, items = Cui.shape

    X = np.random.rand(users, factors) * 0.01
    Y = np.random.rand(items, factors) * 0.01

    Ciu = Cui.T.tocsr()
    for iteration in range(iterations):
        least_squares(Cui, X, Y, regularization)
        least_squares(Ciu, Y, X, regularization)

    return X, Y

def least_squares(Cui, X, Y, regularization):
    users, factors = X.shape
    YtY = Y.T.dot(Y)

    for u in range(users):
        # accumulate YtCuY + regularization * I in A
        A = YtY + regularization * np.eye(factors)

        # accumulate YtCuPu in b
        b = np.zeros(factors)

        for i, confidence in nonzeros(Cui, u):
            factor = Y[i]
            A += (confidence - 1) * np.outer(factor, factor)
            b += confidence * factor

        # Xu = (YtCuY + regularization * I)^-1 (YtCuPu)
        X[u] = np.linalg.solve(A, b)


# In[ ]:


frequencies = np.arange(5,105,5)
# Sampling Frequency
samplingFrequency= 400

# Create ndarrays
s1 = np.empty([0]) 

s2 = np.empty([0]) 

# Start Value of the sample
start = 1

# Stop Value of the sample

stop= samplingFrequency+1

for frequency in frequencies:
    sub1 = np.arange(start, stop, 1)
    # Signal - Sine wave with varying frequency + Noise
    sub2 = np.sin(2*np.pi*sub1*frequency*1/samplingFrequency)+np.random.randn(len(sub1))
    s1      = np.append(s1, sub1)
    s2      = np.append(s2, sub2)
    start   = stop+1
    stop    = start+samplingFrequency

# Plot the signal
plot.subplot(211)
plot.plot(s1,s2)
plot.xlabel('Sample')
plot.ylabel('Amplitude')
 

 

# Plot the spectrogram

plot.subplot(212)
powerSpectrum, freqenciesFound, time, imageAxis = plot.specgram(s2, Fs=samplingFrequency)
plot.xlabel('Time')
plot.ylabel('Frequency')


plot.show()   


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(scaled_df, y, test_size=0.2, random_state=3)
forest = RandomForestClassifier(n_estimators=100, max_depth= 5)
forest.fit(X_train, y_train)


# In[ ]:


def plot_feature_importances(model):
    n_features = X_train.shape[1]
    plt.figure(figsize=(8,8))
    plt.barh(range(n_features), model.feature_importances_, align='center') 
    plt.yticks(np.arange(n_features), X_train.columns.values) 
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")


# In[ ]:


pred = forest.predict(X_test)
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))


# In[ ]:


# From BriansRebsnik
def plot_feature_importances(model):
    n_features = X_train.shape[1]
    plt.figure(figsize=(8,8))
    plt.barh(range(n_features), model.feature_importances_, align='center') 
    plt.yticks(np.arange(n_features), X_train.columns.values) 
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")


# In[ ]:



pred = forest.predict(X_test)
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))

