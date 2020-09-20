#!/usr/bin/env python
# coding: utf-8

# **Github**  
# Link: https://github.com/b-goldney/AMATH_582  
# Username: b-goldney

# **Appendix B: Python Code**

# Part I: Handel’s Messiah

# In[2]:


import sounddevice as sd
import numpy as np
import matplotlib. pyplot as plt
from scipy import signal
#from scipy.fft import fftshift
import matplotlib.pyplot as plt

y = np.load('y.npy')
Fs = np.load('Fs.npy') # The elements of a 0d array can be accessed via: Fs[()]
sd.play(y,Fs)


# Note, on the above:  
# - Fs is sampling frequency. The sampling frequency (or sample rate) is the number of samples per second in a Sound. For example: if the sampling frequency is 44100 hertz, a recording with a duration of 60 seconds will contain 2,646,000 samples.
# - y is an array with 73,113 elements.  It's 8.9 seconds long (73,113 samples / 8192 samples per second = 8.924 seconds)

# In[3]:


# Incorporate given code from the homework prompt
v = y/2; # v is simply the original amplitude divided by 2

plt.figure(figsize=(20,10)), plt.tick_params(axis='both', which='major', labelsize=24)
plt.xlabel('Time [sec]', fontsize=24), plt.ylabel('Amplitude',fontsize=24)
plt.title('Handel\'s Messiah', fontsize=24)

plt.plot(range(0, len(v))/Fs,v); 


# In[4]:


L = len(y)/Fs
n = len(v) # v is simply the original amplitude divided by 2
t2 = range(0,n+1)/ Fs # this is length of amplitude divided by 8,192 (sampling frequency). 
t = t2[0:n+1]

# Create k variable
list1 = np.linspace(0,n/2-1,num=(n/2-1*.0000001)) # n is length of amplitude
list2 = np.linspace(-n/2,-1,num=(n/2-1*.0000001))
list3 = []

for i in list1:
    list3.append(i)
    
for i in list2:
    list3.append(i)

# Convert list3 to a numpy array so it can be multiplied
array3 = np.asarray(list3)

# Rescale wavenumbers because FFT assumes 2*pi periodic signals
k = (2*np.pi / L)* array3 


# Note on the above:
# - variable t2 is the length of amplitude divided by 8,192 (i.e. sampling frequency).  The purpose of this is to create frame size.  Frame size is the total time (T) to acquire one block of data.
# - The block size (N) is the total number of time data points that are captured to perform a Fourier transform. A block size of 2000 means that two thousand data points are acquired, then a Fourier transform is performed.
# - For example, with a block size of 2000 data points and a sampling rate of 1000 samples per second, the total time to acquire a single data block is 2 seconds. It takes two seconds to collect 2000 data points.
# - Source: [Digital Signal Processing: Sampling Rates, Bandwidth, Spectral Lines, and more…](https://community.sw.siemens.com/s/article/digital-signal-processing-sampling-rates-bandwidth-spectral-lines-and-more)
# - In other words, t2 tells us how much time is required to obtain the ith element of time in t2

# In[5]:


get_ipython().run_line_magic('matplotlib', 'inline')

# Take FFT of data and and fftshift to create plot of the entire song
ks = np.fft.fftshift(k)
vt = np.fft.fft(v[0:-1]) # v is the amplitude divided by 2

plt.figure(figsize=(20,10)), plt.tick_params(axis='both', which='major', labelsize=24)
plt.xlabel('Frequency', fontsize=24), plt.ylabel('Amplitude',fontsize=24)
plt.title('FFT of Amplitude - Handel\'s Messiah', fontsize=24)

plt.plot(ks,np.fft.fftshift(vt[0:73112]))


# **Set up the Gabor Filter**

# In[6]:


tslide = np.linspace(0,9,91)

window_spc = []
big_window_spc = []
small_window_spc = []

gauss_spc = []
gauss_big_spc = []
gauss_small_spc = []

mexican_hat_spc = []
mexican_big_spc = []
mexican_small_spc = []

shannon_spc = []
shannon_big_spc = []
shannon_small_spc = []

for i in range(len(tslide)):
    window = np.exp(-1*(t-tslide[i])**10) # t is the length of amplitude divided by 8,192 (sampling frequency)
    big_window = np.exp(-0.01*(t-tslide[i])**10)
    small_window = np.exp(-100*(t-tslide[i])**10)
    gauss_window = np.exp(-(t-tslide[i])**2)
    gauss_big = np.exp(-0.01*(t-tslide[i])**2)
    gauss_small = np.exp(-10*(t-tslide[i])**2)
    mexican_hat = (1-(t-tslide[i])**2)*np.exp(-(t-tslide[i])**2)
    mexican_hat_big = ((1-0.1)*(t-tslide[i])**2)*np.exp(-0.1*(t-tslide[i])**2)
    mexican_hat_small = ((1-10)*(t-tslide[i])**2)*np.exp(-10*(t-tslide[i])**2)
    shannon_fcn = abs(t-tslide[i]) <= 0.5
    shannon_fcn_big = 0.1 * abs(t-tslide[i]) <= 0.5
    shannon_fcn_small = 10 * abs(t-tslide[i]) <= 0.5
    
    window_vf = window[0:-1] * v # v is simply the original amplitude divided by 2
    window_vft = np.fft.fft(window_vf)
    window_spc.append(abs(np.fft.fftshift(window_vft[0:-2])))
    
    big_window_vf = big_window[0:-1] * v # v is simply the original amplitude divided by 2
    big_window_vft = np.fft.fft(big_window_vf)
    big_window_spc.append(abs(np.fft.fftshift(big_window_vft[0:-2])))
    
    small_window_vf = small_window[0:-1] * v # v is simply the original amplitude divided by 2
    small_window_vft = np.fft.fft(small_window_vf)
    small_window_spc.append(abs(np.fft.fftshift(small_window_vft[0:-2])))
    
    gauss_window_vf = gauss_window[0:-1] * v # v is simply the original amplitude divided by 2
    gauss_window_vft = np.fft.fft(gauss_window_vf)
    gauss_spc.append(abs(np.fft.fftshift(gauss_window_vft[0:-2])))
    
    gauss_big_vf = gauss_big[0:-1] * v # v is simply the original amplitude divided by 2
    gauss_big_vft = np.fft.fft(gauss_big_vf)
    gauss_big_spc.append(abs(np.fft.fftshift(gauss_big_vft[0:-2])))
    
    gauss_small_vf = gauss_small[0:-1] * v # v is simply the original amplitude divided by 2
    gauss_small_vft = np.fft.fft(gauss_small_vf)
    gauss_small_spc.append(abs(np.fft.fftshift(gauss_small_vft[0:-2])))
    
    mexican_vf = mexican_hat[0:-1] * v # v is simply the original amplitude divided by 2
    mexican_vft = np.fft.fft(mexican_vf)
    mexican_hat_spc.append(abs(np.fft.fftshift(mexican_vft[0:-2])))
    
    mexican_big_vf = mexican_hat_big[0:-1] * v # v is simply the original amplitude divided by 2
    mexican_big_vft = np.fft.fft(mexican_big_vf)
    mexican_big_spc.append(abs(np.fft.fftshift(mexican_big_vft[0:-2])))
    
    mexican_small_vf = mexican_hat_small[0:-1] * v # v is simply the original amplitude divided by 2
    mexican_small_vft = np.fft.fft(mexican_small_vf)
    mexican_small_spc.append(abs(np.fft.fftshift(mexican_small_vft[0:-2])))
    
    shannon_vf = shannon_fcn[0:-1] * v # v is simply the original amplitude divided by 2
    shannon_vft = np.fft.fft(shannon_vf)
    shannon_spc.append(abs(np.fft.fftshift(shannon_vft[0:-2])))
    
    shannon_big_vf = shannon_fcn_big[0:-1] * v # v is simply the original amplitude divided by 2
    shannon_big_vft = np.fft.fft(shannon_big_vf)
    shannon_big_spc.append(abs(np.fft.fftshift(shannon_big_vft[0:-2])))
    
    shannon_small_vf = shannon_fcn_small[0:-1] * v # v is simply the original amplitude divided by 2
    shannon_small_vft = np.fft.fft(shannon_small_vf)
    shannon_small_spc.append(abs(np.fft.fftshift(shannon_small_vft[0:-2])))


# **note, in order for the graphs below to work, the "c" value argument to pcolormesh needs to be transposed for some reason**

# In[14]:


#fig, (ax1,ax2,ax3) = plt.subplots(nrows=3, ncols=1, figsize=(10,10), sharex=True, sharey=True)

# Create graphs
ax1.pcolormesh(tslide, np.fft.fftshift(k[0:-1]),np.array(window_spc).transpose())
ax2.pcolormesh(tslide, np.fft.fftshift(k[0:-1]),np.array(big_window_spc).transpose())
ax3.pcolormesh(tslide, np.fft.fftshift(k[0:-1]),np.array(small_window_spc).transpose())

# Set titles
ax1.title.set_text('Standard Wavelet: Normal Window')
ax2.title.set_text('Standard Wavelet: Big Window')
ax3.title.set_text('Standard Wavelet: Small Window')

# Note, I intentionally did not label each x-axis because it makes everything too cluttered
ax3.set_xlabel('Time (seconds)') 

ax1.set_ylabel('Frequency (Hz)')
ax2.set_ylabel('Frequency (Hz)')
ax3.set_ylabel('Frequency (Hz)')

# Set y-axis limits
ax1.set_ylim([0,8000])


# In[42]:


#fig, (ax1,ax2,ax3) = plt.subplots(nrows=3, ncols=1, figsize=(10,10), sharex=True, sharey=True)

# Create graphs
ax1.pcolormesh(tslide, np.fft.fftshift(k[0:-1]),np.array(gauss_spc).transpose())
ax2.pcolormesh(tslide, np.fft.fftshift(k[0:-1]),np.array(gauss_big_spc).transpose())
ax3.pcolormesh(tslide, np.fft.fftshift(k[0:-1]),np.array(gauss_small_spc).transpose())

# Set titles
ax1.title.set_text('Gaussian Wavelet: Normal Window')
ax2.title.set_text('Gaussian Wavelet: Big Window')
ax3.title.set_text('Gaussian Wavelet: Small Window')

# Note, I intentionally did not label each x-axis because it makes everything too cluttered
ax3.set_xlabel('Time (seconds)') 

ax1.set_ylabel('Frequency (Hz)')
ax2.set_ylabel('Frequency (Hz)')
ax3.set_ylabel('Frequency (Hz)')

# Set y-axis limits
ax1.set_ylim([0,8000])


# In[36]:


#fig, (ax1,ax2,ax3) = plt.subplots(nrows=3, ncols=1, figsize=(10,10), sharex=True, sharey=True)

# Create graphs
ax1.pcolormesh(tslide, np.fft.fftshift(k[0:-1]),np.array(mexican_hat_spc).transpose())
ax2.pcolormesh(tslide, np.fft.fftshift(k[0:-1]),np.array(mexican_big_spc).transpose())
ax3.pcolormesh(tslide, np.fft.fftshift(k[0:-1]),np.array(mexican_small_spc).transpose())

# Set titles
ax1.title.set_text('Mexican Hat Wavelet: Normal Window')
ax2.title.set_text('Mexican Hat Wavelet: Big Window')
ax3.title.set_text('Mexican Hat Wavelet: Small Window')

# Note, I intentionally did not label each x-axis because it makes everything too cluttered
ax3.set_xlabel('Time (seconds)') 

ax1.set_ylabel('Frequency (Hz)')
ax2.set_ylabel('Frequency (Hz)')
ax3.set_ylabel('Frequency (Hz)')

# Set y-axis limits
ax1.set_ylim([0,8000])


# In[37]:


fig, (ax1,ax2,ax3) = plt.subplots(nrows=3, ncols=1, figsize=(10,10), sharex=True, sharey=True)

# Create graphs
ax1.pcolormesh(tslide, np.fft.fftshift(k[0:-1]),np.array(shannon_spc).transpose())
ax2.pcolormesh(tslide, np.fft.fftshift(k[0:-1]),np.array(shannon_big_spc).transpose())
ax3.pcolormesh(tslide, np.fft.fftshift(k[0:-1]),np.array(shannon_small_spc).transpose())

# Set titles
ax1.title.set_text('Shannon Function: Normal Window')
ax2.title.set_text('Shannon Function: Big Window')
ax3.title.set_text('Shannon Function: Small Window')

# Note, I intentionally did not label each x-axis because it makes everything too cluttered
ax3.set_xlabel('Time (seconds)') 

ax1.set_ylabel('Frequency (Hz)')
ax2.set_ylabel('Frequency (Hz)')
ax3.set_ylabel('Frequency (Hz)')

# Set y-axis limits
ax1.set_ylim([0,8000])


# **Part II: Mary Had a Little Lamb (Piano)**

# In[11]:


import wave
import soundfile as sf

#tr_piano = 16
data, samplerate = sf.read('music1.wav')

L = len(data) / samplerate
n = 701440
t2 = np.linspace(0,L, n+1)
t = t2[0:n]

# Create k variable
list1 = np.linspace(0,n/2-1,num=(n/2-1*.0000001)) # n is length of amplitude
list2 = np.linspace(-n/2,-1,num=(n/2-1*.0000001))
list3 = []

for i in list1:
    list3.append(i)
    
for i in list2:
    list3.append(i)

# Convert list3 to a numpy array so it can be multiplied
array3 = np.asarray(list3)

k = (2*np.pi / L)* array3 # Rescale wavenumbers because FFT assumes 2*pi periodic signals


# In[12]:


t_slide = np.linspace(0,16,16/.1)
Max = []
spc = []

for i in range(1,len(t_slide)):
    g = np.exp(-20*(t-t_slide[i])**2)
    Sf = g[1:]*(data[0:-1]); # signal filter
    Sft = np.fft.fft(Sf);
    f_max = max(Sft)
    index_max = np.argmax(np.real(Sft))
    Max.append(k[index_max])
    spc.append(abs(np.fft.fftshift(Sft)))
    


# In[13]:


plt.plot(t_slide[0:-1], abs(np.array(Max)/(2*np.pi)))
plt.title('Mary Had a Little Lamb (Piano)')
plt.xlabel('Time (seconds)'), plt.ylabel('Frequency (Hz)')


# In[14]:


fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(10,10), sharex=True, sharey=True)
ax1.pcolormesh(t_slide[0:-1], np.fft.fftshift(k), np.array(spc).transpose()[0:-1,:])
#ax1.pcolormesh(tslide, np.fft.fftshift(k[0:-1]),np.array(mexican_hat_spc).transpose())
plt.ylim(0,3000)
plt.title("Mary Had a Little Lamb (Piano)")
plt.xlabel("Time (seconds)")
plt.ylabel('Frequency (Hz)')
plt.show()


# **Part II: Mary Had a Little Lamb (Recorder)**

# In[15]:


tr_rec = 14
data, samplerate = sf.read('music2.wav')

L = len(data) / samplerate
n = 627712
t2 = np.linspace(0,L, n+1)
t = t2[0:n]

# Create k variable
list1 = np.linspace(0,n/2-1,num=(n/2-1*.0000001)) # n is length of amplitude
list2 = np.linspace(-n/2,-1,num=(n/2-1*.0000001))
list3 = []

for i in list1:
    list3.append(i)
    
for i in list2:
    list3.append(i)

# Convert list3 to a numpy array so it can be multiplied
array3 = np.asarray(list3)

k = (2*np.pi / L)* array3 # Rescale wavenumbers because FFT assumes 2*pi periodic signals


# In[16]:


t_slide = np.linspace(0,L,L/.1)
Max = []
spc = []

for i in range(1,len(t_slide)):
    g = np.exp(-20*(t-t_slide[i])**2)
    Sf = g[1:]*(data[0:-1]); # signal filter
    Sft = np.fft.fft(Sf);
    f_max = max(Sft)
    index_max = np.argmax(np.real(Sft))
    Max.append(k[index_max])
    spc.append(abs(np.fft.fftshift(Sft)))
    


# In[17]:


plt.plot(t_slide[0:-1], abs(np.array(Max)/(2*np.pi)))
plt.title('Mary Had a Little Lamb (Recorder)')
plt.xlabel('Time (seconds)'), plt.ylabel('Frequency (Hz)')


# In[18]:


fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(10,10), sharex=True, sharey=True)
ax1.pcolormesh(t_slide[0:-1], np.fft.fftshift(k), np.array(spc).transpose()[0:-1,:])
#ax1.pcolormesh(tslide, np.fft.fftshift(k[0:-1]),np.array(mexican_hat_spc).transpose())
plt.ylim(4000,8000)
plt.title("Mary Had a Little Lamb (Recorder)")
plt.xlabel("Time (seconds)")
plt.ylabel('Frequency (Hz)')
plt.show()


# In[ ]:





# In[ ]:




