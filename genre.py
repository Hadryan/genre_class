#!/usr/bin/env python
from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.layers import Dense, Flatten, BatchNormalization,Reshape
from keras.layers import Conv2D, MaxPooling2D, LeakyReLU, Dropout, CuDNNGRU
from keras.models import Sequential
import matplotlib.pylab as plt
import numpy as np
# coding: utf-8

# <a href="https://colab.research.google.com/github/pooyaww/genre_class/blob/master/genre.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# In[ ]:


#get_ipython().system(u"rm -rf 'Data1'")


# In[5]:


#get_ipython().system(u'git clone https://github.com/saraahmadi70/musicsara.git')


# In[6]:


#ls 'musicsara/soundsamples/'


# In[ ]:


import os
dir=os.listdir('musicsara/soundsamples/')
dir.remove('ezafi')
print (dir)


# In[ ]:


get_ipython().system(u'pip install ffmpeg')
get_ipython().system(u'pip install Audiosegment')
from IPython.display import clear_output
get_ipython().system(u'apt-get -qq -y install xvfb freeglut3-dev ffmpeg> /dev/null')
import pydub
from pydub import AudioSegment
from pydub.utils import which
AudioSegment.converter = which("ffmpeg")

import glob as glob


xmin=0
n1=50
n2=100

import ntpath

try:
  os.mkdir('Data1/')
except:
  x=0

c=0
for i in range(len(dir)):
  path='musicsara/soundsamples/'+dir[i]+'/*'
  for file in sorted(glob.glob(path))[n1:n2]:
    sound = pydub.AudioSegment.from_file(file)
    sound.export('Data1/'+ntpath.basename(file)+'.wav', format="wav")
    c+=1
    if c%1==0:
      xx=int((c*100)/((n2-n1)*5))
      if xx>xmin:
        xmin=xx
        clear_output()
        print(xx,' %')

clear_output()


# In[ ]:


from scipy.io import wavfile
from scipy import interpolate
import glob as glob
import os
import numpy as np
from IPython.display import clear_output

def sing(old_audio,old_samplerate,NEW_SAMPLERATE):
  if old_samplerate != NEW_SAMPLERATE:
    duration = old_audio.shape[0] / old_samplerate
    time_old  = np.linspace(0, duration, old_audio.shape[0])
    time_new  = np.linspace(0, duration, int(old_audio.shape[0] * NEW_SAMPLERATE / old_samplerate))
    interpolator = interpolate.interp1d(time_old, old_audio.T)
    new_audio = interpolator(time_new).T 
    return(new_audio)

datal=[]
c=0
  
for filename in sorted(glob.glob('Data1/*.wav')):
  ns=4096
  s, a = wavfile.read(filename)
  
#   data=sing(a,s,ns)
  old_audio=a
  old_samplerate=s
  NEW_SAMPLERATE=ns
  
  if old_samplerate != NEW_SAMPLERATE:
    duration = old_audio.shape[0] / old_samplerate
    time_old  = np.linspace(0, duration, old_audio.shape[0])
    time_new  = np.linspace(0, duration, int(old_audio.shape[0] * NEW_SAMPLERATE / old_samplerate))
    interpolator = interpolate.interp1d(time_old, old_audio.T)
    new_audio = interpolator(time_new).T
  
  data=new_audio  
  #
  
  c+=1
  clear_output()
  print(int((c*100)/250),' %')
  
  l=0
  for i in range(1000000,0,-100000):
    if len(data)>(i+131072):
      l=i
  try:
    data=data[l:l+131072,0]
  except:
    data=data[l:l+131072]
  datal.append(data)
  


# In[ ]:


datal=np.array(datal)


# In[ ]:


datal.shape


# In[ ]:


import numpy as np
import pickle as plk
with open('./SaraDatal2.plk', 'wb') as fn:
  plk.dump(datal,fn)


# 

# 

# In[3]:


##from google.colab import drive
##drive.mount('/content/drive')
##get_ipython().system(u"ls 'drive/My Drive'")


# In[ ]:


##get_ipython().magic(u'reset')
import numpy as np
import pickle as plk
with open('./SaraDatal2.plk', 'rb') as fn:
  datal=plk.load(fn)


# In[ ]:


#get_ipython().system(u"rm -rf 'Data'")
#import numpy as np
#import pickle as plk
with open('./SaraDatal1.plk', 'rb') as fn:
  datal1=plk.load(fn)


# In[ ]:


data=[]
for i in range(len(datal)):
  data.append(datal[i])
  data.append(datal1[i])

  
len(data)
# import numpy as np
# data=np.array(data)
# print(data.shape)


# In[ ]:


import numpy as np
d=np.array(data)
print(d.shape)


# In[ ]:


# !pip install librosa
import librosa
import librosa.core as lc
import librosa.display
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from IPython.display import clear_output
clear_output()
sts=[]
c=0
for i in range(len(d)):
  n_f=510
  hl=514
# st = lc.stft(sounds[0],n_fft=n_f,hop_length=hl)
# st = lc.stft(sounds[0])
  y=d[i]
  n = len(y)
  n_fft = n_f
#   y_pad = librosa.util.fix_length(y, n + n_fft // 2)
  D = librosa.stft(y, n_fft=n_fft, hop_length=hl)
#   y_out = librosa.istft(D, length=n, hop_length=hl)
  D1=librosa.amplitude_to_db(D, ref=np.max)+80
  sts.append(D1)


# axs.ravel()
D=D.astype(int)
librosa.display.specshow(librosa.amplitude_to_db(D,
                          ref=np.max),
                          y_axis='linear', x_axis='time')
plt.title('Power spectrogram')
plt.colorbar(format='%+2.0f dB')
plt.tight_layout()
#   plt.axis("off")
#   plt.savefig('plot.jpg',bbox_inches='tight', pad_inches=0)
#   img=mpimg.imread('plot.jpg')
#   plt.axis("off")
#   img.setflags(write=1)
#   x=img
#   x[297:len(img),:,:]=0
#   x[:,0:50,:]=0
#   x=x[0:298,50:len(img[0]),:]
#   sts.append(x)
#   clear_output()
#   c+=1
#   if c%5==0:
#     print(c)


# In[ ]:


sts=np.array(sts)
sts.shape


# In[ ]:


import numpy as np
import pickle as plk
with open('./stsSara.plk', 'wb') as fn:
  plk.dump(sts,fn)


# In[ ]:


# %reset

import numpy as np
import pickle as plk
with open('./stsSara.plk', 'rb') as fn:
  data=plk.load(fn)
data.shape


# In[ ]:


y=[]
for i in range(len(data)):
  y.append(np.floor(i/100))

y=np.array(y)

xtr=[]
ytr=[]
xte=[]
yte=[]

for i in range(5):
  for j in range(80):
    r=np.random.rand()*99
    r=np.floor(r)
    r=int(r)
    xtr.append(data[i*100+r])
    ytr.append(y[i*100+r])
  for j in range(20):
    r=np.random.rand()*99
    r=np.floor(r)
    r=int(r)
    xte.append(data[i*100+r])
    yte.append(y[i*100+r])

xtr=np.array(xtr)
xte=np.array(xte)
ytr=np.array(ytr)
yte=np.array(yte)

print(xtr.shape)
print(ytr.shape)
print(xte.shape)
print(yte.shape)


# In[ ]:


y


# In[ ]:


xtr


# In[ ]:


#@title Beloved Deep Learning
##from __future__ import print_function
##import keras
##from keras.datasets import mnist
##from keras.layers import Dense, Flatten, BatchNormalization,Reshape
##from keras.layers import Conv2D, MaxPooling2D, LeakyReLU, Dropout, CuDNNGRU
##from keras.models import Sequential
##import matplotlib.pylab as plt
##import numpy as np
batch_size = 10
num_classes = 5
epochs = 50

# input image dimensions
img_x, img_y = 256, 256
x_train=xtr
y_train=ytr
x_test=xte
y_test=yte
# x_train /= 1
# reshape the data into a 4D tensor - (sample_number, x_img_size, y_img_size, num_channels)
# because the MNIST is greyscale, we only have a single channel - RGB colour images would have 3
x_train = x_train.reshape(x_train.shape[0], img_x, img_y, 1)
x_test = x_test.reshape(x_test.shape[0], img_x, img_y, 1)
# x_test = x_test.reshape(x_test.shape[0], img_x, img_y, 1)
input_shape = (img_x, img_y, 1)

#convert the data to the right type
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
# y_test = keras.utils.to_categorical(y_test, num_classes)

lre=0.0
dr1=0
dr2=0
dr3=0
dr4=0
dr5=0

fn=8
nn=2
ks=3

model = Sequential()
model.add(Conv2D(fn, kernel_size=ks, input_shape=input_shape, padding='same'))
model.add(LeakyReLU(alpha=lre))
model.add(Dropout(dr1))
model.add(BatchNormalization(momentum=0.8))
model.add(MaxPooling2D(pool_size=(2, 2)))
fn=fn*nn
model.add(Conv2D(fn, ks, padding='same'))
model.add(LeakyReLU(alpha=lre))
model.add(Dropout(dr2))
model.add(BatchNormalization(momentum=0.8))
model.add(MaxPooling2D(pool_size=(2, 2)))
fn=fn*nn
model.add(Conv2D(fn, ks, padding='same'))
model.add(LeakyReLU(alpha=lre))
model.add(Dropout(dr3))
model.add(BatchNormalization(momentum=0.8))
model.add(MaxPooling2D(pool_size=(2, 2)))
fn=fn*nn
model.add(Conv2D(fn, ks, padding='same'))
model.add(LeakyReLU(alpha=lre))
model.add(Dropout(dr4))
model.add(BatchNormalization(momentum=0.8))
model.add(MaxPooling2D(pool_size=(2, 2)))
fn=fn*nn
model.add(Conv2D(fn, ks, padding='same'))
model.add(LeakyReLU(alpha=lre))
model.add(BatchNormalization(momentum=0.8))
model.add(MaxPooling2D(pool_size=(2, 2)))
conv_to_gru=(64,fn)
model.add(Reshape(target_shape=conv_to_gru))
# model.add(CuDNNGRU(25, return_sequences=True))
# model.add(CuDNNGRU(50, return_sequences=False))
model.add(Dropout(dr5))
model.add(Flatten())
model.add(Dense(3))
model.add(LeakyReLU(alpha=0))
model.add(Dense(num_classes, activation='softmax'))



opt=keras.optimizers.Adam(lr=0.0001)

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=opt,
              metrics=['accuracy'])


# opt=keras.optimizers.Adam(lr=0.01)

# model.compile(loss=keras.losses.categorical_crossentropy,
#               optimizer=opt,
#               metrics=['accuracy'])

# y_integers = np.argmax(y, axis=1)
# from sklearn.utils import compute_class_weight
# classWeight = compute_class_weight('balanced', (y_integers), y_integers) 
# classWeight = dict(enumerate(classWeight))

history=model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=2,
        #   validation_data=(x_test, y_test),
        #  class_weight=classWeight)
          validation_data=(x_test, y_test))
#
#model.summary()


# In[ ]:





# 

# In[ ]:


import matplotlib.pyplot as plt
plt.figure()
#plt.plot(s _losses)
plt.plot(val_losses)
plt.show()


# In[ ]:


from sklearn.metrics import classification_report,confusion_matrix
import numpy as np

y=y_test
x=xte

y_true = np.argmax(y, axis=1)
y_pred1 = model.predict(x_test)
y_pred=np.argmax(y_pred1,axis=1)
print(classification_report(y_true, y_pred))


# In[ ]:


history_dict = history.history
history_dict.keys()


# In[ ]:


from keras.utils.vis_utils import plot_model
plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)

from IPython.display import Image
Image(retina=True, filename='model.png')

