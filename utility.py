import numpy as np
import time
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def MeanPerChannelDataset(path=None, color_mode='rgb', size=(224,224)):
    '''
    Get mean per-channel of entire dataset.
    Using time, Numpy, Tensorflow, and Keras.
    Get image from folder.
    Return mean per-channel.
    '''
    datagen_mpcd = ImageDataGenerator()
    gen_mpcd = datagen_mpcd.flow_from_directory(directory=path,
                                                target_size=size,
                                                color_mode=color_mode,
                                                batch_size=1)
    sum_ = np.zeros(3)
    count = 0
    t = time.time()
    samples = gen_mpcd.samples
    for x, y in gen_mpcd:
        count = count + 1
        if(count%1000==0):
                print("Counting {}/{} images".format(count,samples))

        if(count<=samples):
            img = x.reshape(224,224,-1)
            sum_ = sum_ + (np.sum(img, axis=(0,1)))

        else:
            mean_per_channel = sum_/(size[0]*size[1]*samples)
            print('Done!! elapsed:{}'.format(time.time()- t))
            print(mean_per_channel)
            return mean_per_channel

def StdPerChannelDataset(path=None, color_mode='rgb', size=(224,224), mean_per_channel=[0,0,0]):
    '''
    Get mean std per-channel of entire dataset.
    Using time, Numpy, Tensorflow, and Keras.
    Get image from folder.
    Return std per-channel.
    '''
    datagen_mpcd = ImageDataGenerator()
    gen_mpcd = datagen_mpcd.flow_from_directory(directory=path,
                                                target_size=size,
                                                color_mode=color_mode,
                                                batch_size=1)
    dist_ = np.zeros(3)
    count = 0
    t = time.time()
    samples = gen_mpcd.samples
    for x, y in gen_mpcd:
        count = count + 1
        if(count%1000==0):
            print("Counting {}/{} images".format(count,samples))

        if(count<=samples):
            img = x.reshape(224,224,-1)
            for i in range(3):
                img[:,:,i] = (img[:,:,i]-mean_per_channel[i])**2
            dist_ = dist_ + (np.sum(img, axis=(0,1)))
        else:
            std_per_channel = dist_/(size[0]*size[1]*samples)
            print('Done!! elapsed:{}'.format(time.time()- t))
            print(std_per_channel)
            return std_per_channel