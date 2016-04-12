#path to caffe root
caffe_root = '/home/mani/caffe/' 

#path to Yelp dataset
data_root = '/home/mani/Yelp/'

import numpy as np
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe
import os
#download pre-trained caffemodel 
if not os.path.isfile(caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'):
    print("Downloading pre-trained CaffeNet model...")
    !caffe_root/scripts/download_model_binary.py ../models/bvlc_reference_caffenet

#module for image feature extraction
def image_feature_extract(images, layer = 'fc7'):
	#initialize the pretrainednetwork
    net = caffe.Net(caffe_root + 'models/bvlc_reference_caffenet/deploy.prototxt',
                caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel',caffe.TEST)
    
    #preprocessing the input
    transformer = caffe.io.Transformer({'data':net.blobs['data'].data.shape})
    transformer.set_transpose('data',(2,0,1))
    
    #set the mean
    transformer.set_mean('data', np.load(caffe_root+'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1))
    
    #set the raw scale which operates on images in [0,255] range
    transformer.set_raw_scale('data', 255)
    
    #setting the channel order in BGR order
    transformer.set_channel_swap('data', (2,1,0))  

    #calculate the no.of images
    num_images= len(images)
    
    net.blobs['data'].reshape(num_images,3,227,227)
    net.blobs['data'].data[...] = map(lambda x: transformer.preprocess('data',caffe.io.load_image(x)), images)
    out = net.forward()

    return net.blobs[layer].data   

