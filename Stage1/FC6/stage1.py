#stage1 is a feature extraction phase with each final feature vector of 4096 dimention
#currently used fc6 layer 

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
def image_feature_extract(images, layer = 'fc6'):
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

#feature extraction starts here and results are stored in the form of h5 file format

import h5py

#store test images features to trainedimages_fc6_features.h5
trainedImagesFeaturesFile = h5py.File(data_root+'trainedimages_fc6_features.h5','w')

#setting filenames formats
filenames = trainedImagesFeaturesFile.create_dataset('photo_id',(0,), maxshape=(None,),dtype='|S16')

#set feature size
feature = trainedImagesFeaturesFile.create_dataset('feature',(0,4096), maxshape = (None,4096))

trainedImages.close()

import pandas as pd 

#read the train photos business ids
train_photos = pd.read_csv(data_root+'train_photo_to_biz_ids.csv')
train_folder = data_root+'train_photos/'

#array for train images ids 
train_pimages = [str(x) for x in train_photos['photo_id']]
#array for train image paths
train_images = [os.path.join(train_folder, str(x)+'.jpg') for x in train_photos['photo_id']]

size_trainImages = len(train_images)
#set batch size to 1000
batch_size = 1000

#start feature extraction for train images with given batch size
for i in range(0, size_trainImages, batch_size): 
    images = train_images[i: min(i+batch_size, size_trainImages)]
    pimages = train_pimages[i: min(i+batch_size, size_trainImages)]

    #extract features for set of batch
    features = image_feature_extract(images, layer='fc6')
    done_num = i+features.shape[0]

    #file resizing happens here to accomodate new batch features 
    features_file= h5py.File(data_root+'trainedimages_fc6_features.h5','r+')
    features_file['photo_id'].resize((done_num,))
    features_file['photo_id'][i: done_num] = np.array(pimages)
    features_file['feature'].resize((done_num,features.shape[1]))
    features_file['feature'][i: done_num, :] = features
    features_file.close()

    if done_num%10000==0 or done_num==size_trainImages:
        print "Processed Train images: ", done_num

#store test images features to testImages_fc6features.h5
testImagesFeaturesFiles = h5py.File(data_root+'testImages_fc6features.h5','w')
filenames = testImagesFeaturesFiles.create_dataset('photo_id',(0,), maxshape=(None,),dtype='|S16')
feature = testImagesFeaturesFiles.create_dataset('feature',(0,4096), maxshape = (None,4096))
testImagesFeaturesFiles.close()

test_photos = pd.read_csv(data_root+'test_photo_to_biz.csv')
test_folder = data_root+'test_photos/'
test_pimages = [str(x) for x in test_photos['photo_id'].unique()]
test_images = [os.path.join(test_folder, str(x)+'.jpg') for x in test_photos['photo_id'].unique()]  

size_testImages = len(test_images)

#start feature extraction for test images with given batch size
for i in range(0, size_testImages, batch_size): 
    images = test_images[i: min(i+batch_size, size_testImages)]
    #extract features for set of batch
    features = image_feature_extract(images, layer='fc6')
    done_num = i+features.shape[0]

    #file resizing happens here to accomodate new batch features    
    features_file= h5py.File(data_root+'testImages_fc6features.h5','r+')
    features_file['photo_id'].resize((done_num,))
    features_file['photo_id'][i: done_num] = np.array(pimages)
    features_file['feature'].resize((done_num,features.shape[1]))
    features_file['feature'][i: done_num, :] = features
    features_file.close()
    
    if done_num%10000==0 or done_num==size_testImages:
        print "Processed Test images: ", done_num
