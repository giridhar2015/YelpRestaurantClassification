caffe_root = '/home/mani/caffe/' 
data_root = '/home/mani/serviceUploads/'

import numpy as np
import sys
sys.path.append("/home/mani/caffe/python")
sysLayer = sys.argv[1]
sysPhotoPath = sys.argv[2]
print sys.argv[1]
import h5py
import caffe
import os
import pandas as pd 

def extract_features(images, layer = sysLayer):
    net = caffe.Net(caffe_root + 'models/bvlc_reference_caffenet/deploy.prototxt',
                caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel',
                caffe.TEST)
    
    # input preprocessing: 'data' is the name of the input blob == net.inputs[0]
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2,0,1))
    transformer.set_mean('data', np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1)) # mean pixel
    transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
    transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB]]

    num_images= len(images)
    net.blobs['data'].reshape(num_images,3,227,227)
    net.blobs['data'].data[...] = map(lambda x: transformer.preprocess('data',caffe.io.load_image(x)), images)
    out = net.forward()

    return net.blobs[layer].data

#f = h5py.File(data_root+'serviceUploadPic.h5','w')
#filenames = f.create_dataset('photo_id',(0,), maxshape=(None,),dtype='|S32')
#feature = f.create_dataset('feature',(0,4096), maxshape = (None,4096))
#f.close()

#photo_name = ["yelp"]
#photo_path = ["/home/mani/serviceUploads/yelp.png"]
photo_path = [sysPhotoPath]
features = extract_features(photo_path, layer=sysLayer)
    
#f= h5py.File(data_root+'serviceUploadPic.h5','r+')
#f['photo_id'].resize((1,))
#f['photo_id'][0] = np.array(photo_name)
#f['feature'].resize((1,features.shape[1]))
#f['feature'][0] = features
#f.close()
print "Image processed"

from sklearn.externals import joblib

data_root = '/home/mani/yelp/'
filename = '/home/mani/svm_classifier.pkl'
classifier = joblib.load(filename)
y_predict = classifier.predict(features)

file_path = open('/home/mani/serviceUploads/result.txt','r+')
file_path.write(str(y_predict[0]))
file_path.truncate()
file_path.close()

print y_predict[0]

