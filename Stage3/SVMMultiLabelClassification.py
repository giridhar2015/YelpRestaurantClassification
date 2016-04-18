import numpy 
import pandas
from sklearn import svm, datasets
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score
import time

#Input location for data set
source = '/media/giridhar/289454CE9454A05C/Giridhar1/Ubuntu/spring2016/EEL6935-Bigdata/FinalProject/DataSet/'

#Read all the image ids from train_photo_to_biz_ids.csv
image_id = pandas.read_csv(source+'train_photo_to_biz_ids.csv', index_col='photo_id')

#Read computed business labels for train data
train_business_features = pandas.read_csv(source+"train_business_features.csv")

#Read computed business labels for test data
test_business_features  = pandas.read_csv(source+"test_business_features.csv")

#Extract train data labels
train_labels = train_business_features['label'].values

#Extract feature vectors of train data
train_features = train_business_features['feature vector'].values

#Extract feature vectors of test data
test_features = test_business_features['feature vector'].values

#Function to access label string as label array
def getLabels(labels):
    labels = labels[1:-1]
    labels = labels.split(',')
    return [int(x) for x in labels if len(x)>0]

#Function to access feature vector string as feature vector array
def getFeatureVectors(features):
    features = features[1:-1]
    features = features.split(',')
    return [float(x) for x in features]

#Get labels of all the train businesses as arrays
train_labels = numpy.array([getLabels(y) for y in train_business_features['label']])

#Get feature vectors of all the train businesses as arrays
train_features = numpy.array([getFeatureVectors(x) for x in train_business_features['feature vector']])

#Get feature vectors of all the test businesses as arrays
test_features = numpy.array([getFeatureVectors(x) for x in test_business_features['feature vector']])

#Convert train labels into binary format to avail for multi-classification
mul_bin = MultiLabelBinarizer()
train_labels_bin= mul_bin.fit_transform(train_labels)  

#Split the train data set to predict f1 score on 20% of the train data
random_state = numpy.random.RandomState(0)
train_feat, test_feat, train_lab, test_lab = train_test_split(train_features, train_labels_bin, test_size=.2,random_state=random_state)

#Initialize the linear svm classifier
classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True))

#Train svm using 80% of train data 
classifier.fit(train_feat, train_lab)

#Predict labels of 20% of train data
predict_test_lab = classifier.predict(test_feat)

#Compute f1 score for the splitted train data
f1Score = f1_score(test_lab, predict_test_lab, average='micro') 

#Compute f1 score for each feature for the splitted train data
f1Score_all = f1_score(test_lab, predict_test_lab, average=None)
  
#Train svm using entire train data set
classifier.fit(train_features, train_labels_bin)

#Predict labels for test businesses
predict_test_labels = classifier.predict(test_features)

#Get string format of predicted test business labels
predicted_test_labels = mul_bin.inverse_transform(predict_test_labels) 

#Get data frame model for final output writer
dataFrame = pandas.DataFrame(columns=['business_id','labels'])

#Write the resulted business labels into output file
for i in range(len(test_business_features)):
    business = test_business_features.loc[i]['business']
    label = predicted_test_labels[i]
    label = str(label)[1:-1].replace(",", " ")
    dataFrame.loc[i] = [str(business), label]

#Write the results into submission file
with open(source+"submission_fc7.csv",'w') as file:
    dataFrame.to_csv(file, index=False)   

