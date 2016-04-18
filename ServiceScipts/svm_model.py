import numpy as np
import pandas as pd 
from sklearn.externals import joblib

data_root = '/home/mani/yelp/'

filename = '/home/mani/svm_classifier.pkl'
# _ = joblib.dump(y_predict_label, filename, compress=9)

y_predict_label2 = joblib.load(filename)

test_data_frame  = pd.read_csv(data_root+"test_biz_fc7features.csv")

df = pd.DataFrame(columns=['business_id','labels'])

for i in range(len(test_data_frame)):
    biz = test_data_frame.loc[i]['business']
    label = y_predict_label2[i]
    label = str(label)[1:-1].replace(",", " ")
    df.loc[i] = [str(biz), label]

with open(data_root+"submission_fc7_tmp.csv",'w') as f:
    df.to_csv(f, index=False) 
