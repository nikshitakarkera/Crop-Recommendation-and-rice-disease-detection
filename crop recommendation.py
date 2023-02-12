

import pickle
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys

# In[ ]:
def functon():


    df = pd.read_csv('Crop_recommendation.csv')


    # In[ ]:


    #df.head()


    # **Data Cleaning**

    # In[ ]:


    # checking null values
    df.isnull().sum()


    # In[ ]:


    # checking dtype of columns
    #df.info()


    # **Data Exploration**

    # In[ ]:


    #df.describe()


    # In[ ]:


    #df.nunique()


    # In[ ]:


    #df['label'].unique()


    # In[ ]:


    #df['label'].value_counts()


    # **Data Preprocessing**

    # In[ ]:


    # In[ ]:


    label_encoder = preprocessing.LabelEncoder()


    # In[ ]:


    df['label'] = label_encoder.fit_transform(df['label'])


    # In[ ]:


    #df['label'].unique()


    # In[ ]:


    x = df.drop('label', axis=1)
    y = df['label']


    # In[ ]:


    # In[ ]:


    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.20, random_state=42)


    # In[ ]:


    # print(x_train.shape)
    # print(x_test.shape)
    # print(y_train.shape)
    # print(y_test.shape)


    # In[ ]:


    #print(x_test)


    # In[ ]:


    # In[ ]:


    model = GaussianNB()


    # In[ ]:


    model.fit(x_train, y_train)


    # In[ ]:


    y_pred = model.predict(x_test)


    # In[ ]:


    # In[ ]:


    metrics.accuracy_score(y_test, y_pred)


    # **Taking input to predict in model**

    # In[ ]:


    #df.columns


    # In[ ]:


    df1 = pd.DataFrame()


    # In[ ]:


    # n = int(input('Enter percent nitrogen content in soil'))
    # p = int(input('Enter percent phosphorus content in soil'))
    # k = int(input('Enter percent potassium content in soil'))
    # t = int(input('Enter temperature'))
    # h = int(input('Enter humidity'))
    # ph = int(input('Enter ph'))
    # r = int(input('Enter rainfall in cm'))

    n = sys.argv[1]
    p = sys.argv[2]
    k = sys.argv[3]
    t = sys.argv[4]
    h = sys.argv[5]
    ph = sys.argv[6]
    r = sys.argv[7]

    df1['N'] = [n]
    df1['P'] = [p]
    df1['K'] = [k]
    df1['temperature'] = [t]
    df1['humidity'] = [h]
    df1['ph'] = [ph]
    df1['rainfall'] = [r]

    # print(df1)
    df1_pred = model.predict(df1)


    # In[ ]:


    z=label_encoder.inverse_transform(df1_pred) 


    # In[ ]:


    # In[ ]:


    filename = 'cr'
    pickle.dump(model, open(filename, 'wb'))


    # In[ ]:


    loaded_model = pickle.load(open(filename, 'rb'))
    loaded_model.predict(df1)


    # In[ ]:
    return z[0]
x=functon()
print(x)
