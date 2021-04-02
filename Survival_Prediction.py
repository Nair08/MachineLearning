#!/usr/bin/env python
# coding: utf-8

# ## Chapter 3 - Regression Models
# 
# ### Segment 4 - Logistic Regression

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import sklearn

from pandas import Series, DataFrame
from pylab import rcParams
from sklearn import preprocessing

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict

from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score


# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')
rcParams['figure.figsize'] = 5, 4
sb.set_style('whitegrid')


# ### Logistic regression on the titanic dataset

# In[3]:


address = 'D:/LinkedIN_learning/Ex_Files_Python_Data_Science_EssT_Pt2/Exercise_Files/Data/titanic_training_data.csv'
titanic_training = pd.read_csv(address)

titanic_training.columns = ['PassengerId','Survived','Pclass','Name','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked'
]
titanic_training.head()


# In[4]:


print(titanic_training.info())


# ### Checking that your target variable is binary

# In[5]:


sb.countplot(x='Survived', data = titanic_training, palette = 'hls')


# #### Checking for missing values

# In[6]:


titanic_training.isnull().sum()


# In[7]:


titanic_training.isnull()


# In[8]:


titanic_training.describe()


# #### Taking care of missing values
# 
# So let's just go ahead and drop all the variables that aren't relevant for predicting survival. We should at least keep the following:
# 
# * Survived - This variable is reevant
# * Pclass - Does class affect survivability?
# * Sex - Yes, age is a strong factor
# * SibSp - No of siblings affectng survival rate? (Yes)
# * Parch - No of siblings affectng survival rate? (Yes)
# * Fare - 

# In[9]:


titanic_data = titanic_training.drop (['Name', 'Ticket', 'Cabin'], axis = 1)
titanic_data.head()


# #### Imputing missing values

# In[10]:


sb.boxplot(x = 'Parch', y='Age', data = titanic_data, palette = 'hls')


# In[11]:


parch_groups = titanic_data.groupby(titanic_data['Parch'])
parch_groups.mean()


# In[12]:


def age_approx(cols):
    Age = cols[0]
    Parch = cols[1]
    if pd.isnull(Age):
        if Parch == 0:
            return 32
        elif Parch == 1:
            return 24
        elif Parch == 2:
            return 17
        elif Parch == 3:
            return 22
        elif Parch == 4:
            return 45
        else:
            return 29 #Mean from descriptive
    else:
        return Age


# In[13]:


titanic_data['Age'] = titanic_data[['Age', 'Parch']].apply(age_approx, axis = 1)
titanic_data.isnull().sum()


# In[14]:


titanic_data.dropna(inplace=True)
titanic_data.reset_index(inplace=True, drop = True)
print(titanic_data.info())


# #### Converting categorical variables to a dummy indicator

# In[15]:


from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
gender_cat = titanic_data['Sex']
gender_encoder = label_encoder.fit_transform(gender_cat)
gender_encoder[0:5]


# In[16]:


titanic_data.head()


# In[17]:


## 1 is male and 0 is female
gender_df = pd.DataFrame(gender_encoder)
gender_df.head()


# In[18]:


embarked_cat = titanic_data['Embarked']
embarked_encoded = label_encoder.fit_transform(embarked_cat)
embarked_encoded[0:100]


# In[19]:


from sklearn.preprocessing import OneHotEncoder
binary_encoder = OneHotEncoder(categories = 'auto')
embarked_1hot = binary_encoder.fit_transform(embarked_encoded.reshape(-1,1))
embarked_1hot_mat = embarked_1hot.toarray()
embarked_DF = pd.DataFrame(embarked_1hot_mat, columns = ['C', 'Q', 'S'])
embarked_DF.head()


# In[20]:


titanic_data.drop(['Sex', 'Embarked'], axis = 1, inplace=True)
titanic_data.head()


# In[23]:


titanic_dmy = pd.concat([titanic_data, gender_df, embarked_DF], axis = 1, verify_integrity = True). astype(float)
titanic_dmy[0:5]


# ### Checking for inpendence before features

# link - https://stats.stackexchange.com/questions/392517/how-can-one-interpret-a-heat-map-plot 
# 
# > Each square shows the correlation between the variables on each axis. Correlation ranges from -1 to +1. Values closer to zero means there is no linear trend between the two variables. The close to 1 the correlation is the more positively correlated they are; that is as one increases so does the other and the closer to 1 the stronger this relationship is. A correlation closer to -1 is similar, but instead of both increasing one variable will decrease as the other increases. The diagonals are all 1/dark green because those squares are correlating each variable to itself (so it's a perfect correlation). For the rest the larger the number and darker the color the higher the correlation between the two variables. The plot is also symmetrical about the diagonal since the same two variables are being paired together in those squares.

# In[24]:


sb.heatmap(titanic_dmy.corr())


# In[25]:


titanic_dmy.drop(['Fare','Pclass'], axis = 1, inplace = True)
titanic_dmy.head()


# ### Checking that your dataset is sufficient

# In[28]:


titanic_dmy.info()

#We have 6 predictive variables (we are trying to predict survived)
# 1. PassengerId
# 2. Age
# 3. SibSp
# 4. Parch
# 5. 0
# 6. C,Q,S

# so we need 6 x 5 = 300 entries at least to perform logistic regression


# In[31]:


X_train, X_test, y_train, y_test = train_test_split(titanic_dmy.drop(['Survived'], axis = 1),titanic_dmy['Survived'], test_size=0.2, random_state = 200)


# In[32]:


print(X_train.shape)
print(y_train.shape)


# In[33]:


X_train[0:5]


# ### Deploying and evaluating the model

# In[34]:


LogReg =LogisticRegression(solver = 'liblinear')
LogReg.fit(X_train, y_train)


# In[35]:


y_pred = LogReg.predict(X_test)


# ## Model Evaluation

# ### Classification report without cross-validation

# In[36]:


print(classification_report(y_test,y_pred))


# ### K-fold cross validation & cross-validation

# In[38]:


y_train_pred = cross_val_predict(LogReg, X_train, y_train, cv= 5)
confusion_matrix(y_train, y_train_pred)


# In[39]:


precision_score(y_train, y_train_pred)


# ### Make a test prediction

# In[40]:


titanic_dmy[863:864]


# In[41]:


test_passenger = np.array([866, 40, 0, 0, 0, 0, 0, 1]).reshape(1,-1)


# In[42]:


print(LogReg.predict(test_passenger))

print(LogReg.predict_proba(test_passenger))


# In[43]:


X_train, X_test, y_train, y_test = train_test_split(titanic_dmy.drop(['Survived'], axis = 1),titanic_dmy['Survived'], test_size=0.4)

LogReg =LogisticRegression(solver = 'liblinear')
LogReg.fit(X_train, y_train)

y_pred = LogReg.predict(X_test)

print(classification_report(y_test,y_pred))


# In[44]:


y_train_pred = cross_val_predict(LogReg, X_train, y_train, cv= 5)
confusion_matrix(y_train, y_train_pred)


# In[45]:


precision_score(y_train, y_train_pred)


# In[46]:


print(LogReg.predict(test_passenger))

print(LogReg.predict_proba(test_passenger))


# In[ ]:




