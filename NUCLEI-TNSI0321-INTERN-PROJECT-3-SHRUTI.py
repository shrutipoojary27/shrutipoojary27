#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option("display.max_columns", 80) #https://www.kaggle.com/faressayah/ibm-hr-analytics-employee-attrition-performance


# In[2]:


data = pd.read_csv('D:/Data Science Course 7-12/Project 3/WA_Fn-UseC_-HR-Employee-Attrition.csv')


# In[3]:


data.head()


# In[4]:


data.info()


# In[5]:


data = data.drop(columns=['StandardHours','EmployeeCount','Over18','EmployeeNumber'])


# In[6]:


data.describe()


# In[7]:


sns.set_style('ticks')
sns.countplot(x='Attrition', palette = 'deep', data = data) #https://www.kaggle.com/winternguyen/titanic-beginners


# In[8]:


sns.set_style('ticks')
sns.countplot(x='Gender', palette = 'deep', data = data) #https://www.kaggle.com/winternguyen/titanic-beginners


# In[9]:


sns.set_style('ticks')
sns.countplot(x='Attrition', hue='Gender', palette = 'deep', data = data) #https://www.kaggle.com/winternguyen/titanic-beginners
plt.title('-------Attrition  Vs Gender------')


# In[10]:


sns.set_style('ticks')
sns.countplot(x='Department', data = data) #https://www.kaggle.com/winternguyen/titanic-beginners


# In[11]:


sns.set_style('ticks')
sns.countplot(x='Department', hue='Gender', data = data) #https://www.kaggle.com/winternguyen/titanic-beginners
plt.title('-------Department Vs Gender------')


# In[12]:


sns.set_style('ticks')
sns.countplot(x='Attrition', hue='Department',  data = data) #https://www.kaggle.com/winternguyen/titanic-beginners
plt.title('-------Attrition  Vs Department------')


# In[13]:


sns.set_style('ticks')
sns.countplot(x='Education', palette = 'deep', data = data)


# In[14]:


plt.figure(figsize =(10, 6))
sns.boxplot(y ='MonthlyIncome', x ='Attrition', data = data)


# In[15]:


object_col = []
for column in data.columns:
    if data[column].dtype == object and len(data[column].unique()) <= 30:
        object_col.append(column)
        print(f"{column} : {data[column].unique()}")
        print(data[column].value_counts())
        print("====================================")
object_col.remove('Attrition')


# In[16]:


from sklearn.preprocessing import LabelEncoder, StandardScaler

label = LabelEncoder()
data["Attrition"] = label.fit_transform(data.Attrition)


# In[17]:


disc_col = []
for column in data.columns:
    if data[column].dtypes != object and data[column].nunique() < 30:
        print(f"{column} : Minimum: {data[column].min()}, Maximum: {data[column].max()}")
        disc_col.append(column)
        print("====================================")
disc_col.remove('Attrition')


# In[18]:


cont_col = []
for column in data.columns:
    if data[column].dtypes != object and data[column].nunique() > 30:
        print(f"{column} : Minimum: {data[column].min()}, Maximum: {data[column].max()}")
        cont_col.append(column)
        print("====================================")


# In[19]:


plt.figure(figsize=(20, 20))

for i, column in enumerate(disc_col, 1):
    plt.subplot(4, 4, i)
    data[data["Attrition"] == 0][column].hist(bins=25, color='green', label='Attrition = NO', alpha=0.8)
    data[data["Attrition"] == 1][column].hist(bins=25, color='red', label='Attrition = YES', alpha=0.8)
    plt.legend()
    plt.xlabel(column)


# In[20]:


plt.figure(figsize=(20, 10))

for i, column in enumerate(cont_col, 1):
    plt.subplot(2, 4, i)
    data[data["Attrition"] == 0][column].hist(bins=35, color='green', label='Attrition = NO', alpha=0.6)
    data[data["Attrition"] == 1][column].hist(bins=35, color='red', label='Attrition = YES', alpha=0.6)
    plt.legend()
    plt.xlabel(column)


# In[21]:


sns.set(font_scale=2.0)
plt.figure(figsize=(30, 30))
sns.heatmap(data.corr(), annot=True, cmap="RdBu", annot_kws={"size":18})


# In[22]:


from sklearn import preprocessing
le = preprocessing.LabelEncoder()


# In[23]:


new_data = data.apply(le.fit_transform)


# In[24]:


y=new_data.Attrition
X=new_data.drop(['Attrition'], axis=1)


# In[25]:


#Splitting Data in Train and Test Set
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,accuracy_score,roc_auc_score,precision_recall_curve,confusion_matrix,precision_score,confusion_matrix
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=100)


# In[26]:


base=np.zeros(1470)
print(accuracy_score(base,data.Attrition)) #https://www.kaggle.com/anvesh1216/ibm-hr-attrition


# In[27]:


#Method that applies model on the data and Predict the attrition
def model(mod,model_name,x_tr,y_tr,x_tes,y_te):
    mod.fit(x_tr,y_tr)
    pred_dt=mod.predict(x_tes)
    print("     ",model_name,"      ")
    print("Accuracy ",accuracy_score(pred_dt,y_te))
    print("ROC_AUC  ",roc_auc_score(pred_dt,y_te))
    cm=confusion_matrix(pred_dt,y_te)
    print("Confusion Matrix  \n",cm)
    print("                    Classification Report \n",classification_report(pred_dt,y_te))


# In[28]:


from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
model(lr,"Logistic Regression",X_train,y_train,X_test,y_test)


# In[29]:


from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(min_samples_leaf=20, max_depth=4)
model(dt,"Decision Tree",X_train,y_train,X_test,y_test)


# In[30]:


from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier(n_estimators=10,max_depth=4)
model(rf,"Random Forest",X_train,y_train,X_test,y_test)


# In[31]:


pip install imblearn


# In[32]:


from imblearn.over_sampling import SMOTE


# In[33]:


#Performing OverSample using SMOTE(Synthetic Minority Over Sampling Technique) since our data was imbalanced and it was showing low precision and recall scores
smote=SMOTE()


# In[35]:


X_sm, y_sm=smote.fit_resample(X,y)


# In[36]:


X_train_sm,X_test_sm,y_train_sm,y_test_sm=train_test_split(X_sm,y_sm,test_size=0.2,random_state=100)


# In[37]:


lr_sm=LogisticRegression()
model(lr_sm,"Logistic Regression",X_train_sm,y_train_sm,X_test_sm,y_test_sm)


# In[38]:


dt_sm=DecisionTreeClassifier(min_samples_leaf=20, max_depth=4)
model(dt_sm,"Decision Tree",X_train_sm,y_train_sm,X_test_sm,y_test_sm)


# In[39]:


rf_sm=RandomForestClassifier(n_estimators=10,max_depth=4)
model(rf_sm,"Random Forest",X_train_sm,y_train_sm,X_test_sm,y_test_sm)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




