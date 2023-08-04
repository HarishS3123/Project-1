#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import re
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


# # Madrid 2017

# In[332]:


a=pd.read_csv(r"C:\Users\user\Downloads\C10_air\madrid_2017.csv")
a


# In[333]:


a.info()


# In[334]:


b=a.fillna(value=87)
b


# In[335]:


b.columns


# In[336]:


c=b.head(10)
c


# In[337]:


d=c[['BEN', 'CO', 'EBE', 'NMHC', 'NO', 'NO_2', 'O_3', 'PM10', 'PM25',
       'SO_2', 'TCH', 'TOL', 'station']]
d


# In[338]:


x=d[['BEN', 'CO', 'EBE', 'NMHC', 'NO', 'NO_2', 'O_3', 'PM10']]
y=d['TCH']


# In[339]:


from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)


# In[340]:


from sklearn.linear_model import LinearRegression

lr=LinearRegression()
lr.fit(x_train,y_train)


# In[341]:


print(lr.intercept_)


# In[342]:


coeff=pd.DataFrame(lr.coef_,x.columns,columns=['Co-efficient'])
coeff


# In[343]:


prediction=lr.predict(x_test)
plt.scatter(y_test,prediction)


# In[344]:


print(lr.score(x_test,y_test))


# In[345]:


from sklearn.linear_model import Ridge,Lasso


# In[346]:


rr=Ridge(alpha=10)
rr.fit(x_train,y_train)


# In[347]:


rr.score(x_test,y_test)


# In[348]:


la=Lasso(alpha=10)
la.fit(x_train,y_train)


# In[349]:


la.score(x_test,y_test)


# In[350]:


a1=b.head(7000)
a1


# In[351]:


e=a1[['BEN', 'CO', 'EBE', 'NMHC', 'NO', 'NO_2', 'O_3', 'PM10', 'PM25',
       'SO_2', 'TCH', 'TOL', 'station']]
e


# In[352]:


f=e.iloc[:,0:14]
g=e.iloc[:,-1]


# In[353]:


h=StandardScaler().fit_transform(f)


# In[354]:


logr=LogisticRegression(max_iter=10000)
logr.fit(h,g)


# In[355]:


from sklearn.model_selection import train_test_split

h_train,h_test,g_train,g_test=train_test_split(h,g,test_size=0.3)


# In[356]:


i=[[10,20,30,40,50,60,11,22,33,44,55,54,21]]


# In[357]:


prediction=logr.predict(i)
print(prediction)


# In[358]:


logr.predict_proba(i)[0][0]


# In[359]:


logr.score(h_test,g_test)


# In[360]:


from sklearn.linear_model import ElasticNet
en=ElasticNet()
en.fit(x_train,y_train)


# In[361]:


prediction=en.predict(x_test)
print(en.score(x_test,y_test))


# In[362]:


from sklearn.ensemble import RandomForestClassifier

rfc=RandomForestClassifier()
rfc.fit(h_train,g_train)


# In[363]:


parameters={'max_depth':[1,2,3,4,5],
           'min_samples_leaf':[5,10,15,20,25],
           'n_estimators':[10,20,30,40,50]
           }


# In[364]:


from sklearn.model_selection import GridSearchCV

grid_search=GridSearchCV(estimator=rfc,param_grid=parameters,cv=2,scoring="accuracy")
grid_search.fit(h_train,g_train)


# In[365]:


grid_search.best_score_


# In[366]:


rfc_best=grid_search.best_estimator_


# In[370]:


from sklearn.tree import plot_tree

plt.figure(figsize=(30,10))
plot_tree(rfc_best.estimators_[2],filled=True)


# Conclusion: RandomForest score=0.9838775510204082.It has the highest accuracy.

# # Madrid 2018

# In[371]:


a=pd.read_csv(r"C:\Users\user\Downloads\C10_air\madrid_201.csv")
a


# In[372]:


a=a.head(1000)
a


# In[373]:


b=a.dropna()
b


# In[374]:


b.columns


# In[375]:


b=b[['BEN', 'CO', 'EBE', 'NMHC', 'NO', 'NO_2', 'O_3', 'PM10', 'PM25',
       'SO_2', 'TCH', 'TOL', 'station']]
b


# In[376]:


x=b.iloc[:,0:10]
y=b.iloc[:,-1]


# In[377]:


from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)


# In[378]:


lr=LinearRegression()
lr.fit(x_train,y_train)


# In[379]:


print(lr.score(x_test,y_test))


# In[380]:


rr=Ridge(alpha=10)
rr.fit(x_train,y_train)


# In[381]:


rr.score(x_test,y_test)


# In[382]:


la=Lasso(alpha=10)
la.fit(x_train,y_train)


# In[383]:


la.score(x_test,y_test)


# In[384]:


from sklearn.linear_model import ElasticNet
en=ElasticNet()
en.fit(x_train,y_train)


# In[385]:


prediction=en.predict(x_test)
print(en.score(x_test,y_test))


# In[386]:


f=StandardScaler().fit_transform(x)


# In[387]:


logr=LogisticRegression()
logr.fit(f,y)


# In[388]:


g=[[10,20,30,40,50,60,70,80,90,10]]


# In[389]:


prediction=logr.predict(g)
print(prediction)


# In[390]:


logr.predict_proba(g)[0][0]


# In[391]:


logr.score(x_test,y_test)


# In[392]:


rfc=RandomForestClassifier()
rfc.fit(x_train,y_train)


# In[393]:


parameters={'max_depth':[1,2,3,4,5],
           'min_samples_leaf':[5,10,15,20,25],
           'n_estimators':[10,20,30,40,50]
           }


# In[394]:


grid_search=GridSearchCV(estimator=rfc,param_grid=parameters,cv=2,scoring="accuracy")
grid_search.fit(x_train,y_train)


# In[398]:


grid_search.best_score_


# In[399]:


rfc_best=grid_search.best_estimator_


# In[402]:


plt.figure(figsize=(10,10))
plot_tree(rfc_best.estimators_[2],filled=True)


# Conclusion: RandomForest score=0.9310344827586208.It has the highest accuracy.

# In[ ]:




