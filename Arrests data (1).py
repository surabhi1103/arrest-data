#!/usr/bin/env python
# coding: utf-8

# Importing the basic libraries

# In[60]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[61]:


pwd


# Importing the dataset

# In[62]:


Arrest=pd.read_csv(r"C:\Users\Surabhi\Desktop\Arrests.csv")
print(Arrest)


# The data set has 5226 rows and 9 columns. The variables are
# 1) released- Tells whether the given criminal has been released or not( Categorical)
# 2) colour- Tells whether the criminal is white/black (Categorical)
# 3) year- The year of arrest (Numeric variable)
# 4) age- The age of the criminal in years (numeric)
# 5) sex- The  gender of the criminal (Categorical variable)
# 6) employed- Whether the criminal was employed at the time of arrest( Categorical)
# 7) citizen- Whether the criminal is a citizen (Categorical variable )
# 8) checks- Number of police data bases in which the criminal's name has appeared( Numeric variable)
# 

# First we consider that the variable "released" is our dependent variable and the other variables are the features.
# We consider contingency tables to see the relationship of the variable "released" with all the other variables.

# In[63]:


cont_table_1=pd.crosstab(Arrest['released'],Arrest['colour'])
print(cont_table_1)


# In[64]:


from scipy.stats import chi2_contingency


# In[65]:


chi2_contingency(cont_table_1)


# Since the Chi-square test gives  p-value very close to 0, we conclude that the colour of the individual and release status are not independent variables

# In[66]:


cont_table_2=pd.crosstab(Arrest['released'],Arrest['sex'])
print(cont_table_2)


# In[67]:


chi2_contingency(cont_table_2)


# The variables "released" and "gender" are not related.

# In[68]:


cont_table_3=pd.crosstab(Arrest['released'],Arrest['employed'])
print(cont_table_3)


# In[69]:


chi2_contingency(cont_table_3)


# Since the Chi-square test gives p-value very close to 0, we conclude that the employment status of the individual and  release status are not independent variables

# In[75]:


cont_table_4=pd.crosstab(Arrest['released'],Arrest['citizen'])
print(cont_table_4)


# In[76]:


chi2_contingency(cont_table_4)


# Since the Chi-square test gives p-value very close to 0, we conclude that the citizenship of an individual and his release status are not independent variables

# Since the "sex" column is not related to the "released" column(which is our dependent variable), we ignore it and remove it from the set of features (feature selection?)

# Performing logistic regression on the dataset( Without the Sex column)
# 

# Importing the dataset

# In[77]:


X=Arrest.iloc[ :,[2,3,4,6,7,8]].values
y=Arrest.iloc[ :,1].values
y


# In[78]:


X


# In[79]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
y=le.fit_transform(y)
print(y)


# In[80]:


X[: ,0]=le.fit_transform(X[: ,0])
X[: ,3]=le.fit_transform(X[: ,3])
X[: ,4]=le.fit_transform(X[: ,4])
print(X)


# Splitting the dataset into training set and test set

# In[81]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1,random_state=0)
print(X_test)
print(y_test)


# Fitting the model

# In[82]:


from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression(random_state=0)
classifier.fit(X_train,y_train)


# In[83]:


y_pred=classifier.predict(X_train)
from sklearn.metrics import confusion_matrix,accuracy_score
cm=confusion_matrix(y_train,y_pred)
print(cm)
accuracy_score(y_train,y_pred)


# In[84]:


prob=classifier.predict_proba(X_test)
print(prob)
y_pred=classifier.predict(X_test)

print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))


# ANOVA Table of logistic Regression

# In[24]:


import statsmodels.api as sm
from scipy import stats


# In[25]:


float_y = np.vstack(y_train[:]).astype(np.float)
float_x = np.vstack(X_train[:, :]).astype(np.float)
features=["colour","year","age","employed","citizen","checks"]


# In[26]:


lr = sm.Logit(float_y,sm.add_constant(float_x)).fit(disp = False)
result = zip(['(Intercept)']+features,lr.params , lr.bse , lr.tvalues)
print('               Coefficient   Std. Error   Z Score')
print('-------------------------------------------------')
for term, coefficient, std_err, z_score in result:
    print(f'{term:>12}{coefficient:>14.3f}{std_err:>13.3f}{z_score:>10.3f}')


# In[87]:



X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1,random_state=0)
print(X_test)
print(y_test)


# Making the confusion matrix

# In[88]:


from sklearn.metrics import confusion_matrix,accuracy_score
cm=confusion_matrix(y_test,y_pred)
print(cm)
accuracy_score(y_test,y_pred)


# Alternative metrics of calculation

# In[86]:


from sklearn.metrics import classification_report
report=classification_report(y_test,y_pred,digits=3,output_dict=True)
print("Report:")
print("Accuracy={0:0.3f}".format(report["accuracy"]))
print("Precison={0:0.3f}".format(report["1"]["precision"]))
print("Sensitivity={0:0.3f}".format(report["1"]["recall"]))
print("Specificity={0:0.3f}".format(report["0"]["recall"]))


# Plotting the ROC curve

# In[89]:


from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
fpr,tpr,threshold=roc_curve(y_test,prob[ :, 1 ])
plt.plot([0,1],[0,1],'r--')
plt.plot(fpr,tpr)
plt.fill_between(fpr,tpr,color='b',alpha=0.2)
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.title("ROC Curve={}".format(roc_auc_score(y_test,prob[:,1])))
plt.show()


# Trying to decrease the false postive rate by increasing the threshold from 0.5 to 0.65

# In[90]:


y_pred_1=(prob[ :,1]>=0.65).astype(int)
print(y_pred_1)


# Confusion Matrix

# In[92]:



cm=confusion_matrix(y_test,y_pred_1)
print(cm)
accuracy_score(y_test,y_pred_1)


# In[93]:


report=classification_report(y_test,y_pred_1,digits=3,output_dict=True)
print("Report:")
print("Accuracy={0:0.3f}".format(report["accuracy"]))
print("Precison={0:0.3f}".format(report["1"]["precision"]))
print("Sensitivity={0:0.3f}".format(report["1"]["recall"]))
print("Specificity={0:0.3f}".format(report["0"]["recall"]))


# Perfoming Decision Tree Classification(Without Sex column)
# 

# In[29]:


from sklearn.tree import DecisionTreeClassifier

classifier=DecisionTreeClassifier(criterion="entropy",random_state=0)
classifier.fit(X_train,y_train)


# In[30]:


y_pred=classifier.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))


# In[31]:


cm=confusion_matrix(y_test,y_pred)
print(cm)
accuracy_score(y_test,y_pred)


# Random forest classifiaction(Without the sex column)

# In[33]:


from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier(n_estimators=100,criterion="entropy",random_state=0)
classifier.fit(X_train,y_train)


# In[34]:


y_pred=classifier.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))


# In[35]:


cm=confusion_matrix(y_test,y_pred)
print(cm)
accuracy_score(y_test,y_pred)


# How to determine the optimal number of trees in random forest classification?

# In[36]:


acc=list()
for i in range(10,1001,10):
    classifier=RandomForestClassifier(n_estimators=i,criterion="entropy",random_state=0)
    classifier.fit(X_train,y_train)
    y_pred=classifier.predict(X_test)
    acc.append(accuracy_score(y_test,y_pred))

print(acc)


# In[39]:


no_of_trees=[]
for i in range(10,1001,10):
    no_of_trees.append(i)

print(no_of_trees)

plt.plot(no_of_trees,acc)
plt.xlabel("No. of trees in the forest")
plt.ylabel("Accuracy score")
plt.title("To find optimum numer of trees for random forest algorithm")

plt.show()


# In[42]:


xmax=no_of_trees[np.argmax(acc)]
ymax=max(acc)

print(ymax)


# So optimal number of trees that maximises accuracy score is 150

# SVM

# In[43]:


from sklearn.svm import SVC


# In[44]:


x=SVC(kernel="rbf")


# In[45]:



x.fit(X_train,y_train)


# In[46]:


y_pred=x.predict(X_test)


# In[47]:


cm=confusion_matrix(y_test,y_pred)
print(cm)
accuracy_score(y_test,y_pred)


# In[48]:


classifier=SVC(kernel="poly",degree=2)


# In[49]:


classifier.fit(X_train,y_train)
y_pred=classifier.predict(X_test)


# In[50]:


cm=confusion_matrix(y_test,y_pred)
print(cm)
accuracy_score(y_test,y_pred)


# Now we do all these classification algorithms including the sex columns (Why?)

# Performing Logistic Regression in (With Sex column)

# Importing the dataset

# In[94]:


X=Arrest.iloc[ :,[2,3,4,5,6,7,8]].values
y=Arrest.iloc[ :,1].values
y


# In[95]:


X


# In[96]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
y=le.fit_transform(y)
print(y)


# In[97]:


X[: ,0]=le.fit_transform(X[: ,0])
X[: ,3]=le.fit_transform(X[: ,3])
X[: ,4]=le.fit_transform(X[: ,4])
X[: ,5]=le.fit_transform(X[: ,5])
print(X)


# Splitting the data into training set and test set

# In[98]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1,random_state=0)
print(X_test)
print(y_test)


# Fitting the model

# In[99]:


from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression(random_state=0)
classifier.fit(X_train,y_train)


# In[100]:


y_pred=classifier.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))


# In[101]:


cm=confusion_matrix(y_test,y_pred)
print(cm)
accuracy_score(y_test,y_pred)


# Decision tree classification

# In[102]:


from sklearn.tree import DecisionTreeClassifier

classifier=DecisionTreeClassifier(criterion="entropy",random_state=0)
classifier.fit(X_train,y_train)


# In[103]:


y_pred=classifier.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))


# In[104]:


cm=confusion_matrix(y_test,y_pred)
print(cm)
accuracy_score(y_test,y_pred)


# Random forest classification

# In[105]:


from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier(n_estimators=100,criterion="entropy",random_state=0)
classifier.fit(X_train,y_train)


# In[106]:


y_pred=classifier.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))


# In[107]:


cm=confusion_matrix(y_test,y_pred)
print(cm)
accuracy_score(y_test,y_pred)


# Now we consider that the variable "citizen" is our dependent variable and the other variables are the features. We consider contingency tables to see the relationship of the variable "citizen" with all the other variables.

# In[108]:


cont_table_1=pd.crosstab(Arrest['citizen'],Arrest['colour'])
print(cont_table_1)


# In[109]:


chi2_contingency(cont_table_1)


# In[110]:


cont_table_2=pd.crosstab(Arrest['citizen'],Arrest['sex'])
print(cont_table_2)


# In[111]:


chi2_contingency(cont_table_2)


# In[112]:


cont_table_3=pd.crosstab(Arrest['citizen'],Arrest['employed'])
print(cont_table_3)


# In[113]:


chi2_contingency(cont_table_3)


# In[114]:


cont_table_4=pd.crosstab(Arrest['citizen'],Arrest['released'])
print(cont_table_4)


# In[115]:


chi2_contingency(cont_table_4)


# Doing all the classification algorithms with citizenship as dependent variable
# 
# Logistic Regression

# In[124]:


X=Arrest.iloc[ :,[1,2,3,4,5,6,8]].values
y=Arrest.iloc[ :,7].values
y


# In[125]:


X


# In[126]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
y=le.fit_transform(y)
print(y)


# In[127]:


X[: ,0]=le.fit_transform(X[: ,0])
X[: ,1]=le.fit_transform(X[: ,1])
X[: ,4]=le.fit_transform(X[: ,4])
X[: ,5]=le.fit_transform(X[: ,5])
print(X)


# In[128]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1,random_state=0)
print(X_test)
print(y_test)


# In[129]:


from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression(random_state=0)
classifier.fit(X_train,y_train)


# In[130]:


y_pred=classifier.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))


# In[131]:


cm=confusion_matrix(y_test,y_pred)
print(cm)
accuracy_score(y_test,y_pred)


# Decision Tree Classifier

# In[221]:


from sklearn.tree import DecisionTreeClassifier

classifier=DecisionTreeClassifier(criterion="entropy",random_state=0)
classifier.fit(X_train,y_train)


# In[222]:


y_pred=classifier.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))


# In[223]:


cm=confusion_matrix(y_test,y_pred)
print(cm)
accuracy_score(y_test,y_pred)


# Random Forest Classification

# In[224]:


from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier(n_estimators=100,criterion="entropy",random_state=0)
classifier.fit(X_train,y_train)


# In[225]:


y_pred=classifier.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))


# In[226]:


cm=confusion_matrix(y_test,y_pred)
print(cm)
accuracy_score(y_test,y_pred)


# In[ ]:




