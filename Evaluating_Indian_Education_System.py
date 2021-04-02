#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

#https://www.analyticsvidhya.com/blog/2020/10/evaluating-the-quality-of-education-in-india-using-clustering/


# In[3]:


data = pd.read_csv('D:/Data_Science/d_c__.csv')
data.head()


# In[5]:


data.describe()


# In[7]:


data_copy = data.copy()
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler

min_max_scaler = MinMaxScaler()
model = min_max_scaler.fit(data.drop('State_UT', axis= 1))

data[['mean_dropout','enrollment_ratio','comp','electricity','water','boys_toilet','female_toilet']] = model.transform(data.drop('State_UT',axis=1))

data


# In[8]:


from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

wcss = []

for i in range(1,10):
    kmeans = KMeans(i, random_state = 3)
    kmeans.fit(data.drop('State_UT', axis = 1))
    wcss.append(kmeans.inertia_)

plt.plot(range(1,10), wcss, '-o')
plt.xlabel('No. of clusters')
plt.ylabel('WCSS')

print(wcss)


# We can use suitable metrics to decide exact number of cluster between 2 and 3. The metrics we will use are:
# 
# > Silhoutte Score- It ranges from -1 to 1. Higher the value better our clusters are. Closer to 1 means perfect clusters. 0 mean the point lies at the border of it's cluster. Negative value means that the point is classified into wrong cluster.
# 
# > Calinski-Harabasz Index denotes how the data points are spread within a cluster. Higher the score, denser is the cluster thus the cluster is better. It starts from 0 and have no upper limit.
# 
# > Davies Boulden Index measures the average similarity between cluster using the ratio of the distance between a cluster and it's closest point & the average distance between each data point of a cluster and it's cluster center.Closer the score is to 0, better our clusters are as it indicates clusters are well separated.

# In[10]:


from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score


# In[14]:


l1 = []
l2 = []
l3 = []
l4 = []
for i in range(2,6):
    kmean = KMeans(i,random_state=42)
    mod_k = kmean.fit(data.drop('State_UT',axis=1))
    pred_k1 = mod_k.predict(data.drop('State_UT',axis=1))
    silhoute_score = silhouette_score(data.drop('State_UT',axis=1),pred_k1,metric='euclidean')
    calinski_score = calinski_harabasz_score(data.drop('State_UT',axis=1),pred_k1)
    davies_score = davies_bouldin_score(data.drop('State_UT',axis=1),pred_k1)
#     print(i,'\t','Silhouette Score:' , silhoute_score,',Calinski Harbasz Score: ', calinski_score,',Davies Bouldin Score: ', davies_score)
    l1.append(silhoute_score)
    l2.append(calinski_score)
    l3.append(davies_score)
    l4.append(i)


# In[15]:


pd.DataFrame({'Cluster': l4, 'Silhouette Score' : l1,'Calinski Harbasz Score': l2,'Davies Bouldin Score': l3})


# In[18]:


from scipy.cluster.hierarchy import dendrogram
import scipy.cluster.hierarchy as sch

dendrogram=sch.dendrogram(sch.linkage(data.drop('State_UT',axis=1),method='ward'))


# In[19]:


list1=['single','complete','average','ward']


# In[24]:


from sklearn.cluster import AgglomerativeClustering
l1=[]
l2=[]
l3=[]
l4=[]
l5=[]

for j in list1:
    for i in range (2,6):
        agg_m = AgglomerativeClustering(n_clusters=i,affinity='euclidean',linkage=j)
        mod_k = agg_m.fit(data.drop('State_UT',axis=1))
        #pred_k = mod_k.predict(df.drop('State_UT',axis=1))
        silhoute_score = silhouette_score(data.drop('State_UT',axis=1),agg_m.labels_,metric='euclidean')
        calinski_score = calinski_harabasz_score(data.drop('State_UT',axis=1),agg_m.labels_)
        davies_score = davies_bouldin_score(data.drop('State_UT',axis=1),agg_m.labels_)
    #     print(i,j)
    #     print(i,'\t','Silhouette Score:' ,silhoute_score,',Calinski Harbasz Score: ',calinski_score,',Davies Bouldin Score: ',davies_score)
    #     print('---------')
    #     print('\n')
        l1.append(i)
        l2.append(j)
        l3.append(silhoute_score)
        l4.append(calinski_score)
        l5.append(davies_score)


# In[25]:


pd.DataFrame({'Cluster':l1,'Linkage':l2,'Silhoutte Score':l3,'Calinski Harabasz Index':l4,'Davies Bouldin Score':l5})


# In[32]:


kmeansf = KMeans(n_clusters = 2, random_state = 31)
kmeansf.fit(data.drop(['State_UT'], axis = 1))

pred_kk = kmeansf.predict(data.drop(['State_UT'], axis=1))
data['pred_k2'] = pred_kk


# In[33]:


data.head()


# In[40]:


# This function is useful to massage a DataFrame into a format where,
# one or more columns are identifier variables (id_vars), 
# while all other columns, considered measured variables (value_vars), are “unpivoted” to the row axis, leaving just two non-identifier columns, ‘variable’ and ‘value’.

df_kmeans = pd.melt(frame = data.drop('State_UT', axis = 1), id_vars = 'pred_k2', var_name = 'parameters', value_name = 'values')
df_kmeans.head()


# In[41]:


df_kmeans


# In[42]:


import seaborn as sns
sns.factorplot(data = df_kmeans, y = 'values', x = 'pred_k2', col = 'parameters', kind = 'box', sharey = False)


# In[45]:


agg_mod_= AgglomerativeClustering(n_clusters=2,affinity='euclidean',linkage='ward')
agg_mod_.fit(data.drop('State_UT',axis=1))
data['agg_clust2'] = agg_mod_.labels_


# In[46]:


data_aggCluster = pd.melt(frame=data.drop(['State_UT','pred_k2'],axis=1),id_vars='agg_clust2',var_name='parameters',value_name='values')
data_aggCluster.head()


# In[48]:


sns.factorplot(data = data_aggCluster, y = 'values', x = 'agg_clust2', col = 'parameters', kind = 'box', sharey = False)


# We see that both the KMeans and Agglomerative Clustering have the value range of each of the feature/category exactly the same.
# 
# Based on careful observations of the boxplots we can conclude that category has higher values of for comp, electricity, water and the toilets features. So we can says the states falling in category 0 has much better infrastructure than schools of category 1. On the otherhand, the dropout rate is almost same for both groups with 0 has higher variablity. While the enrollment ratio is good for group 1.
# 
# So we can call group 0 as Higher Infrastructure, Lesser Enrollment-Ratio and and group 1 as Less-Infrastructure, Better Enrollment-Ratio.

# In[49]:


# Checking which states fall in cluster 1 i.e, Less-Infrastructure, Better Enrollment-Ratio cluster
data[data['pred_k2']==1]


# In[50]:


# Renaming the clusters as discussed before
data['pred_k2_label']=data.pred_k2.map({0:'Higher Infrastructure, Lesser Enrollment-Ratio',1:'Less-Infrastructure, Better Enrollment-Ratio'})


# In[51]:


data


# In[ ]:




