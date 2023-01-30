#Density Based Spatial Clustering of Applications with Noise
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score


x=[1,2,3,2,1,11,12,13,12,11,21,22,23,22,21]
y=[2,3,2,1,1,12,13,12,11,11,22,23,22,21,21]

data = list(zip(x, y))


clustering= DBSCAN(eps = 2, min_samples= 5).fit(data)

l1=clustering.fit_predict(data)

print(len(np.unique(l1)))# finding number of clusters

score1 = silhouette_score(data, l1)
score2= calinski_harabasz_score(data ,l1)

print(score1)
print(score2)

#plt.figure(figsize=(4,5))
plt.scatter(x,y,c=l1)
#plt.legend()
plt.show()



'''
clustering= DBSCAN(eps = 2, min_samples= 6).fit(data)

l1=clustering.labels_

print(len(np.unique(l1)))# finding number of clusters

score1 = silhouette_score(data, l1)
score2= metrics.calinski_harabasz_score(data ,l1)

print(score1)
print(score2)

plt.scatter(x,y,c=l1)
#plt.legend()
plt.show()'''
