import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn import metrics

x=[1,2,3,2,1,11,12,13,12,11,21,22,23,22,21]
y=[2,3,2,1,1,12,13,12,11,11,22,23,22,21,21]

plt.scatter(x,y)
plt.show()


data= list (zip(x,y))

lin= linkage(data, method='ward' , metric='euclidean')
dendrogram(lin)

plt.show()


hierarchical_cluster = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')
labels = hierarchical_cluster.fit_predict(data)

score1 = silhouette_score(data, labels)
score2= metrics.calinski_harabasz_score(data ,labels)

print("the total number of clusters are: ",hierarchical_cluster.n_clusters_)
print(score1)
print(score2)
plt.scatter(x, y, c=labels,cmap='rainbow')
plt.show()
