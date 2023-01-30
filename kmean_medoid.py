import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_score
from sklearn import metrics
from yellowbrick.cluster import KElbowVisualizer

from kneed import KneeLocator



x=[1,2,3,2,1,11,12,13,12,11,21,22,23,22,21]
y=[2,3,2,1,1,12,13,12,11,11,22,23,22,21,21]

data = list(zip(x, y))

kmeans=KMeans(init="random",n_clusters=2,n_init=10,max_iter=300,random_state=42)
kmeans.fit(data)
print(kmeans.inertia_)#lowest SSE VALUE(sum of the squared error)

print(kmeans.cluster_centers_)#final location of the centroid

print(kmeans.n_iter_)  #number of iterations required to converge

visualiser=KElbowVisualizer(kmeans, k=(2,7) , metric= 'calinski_harabasz',timing=True)
visualiser.fit(data)
visualiser.show()
plt.scatter(x, y, c=kmeans.labels_)
plt.show()
#----------------------------------------------------------------------
#Elbow method
inertias_mean=[]
inertias_medoid=[]

# A list holds the silhouette coefficients for each k
silhouette_coefficients_mean = []
silhouette_coefficients_medoid = []

calinski_harabasz_mean=[]
calinski_harabasz_medoid=[]

for i in range(3,12,2):

    #----------------------------k means--------------------------------------------
    kmeans=KMeans(n_clusters=i)
    kmeans.fit(data)
    
    inertias_mean.append(kmeans.inertia_)
    
    score = silhouette_score(data, kmeans.labels_)#The silhouette score() function needs a minimum of two clusters, or it will raise an exception.
    silhouette_coefficients_mean.append(score)
    
    cal_score=metrics.calinski_harabasz_score(data, kmeans.labels_)
    calinski_harabasz_mean.append(cal_score)
    
    
    #-------------------k medoids---------------------------------------------
    kmedoids = KMedoids(n_clusters= i, random_state= 0)
    kmedoids.fit(data)
    inertias_medoid.append(kmedoids.inertia_)
    score = silhouette_score(data, kmedoids.labels_)
    silhouette_coefficients_medoid.append(score)
    
    cal_score=metrics.calinski_harabasz_score(data, kmedoids.labels_)
    calinski_harabasz_medoid.append(cal_score)
    
    
    
#print(kmeans.elbow_value_)  
#graph to plot SSE vs number of clusters
#plt.plot(kmeans, k =(2,12) , metric= ' calinski_harabasz',timing=True)
#visualiser.fit(data)
#visualiser.show()
visualiser=KElbowVisualizer(kmeans, k=(3,12) , metric= 'calinski_harabasz',timing=True)
visualiser.fit(data)
visualiser.show()
plt.plot(range(3,12,2),inertias_mean,marker='o',label="SSE for kMEAN")
plt.plot(range(3,12,2),inertias_medoid,marker='o', label= "SSE for Kmedoid")
plt.legend()
plt.title('Elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')#SSE
plt.show()

#finding the elbow using kneed locator
kl=KneeLocator(range(3,12,2), inertias_mean, curve="convex", direction="decreasing")
print("Number of cluster =",kl.elbow)

#graph to plot silhouette coefficient vs no. of clusters
plt.plot(range(3,12,2), silhouette_coefficients_mean , marker='o',label="silhouette score for kMEAN")
plt.plot(range(3,12,2), silhouette_coefficients_medoid , marker='o',label="silhouette score for kMedoid")
plt.legend()
plt.xlabel("No. of clusters")
plt.ylabel("silhouette_coefficients")
plt.show()


#graph to plot calinski_harabasz vs no. of clusters
plt.plot(range(3,12,2), calinski_harabasz_mean , marker='o' , label= "calinski harabasz score for kmean")
plt.plot(range(3,12,2), calinski_harabasz_medoid , marker='o' , label= "calinski harabasz score for kmedoid")
plt.legend()
plt.xlabel("No. of clusters")
plt.ylabel("calinski_harabasz")
plt.show()

#-------------------------------------------------------------------------
#Kneed ----->to identify the elbow point programmatically
#kl=KneeLocator(range(3,8,2) , inertias, curve="convex",direction= "decreasing")

#print(kl.elbow)
#---------------------------------------------------------------------------

#from kneed import KneeLocator
#from sklearn.datasets import make_blobs
#from sklearn.preprocessing import StandardScaler
	
