import pandas as pd



# The kmeans algorithm is implemented in the scikits-learn library
from sklearn.cluster import KMeans


print "Dataset: P.csv"

P = pd.read_csv("P.csv")


for k in range (1, 11):

	# Create a kmeans model on our data, using k clusters.  random_state helps ensure that the algorithm returns the same results each time.
	kmeans_model = KMeans(n_clusters=k, random_state=1).fit(P.iloc[:, :])
	
	# These are our fitted labels for clusters -- the first cluster has label 0, and the second has label 1.
	labels = kmeans_model.labels_

	# Sum of distances of samples to their closest cluster center
	interia = kmeans_model.inertia_
	print "k:",k, " cost:", interia

print ""
print "Dataset: Q.csv"

Q = pd.read_csv("Q.csv")

for k in range (1, 11):

        # Create a kmeans model on our data, using k clusters.  random_state helps ensure that the algorithm returns the same results each time.
        kmeans_model = KMeans(n_clusters=k, random_state=1).fit(Q.iloc[:, :])

        # These are our fitted labels for clusters -- the first cluster has label 0, and the second has label 1.
        labels = kmeans_model.labels_
        
	# Sum of distances of samples to their closest cluster center
	interia = kmeans_model.inertia_
        print "k:",k, " cost:", interia




