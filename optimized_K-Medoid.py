import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

def euclidean_distance(point1, point2):
    point1 = np.array(point1)                                   #Special type of array that allows for efficient mathematical operations.
    point2 = np.array(point2)
    return np.sqrt(np.sum((point1 - point2) ** 2))

def divide_data_into_sets(data, k, verbose=True):                     #optional verbose flag (to print details if set to True).
    num_data_points = len(data)
    num_points_in_set = num_data_points // k
    extra_points = num_data_points % k

    sets = []
    start_index = 0
    if verbose:
        print("Data distribution into sets:")
    for i in range(k):                                                 #This loop runs k times, once for each set to be created.
        end_index = start_index + num_points_in_set
        if i < extra_points:
            end_index += 1                                       #the first few sets, one extra point is added to make the distribution evenly
        dataset = data[start_index:end_index]
        sets.append(dataset)
        if verbose:
            print(f"Set {i + 1}: {len(dataset)} points")
        start_index = end_index
    return sets

def initialize_medoids(data, k, verbose=True):
    sets = divide_data_into_sets(data, k, verbose)
    medoids = []
    means = []

    means = [np.mean(dataset, axis=0) for dataset in sets]

    if verbose:
        print("\nMedoids selection:")
    for i, dataset in enumerate(sets):
        mean = means[i]
        min_distance = float('inf')                                    #They will be used to find the point in the current set closest to the mean.
        nearest_point = None
        for point in dataset:
            distance = euclidean_distance(point, mean)
            if distance < min_distance:
                min_distance = distance
                nearest_point = point
        medoids.append(nearest_point)
        if verbose:
            print(f"Medoid for Set {i + 1}: {nearest_point}")
    
    if verbose:
        print("\nMeans of each set:")
        for i, mean in enumerate(means):
            print(f"Mean for Set {i + 1}: {mean}")

    return medoids, means

def compute_total_cost(clusters, medoids):
    total_cost = 0
    for i, cluster in clusters.items():
        for point in cluster:
            total_cost += euclidean_distance(point, medoids[i])               #Adds the distance between the current point and the medoid of the cluster
    return total_cost                                                           #Returns the final total cost after summing up all distances.

def k_medoids_clustering(data_tuples, k, verbose=True):          #data_tuples (data points), k(number of clusters),  verbose (optional flag print details).
    initial_medoids, means = initialize_medoids(data_tuples, k, verbose)#Calls the initialize_medoids function to get the starting medoids and means of each set.
    medoids = initial_medoids.copy()              # This copy is used to track and update the medoids during the clustering process without modifying the original
    iteration = 0
    old_total_cost = float('inf')
    
    while True:                                   #Starts an infinite loop to repeatedly update medoids until convergence.
        iteration += 1       #Dictionaries are key-value pairs and used to manage and organize clusters, while tuples are ordered collections of elements.
        if verbose:
            print(f"\nIteration {iteration}:")
            print("Current medoids:")
            for i, medoid in enumerate(medoids):
                print(f"Cluster {i + 1}: {medoid}")
        
        clusters = {i: [] for i in range(k)}              #Initializes an empty list for each cluster in a dictionary where i is the cluster index.
        labels = []                                       #Initializes an empty list to keep track of which cluster each data point belongs to.
        for point in data_tuples:
            distances = [euclidean_distance(point, medoid) for medoid in medoids]
            cluster_index = np.argmin(distances)
            clusters[cluster_index].append(point)       #Adds the point to the appropriate cluster based on the closest medoid.
            labels.append(cluster_index)

        new_medoids = []
        for cluster in clusters.values():
            mean_point = np.mean(cluster, axis=0)
            min_distance = float('inf')
            nearest_point = None
            for point in cluster:
                distance = euclidean_distance(point, mean_point)
                if distance < min_distance:
                    min_distance = distance
                    nearest_point = point
            new_medoids.append(nearest_point)
        
        # Calculate total cost of new medoids
        new_total_cost = compute_total_cost(clusters, new_medoids)
        S = new_total_cost - old_total_cost
        print(f"Total Cost - Old: {old_total_cost}, New: {new_total_cost}")

        # Check for convergence
        if S > 0 or all(np.array_equal(medoids[i], new_medoids[i]) for i in range(k)):
            print("Convergence reached.")
            break
        
        medoids = new_medoids
        old_total_cost = new_total_cost

    silhouette_avg = silhouette_score(data_tuples, labels)    #data_tuples: The data points being clustered & labels: The cluster assignments for each data point,
    return medoids, labels, silhouette_avg, iteration

# Read the data
data = pd.read_csv('DiamondsPrices2022.csv')
numeric_columns = data.select_dtypes(include=[np.number])
data_tuples = [tuple(x) for x in numeric_columns.to_numpy()]

# Convert data_tuples back to a NumPy array for PCA
data_array = np.array(data_tuples)

# Determine the optimal number of clusters
silhouette_avgs = []
max_k = 10  # Maximum number of clusters to try

print("Finding the optimal number of clusters:")
for k in range(2, max_k + 1):
    print(f"\nTrying k={k}:")
    _, _, silhouette_avg, _ = k_medoids_clustering(data_tuples, k, verbose=False)    #Calls the k_medoids_clustering function with the current number of clusters 
    silhouette_avgs.append(silhouette_avg)                     #Adds the silhouette score for the current number of clusters to the silhouette_avgs list.

# Print silhouette scores for verification
print("Silhouette scores for different k values:")
for i, score in enumerate(silhouette_avgs, start=2):
    print(f"k={i}: Silhouette Score={score}")

optimal_k = np.argmax(silhouette_avgs) + 2  # Adding 2 because range starts at 2
print(f"\nOptimal number of clusters: {optimal_k}")

# Perform clustering with the optimal number of clusters
print("\nFinal Clustering with Optimal k:")
medoids, labels, silhouette_avg, iterations = k_medoids_clustering(data_tuples, optimal_k, verbose=True)

print(f"\nFinal medoids: {medoids}")
print(f"Number of iterations: {iterations}")
print(f"Silhouette Coefficient: {silhouette_avg}")

# Reduce dimensions for plotting using PCA
pca = PCA(n_components=2, random_state=42)                 #n_components=2), which means the data will be reduced to two dimensions.
reduced_data = pca.fit_transform(data_array)                #random_state=42: Sets a random seed This makes sure that the results of the PCA are the same every time the code is run.

# Plot clusters with random colors
colors = np.random.rand(optimal_k, 3)  # Generate random RGB colors
clusters = {i: [] for i in range(optimal_k)}
for point, label in zip(reduced_data, labels):   #zip is used iterates through two lists simultaneously: reduced_data (which contains the 2D points after PCA) and labels (which contains the cluster labels for each point)
    clusters[label].append(point)

for i, cluster in clusters.items():
    cluster_array = np.array(cluster)
    plt.scatter(cluster_array[:, 0], cluster_array[:, 1], color=colors[i], label=f'Cluster {i + 1}')
medoids_reduced = pca.transform(np.array(medoids))
plt.scatter([m[0] for m in medoids_reduced], [m[1] for m in medoids_reduced], c='red', marker='x', s=100, label='Medoids') # [m[0] for m in medoids_reduced]: Extracts the x-coordinates of the medoids.
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title(f'Optimized K-Medoids Clustering (Optimal K ={optimal_k}) - Silhouette Score: {silhouette_avg:.2f} - Iterations: {iterations}')
plt.legend()              #improving readability and helping viewers interpret the different elements of the plot.
plt.show()
