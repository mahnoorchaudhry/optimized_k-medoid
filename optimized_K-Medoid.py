import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

def euclidean_distance(point1, point2):
    point1 = np.array(point1)                                   
    point2 = np.array(point2)
    return np.sqrt(np.sum((point1 - point2) ** 2))

def divide_data_into_sets(data, k, verbose=True):                     
    num_data_points = len(data)
    num_points_in_set = num_data_points // k
    extra_points = num_data_points % k

    sets = []
    start_index = 0
    if verbose:
        print("Data distribution into sets:")
    for i in range(k):                                                 
        end_index = start_index + num_points_in_set
        if i < extra_points:
            end_index += 1                                       
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
        min_distance = float('inf')                                    
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
            total_cost += euclidean_distance(point, medoids[i])              
    return total_cost                                                           

def k_medoids_clustering(data_tuples, k, verbose=True):          
    initial_medoids, means = initialize_medoids(data_tuples, k, verbose)
    medoids = initial_medoids.copy()             
    iteration = 0
    old_total_cost = float('inf')
    
    while True:                                   
        iteration += 1       
        if verbose:
            print(f"\nIteration {iteration}:")
            print("Current medoids:")
            for i, medoid in enumerate(medoids):
                print(f"Cluster {i + 1}: {medoid}")
        
        clusters = {i: [] for i in range(k)}              
        labels = []                                       
        for point in data_tuples:
            distances = [euclidean_distance(point, medoid) for medoid in medoids]
            cluster_index = np.argmin(distances)
            clusters[cluster_index].append(point)       
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
        
        
        new_total_cost = compute_total_cost(clusters, new_medoids)
        S = new_total_cost - old_total_cost
        print(f"Total Cost - Old: {old_total_cost}, New: {new_total_cost}")

        
        if S > 0 or all(np.array_equal(medoids[i], new_medoids[i]) for i in range(k)):
            print("Convergence reached.")
            break
        
        medoids = new_medoids
        old_total_cost = new_total_cost

    silhouette_avg = silhouette_score(data_tuples, labels)    
    return medoids, labels, silhouette_avg, iteration


data = pd.read_csv('DiamondsPrices2022.csv')
numeric_columns = data.select_dtypes(include=[np.number])
data_tuples = [tuple(x) for x in numeric_columns.to_numpy()]

data_array = np.array(data_tuples)
silhouette_avgs = []
max_k = 10  
print("Finding the optimal number of clusters:")
for k in range(2, max_k + 1):
    print(f"\nTrying k={k}:")
    _, _, silhouette_avg, _ = k_medoids_clustering(data_tuples, k, verbose=False)    
    silhouette_avgs.append(silhouette_avg)                    

print("Silhouette scores for different k values:")
for i, score in enumerate(silhouette_avgs, start=2):
    print(f"k={i}: Silhouette Score={score}")

optimal_k = np.argmax(silhouette_avgs) + 2  
print(f"\nOptimal number of clusters: {optimal_k}")

print("\nFinal Clustering with Optimal k:")
medoids, labels, silhouette_avg, iterations = k_medoids_clustering(data_tuples, optimal_k, verbose=True)

print(f"\nFinal medoids: {medoids}")
print(f"Number of iterations: {iterations}")
print(f"Silhouette Coefficient: {silhouette_avg}")

pca = PCA(n_components=2, random_state=42)                 
reduced_data = pca.fit_transform(data_array)                
colors = np.random.rand(optimal_k, 3)  
clusters = {i: [] for i in range(optimal_k)}
for point, label in zip(reduced_data, labels):   
    clusters[label].append(point)

for i, cluster in clusters.items():
    cluster_array = np.array(cluster)
    plt.scatter(cluster_array[:, 0], cluster_array[:, 1], color=colors[i], label=f'Cluster {i + 1}')
medoids_reduced = pca.transform(np.array(medoids))
plt.scatter([m[0] for m in medoids_reduced], [m[1] for m in medoids_reduced], c='red', marker='x', s=100, label='Medoids') 
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title(f'Optimized K-Medoids Clustering (Optimal K ={optimal_k}) - Silhouette Score: {silhouette_avg:.2f} - Iterations: {iterations}')
plt.legend()              
plt.show()
