# optimized_k-medoid
Optimized K-Medoid: A Comprehensive Approach  to Medoid Discovery and Finding Optimal K


Overview
This repository contains the implementation of the Optimized K-Medoid algorithm, created as part of my research.
In this code, we worked on improving how medoids are discovered. The algorithm finds medoids in fewer iterations, gives faster convergence, and shows high accuracy and scalability. Accuracy was checked using the Silhouette Score, which measures how well the data points fit within clusters.
In simple terms, this version of K-Medoid is quicker and more reliable than the basic approach.


Real-world use
This algorithm can be applied in many real-world areas, such as:
Customer segmentation – grouping customers with similar behavior.
Health data analysis – finding patient groups with similar conditions.
Market research – clustering products or user preferences.
Network security – detecting unusual traffic patterns.



License: MIT License
Author: Mahnoor Chaudhry



Dependencies
Required packages: numpy, pandas, scikit-learn, matplotlib
Python version: =3.12.2
Hardware: Intel i5, 8GB RAM
numpy = Version: 1.26.4
pandas = Version: 2.2.1
matplotlib = Version: 3.8.3
scikit-learn = Version: 1.4.1.post1
GPU: Not used
CUDA: Not applicable


Performance
The optimized K-Medoid implementation achieves:
Faster convergence than K-Mean, K-Medoid, PAM and CLARA, CLARANS on all datasets(small,medium and large).
Better clustering quality measured by Silhouette Score.
Robustness in noisy datasets compared to traditional K-Medoids.



Proposals, Questions, Bugs
If you have questions, feature requests, or found a bug:
contact me via GitHub