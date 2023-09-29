import matplotlib.pyplot as plt
import numpy as np
import random
from time import sleep

points = [[2,10], [2,5],[8,4],[5,8],[7,5],[6,4],[1,2],[4,9], [6,8]]
n_clusters = 4
LIM = 0
MAX_ITR = 100
colors = ['red', 'green', 'blue', 'yellow', 'purple']

def main():
    # clusters 0, 1, ...
    centroids = init_centroids(points, n_clusters)
    p_assignments = np.zeros(len(points))
    iterate_clusters(points, centroids, p_assignments)
    print_clusters(points, centroids, p_assignments)
    

def iterate_clusters(points:list, centroids: list, p_assignments:list):
    diffs = len(points)
    i = 0
    while(diffs > LIM and i<MAX_ITR):
        diffs = assign_points_to_clusters(centroids, points, p_assignments)
        print("ITER "+ str(i))
        centroids = calculate_centroids(points, p_assignments, n_clusters)
        i+=1

def print_clusters(points:list, centroids: list, p_assignments:list ):
    cluster_coords = []
    for c in centroids:
        cluster_coords.append([])
    for i in range(len(points)):
        cluster_coords[int(p_assignments[i])].append(points[i])
    
    print(cluster_coords)
    # Create a scatter plot for the first half with dynamically chosen colors
    for i in range(len(centroids)):
        x = []
        y = []
        for j in range(len(cluster_coords[i])):
            x.append(cluster_coords[i][j][0])
            y.append(cluster_coords[i][j][1])
        plt.scatter(x,y, c=colors[i], label='CLUSTER ' + str(i))
        
    # Add labels and legend
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend()

    # Show the plot
    plt.show()
    


def calculate_centroids(points, p_assignments, n_clusters):
    means = np.zeros((n_clusters,len(points[0])))
    cardinalities = np.zeros(n_clusters)
    for i in range(len(points)):
        means[int(p_assignments[i])] += points[i]
        cardinalities[int(p_assignments[i])] += 1
    for i in range(len(means)):
        if cardinalities[i]!=0:
            means[i] /= cardinalities[i]
    return means
    
# return changes in assignments
def assign_points_to_clusters(centroids, points, p_assignments) -> int:
    distances = np.zeros(len(centroids))
    diffs = 0
    nearest_cluster_idx: int
    for p in range(len(points)):
        for i in range(len(centroids)):
            distances[i] = calculate_distance_between(points[p], centroids[i])
        nearest_cluster_idx = 0
        for j in range(1, len(centroids)):
            if distances[j] < distances[nearest_cluster_idx]:
                nearest_cluster_idx = j
        if(p_assignments[p] != nearest_cluster_idx):
            diffs += 1
            p_assignments[p] = int(nearest_cluster_idx) 
    return diffs

def init_centroids(points, n)->list:
    centroids = []
    for i in range(n):
        centroids.append([0,0])
    key: bool
    for i in range(n):
        r = 0
        key = True
        while(key):
            key = False
            r = random.randint(0, len(points)-1)
            for j in range(i):
                key = True
                for k in range(len(points[0])):
                    key = key and centroids[j][k] == points[r][k]
        centroids[i] = points[r]
    return centroids

def calculate_distance_between(p1, p2)->float:
    if(len(p1)!=len(p2)): raise Exception()
    else:
        d = 0.0
        for i in range(len(p1)):
            d+= (p1[i]-p2[i])**2
        return d
    
if __name__ == "__main__":
    main()