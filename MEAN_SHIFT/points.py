import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris

W = 1.6
colors = ['red', 'green', 'blue', 'yellow', 'purple', 'black', 'pink', 'orange', 'cyan', 'magenta']

def main():
    # 4 dimensions
    points = load_iris().data
    centroids = []
    # key: centroid idx, value: idx of point in cluster with key centroid
    clusters = []
    for i in range(len(points)):
        c = calculate_centroid(points, points[i])
        idx = get_idx(centroids, c)
        if idx < 0:
            clusters.append([])
            centroids.append(c)    
        clusters[idx].append(points[i])
    print_clusters(points, clusters, centroids)

def get_idx(points, point):
    for i in range(len(points)):
        if distance_between(points[i], point) == 0:
            return i
    return -1
def contains(points, point):
    for p in points:
        if distance_between(p,point) == 0: return True
    return False

def print_clusters(points:list, clusters: list, centroids: list):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    i = 0
    for cluster in clusters:
        label='CLUSTER ' + str(i+1)
        x=[]
        y=[]
        z=[]
        w=[]
        for point in cluster:
            x.append(point[0])
            y.append(point[1])
            z.append(point[2])
            w.append(point[3])

        # Create the first scatter plot in the first subplot
        ax1.scatter(x, y, c=colors[i], label=label)

        # Create the second scatter plot in the second subplot
        ax2.scatter(z, w, c=colors[i], label=label)
        i+=1
        
        

    ax1.set_title('Sepal')
    ax1.set_xlabel('Length')
    ax1.set_ylabel('Width')
    ax1.legend()

    ax2.set_title('Petal')
    ax2.set_xlabel('Length')
    ax2.set_ylabel('Width')
    ax2.legend()
    # Adjust layout
    plt.tight_layout()

    # Show the plots
    print(centroids)
    plt.show()

def calculate_centroid(points, point):
    near_points = []
    
    for p in points:
        if distance_between(p, point) <= W:
            near_points.append(p)

    local_centroid = mean(near_points) 
    if distance_between(local_centroid, point) == 0:
        return local_centroid
    else: return calculate_centroid(points, local_centroid)

def mean(points):
    m = np.zeros(len(points[0]))
    for p in points:
        m += p
    m /= len(points)
    return m

def distance_between(p1, p2)->float:
    if(len(p1)!=len(p2)): raise Exception()
    else:
        d = 0.0
        for i in range(len(p1)):
            d+= (p1[i]-p2[i])**2
        return d ** 0.5
    
if __name__  == "__main__":
    main()