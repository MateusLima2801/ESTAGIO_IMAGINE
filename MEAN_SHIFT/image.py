import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
import json

W = 30

def main():
    use_state = True
    json_path = "state.json"
    if not use_state:
        # 5 dimensions
        points = get_points()
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
            print("ITR: " + str(i)+"/"+str(len(points)))
        save_state(points, clusters, centroids, json_path)
    state = get_state(json_path)
    create_img_from_state(state)

def get_state(json_path):
    f = open(json_path, "r")
    return json.load(f)

def save_state(points, clusters, centroids, json_path):
    data = {
    "clusters": clusters,
    "points": points,
    "centroids": []#centroids.tolist()
    }

    with open(json_path, "w") as json_file:
        json.dump(data, json_file)

def get_points():
    img = Image.open('pixelated_img.jpeg')
    width, height = img.size
    points = []

    for y in range(height):
        for x in range(width):
            pixel_color = img.getpixel((x, y))
            points.append([pixel_color[0], pixel_color[1], pixel_color[2], x, y])
    return points

def create_img_from_state(json):
    factor = 5
    img = Image.open('pixelated_img.jpeg') 
    img = img.resize((img.width * factor, img.height * factor))
    draw = ImageDraw.Draw(img, 'RGB')

    i = 0
    for cluster in json["clusters"]:
        draw.line([(x * factor, y * factor) for r,g,b, x, y in cluster] + [(cluster[0][3], cluster[0][4])], fill=(255, 255, 255), width=1)
        i+=1

    # Save and display the image
    img.save('output_image.png')
    img.show()

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
    i = 0
    for cluster in clusters:
        label='CLUSTER ' + str(i+1)
        x=[]
        y=[]
        z=[]
        w=[]
        k=[]
        for point in cluster:
            x.append(point[0])
            y.append(point[1])
            z.append(point[2])
            w.append(point[3])
            k.append(point[4])

        c = centroids[i]
        # Create the first scatter plot in the first subplot
        plt.scatter(w, k, c=np.column_stack((c[0] / 255, c[1] / 255, c[2] / 255)), label=label)

    plt.title('IMG')
    plt.xlabel('Length')
    plt.ylabel('Width')
    plt.legend()

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