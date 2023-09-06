from cloud import *
from point import *

def main():
    cloud1_name: str = "first_cloud"
    cloud2_name: str = "second_cloud"
    try:
        cloud1 = Cloud(cloud1_name)
        cloud2 = Cloud(cloud2_name)
        cloud1.print()
        cloud2.print()

        chamfer = calculateChamferDistance(cloud1, cloud2)
        print("Chamfer Distance: " + str(chamfer))

    except Exception as e:
        print(e)

def calculateChamferDistance(c1: Cloud, c2: Cloud) -> float:
    chamfer = 0
    
    for p1 in c1.points:
        nearest = c2.nearestPointOfTheCloud(p1)
        chamfer += pow(nearest.distance(p1),2 )

    for p2 in c2.points:
        nearest = c1.nearestPointOfTheCloud(p2)
        chamfer += pow(nearest.distance(p2), 2)
        
    return chamfer

if __name__ == "__main__":
    main()