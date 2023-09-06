from point import *
import os

class Cloud:
    points: []
    init_file: str

    def __init__(self, file_name: str):
        self.init_file = file_name
        self.points = []
        self.readPointsFromCSV(file_name)

    def isEmpty(self) -> bool:
        return len(self.points) == 0 
    def readPointsFromCSV(self, file_name: str):
        file_path = os.getcwd() + "\\" + file_name + ".csv"
        if os.path.isfile(file_path) == False : raise Exception("File "+ file_path + " not found")
        f = open(file_path, "r")
        lines = f.readlines()
        for i in range(1,len(lines)):
            coordinate = lines[i].split(';')
            if coordinate != None and len(coordinate) >= 3:
                self.points.append(Point(coordinate[0], coordinate[1], coordinate[2]))

    def print(self):
        if self.isEmpty():
            print("Cloud at "+self.init_file +" is empty.")
        else:
            print("Cloud at " + self.init_file + ":")
            for point in self.points:
                point.print()
            print()

    def nearestPointOfTheCloud(self, other: Point)-> Point:
        if self.isEmpty(): return None
        else:
            nearest = 0
            minDistance = self.points[0].distance(other)
            for i in range(1, len(self.points)):
                if other.distance(self.points[i]) < minDistance:
                    nearest = i
                    minDistance = other.distance(self.points[i])
            return self.points[nearest]