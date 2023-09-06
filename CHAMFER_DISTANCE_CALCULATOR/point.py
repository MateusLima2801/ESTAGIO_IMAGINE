from math import *

class Point:
    x: float
    y: float 
    z: float

    def __init__(self, x:str, y:str, z: str):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
    
    def print(self):
        print("x: " + str(self.x) + "; y: " + str(self.y) + "; z: " + str(self.z))
    
    def distance(self, other) -> float:
        return sqrt( pow(self.x - other.x, 2) + pow(self.y - other.y, 2) +  pow(self.z - other.z, 2))