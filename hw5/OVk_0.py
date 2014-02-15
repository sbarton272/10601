import math
def I(p):
    return math.log(1.0/p, 2)
    
def Hw(E):
    w = sum(E)
    return sum( [ (0 if (p == 0) else (p/w)*I((p/w)))  for p in E] )

def H(E):
    return sum( [ (0 if (p == 0) else p*I(p))  for p in E] )