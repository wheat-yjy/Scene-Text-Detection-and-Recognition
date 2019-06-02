import numpy as np
def coml():
    import math
    a = np.array([1,0])
    b = np.array([1,0])
    return math.acos(np.dot(a,b) / (np.linalg.norm(a)*np.linalg.norm(b)))

if __name__ == '__main__':
    print(coml())