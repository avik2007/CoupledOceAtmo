def find_index(y_array, x_array, y_point, x_point):
    import numpy as np
    distance = (y_array-y_point)**2 + (x_array-x_point)**2
    idy,idx = np.where(distance==distance.min())
    return idy[0],idx[0]

