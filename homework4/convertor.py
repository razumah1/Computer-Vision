import numpy as np

def cartesian_to_homogeneous(cartesian_array):
    """
    Convert array from Cartesian -> Homogeneous (2 dimensions -> 3 dimensions)
    
    Parameters:
        cartesian_array (numpy.array): Input array in Cartesian coordinates
        
    Returns:
        numpy.array: Array converted to Homogeneous coordinates
    """
    if cartesian_array.ndim == 1:
        return np.append(cartesian_array, 1)
    else:
        ones_row = np.ones((1, cartesian_array.shape[1]))
        return np.vstack((cartesian_array, ones_row))
