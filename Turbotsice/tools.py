import numpy as np

def cart2pol(x, y):
    """
    This function compute the polar coordinates (tho,theta) of a cartesian field (x,y)

    :param: 
        (x,y) the cartesian field

    :return: 
        (rho,theta) the polar coordinates associated to (x,y)
    """
    rho = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    return(rho, theta)

def pol2cart(rho, phi):
    """
    This function compute the cartesian coordinates (x,y) from the polar coordinates (rho,theta)

    :param: 
        (rho, theta) polar coordinates

    :return: 
        (x,y) the cartesian coordinates
    """    
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)


