import numpy as np

    #TODO check if pixels start at 0 or 1
def calc_angle(x):
    '''
    calibrated to a 1300 pixel wide image, returns angle in radians (-pi,pi]
    '''
    y = (float(x) - 650) * 18 / 65
    return np.radians(y)