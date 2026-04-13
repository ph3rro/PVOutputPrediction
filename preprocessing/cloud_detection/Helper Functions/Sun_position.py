# -*- coding: utf-8 -*-
"""
Created on Aug 6 16:41:00 2019
Revised version on Feb 12 17:26:00 2020
@author: ynie
"""

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from Relative_op_func import *

def sun_position(time, image = None, plotting = False):
    """
    Take inputs of sky image and its assoicated time and identify the position of sun in the sky image
    """
    
    azimuth, zenith = Solar_angle(time)
    # delta is the different between geological north and sky image north
    delta = 14.036 
    rho = zenith/90*29
    theta = azimuth-delta+90
    
    ## circle center of the sky image (29,30)
    origin_x = 29
    origin_y = 30
    sun_center_x = round(origin_x-rho*sin(radians(theta)))
    sun_center_y = round(origin_y+rho*cos(radians(theta)))
    
    if plotting == True:
        sun_position_matrix = np.zeros((64,64,3),dtype=np.uint8)
        for i in range(64):
            for j in range(64):
                if (i-sun_center_x)**2+(j-sun_center_y)**2<=1**2:
                    sun_position_matrix[:,:,0][i,j]=255
                
        plt.imshow(image[:,:,::-1], interpolation='none')
        plt.imshow(sun_position_matrix, interpolation='none', alpha=0.2)
        plt.title(time)
        plt.text(sun_center_y-5,sun_center_x-5,"({0},{1})".format(sun_center_y,sun_center_x),fontsize=12)
        plt.show()
    
    return sun_center_x, sun_center_y