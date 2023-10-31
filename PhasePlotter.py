# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 14:01:44 2023

@author: samyj
"""

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
import glob

    # Define bounds for x and y to be computed
w = 2 
Y, X = np.mgrid[-w:w:100j, -w:w:100j]

parmMax = 2
    # Problem parameters
aMaster = np.sort(np.unique(np.round(np.arange(-parmMax, parmMax, 0.1), 2)))
count = 0

for a in aMaster:

        # xdot equation
    U = X*(X*(1-X)-Y) 

        # ydot equation
    V = Y*(X-a) 

        #   Calculation for linewidth visualizaation
        # Extra square root to flatten for more color variation = easier to see
    speed = np.sqrt(np.sqrt(U**2 + V**2))
    lw = speed / speed.max()

        #  Streamline plot
    plt.streamplot(X, Y, U, V, density=5, linewidth=1.5, color=lw,\
                cmap='rainbow')


    plt.title('a = %a' %a)
    plt.tight_layout()
    plt.grid()
    #plt.show()
    plt.savefig('fig'+str(count)+'.png')
    plt.clf()

    count=count+1

fp_in = "fig*.png"
fp_out = "mygif.gif"

img, *imgs = [Image.open(f) for f in glob.glob(fp_in)]
img.save(fp=fp_out, format='GIF', append_images=imgs,
         save_all=True, duration=(30*count)/parmMax, loop=0)
