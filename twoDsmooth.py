import numpy as np
from scipy import ndimage
from scipy import signal
import math

# Smoothening for better identification of the peaks in a graph. Could have used Gaussian Kernels to do the same
# but it seemed better visual effects were given when this algorithm was followed ( Again, based on original CASA495)
#MAT is the 2D matrix to be smoothed.
#KER is either
#(1)a scalar
#(2)a matrix which is used as the averaging kernel.

def twoDsmooth(mat,ker):
    try:
        len(ker)
        kmat = ker
        
    except:
        kmat = np.ones((ker,ker))
        kmat = kmat / pow(ker, 2)

    [kr,kc] = list(kmat.shape)
    if (kr%2 == 0):
        conmat = np.ones((2,1))
        kmat = signal.convolve2d(kmat,conmat,'symm','same')
        kr = kr + 1

    if (kc%2 == 0):
        conmat = np.ones((2,1))
        kmat = signal.convolve2d(kmat,conmat,'symm','same')
        kc = kc + 1

    [mr,mc] = list(mat.shape)
    fkr = math.floor(kr/2)
    fkc = math.floor(kc/2)
    rota = np.rot90(kmat,2)
    mat=signal.convolve2d(mat,rota,'same','symm')
    return mat