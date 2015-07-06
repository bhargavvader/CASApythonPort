import numpy as np
from scipy.fftpack import rfft, irfft, fftfreq, fft, ifft
import math

#time-frequency analysis
#X is the time domain signal
#AWIN is an analysis window
#TIMESTEP is the # of samples between adjacent time windows.
#NUMFREQ is the # of frequency components per time point.
#
#TFMAT complex matrix time-freq representation

def tfanalysis(x,awin,timestep,numfreq):
    nsamp=x.size
    wlen=awin.size
    x=np.reshape(x,-1,'F')
    awin=np.reshape(awin,-1,'F') #make inputs go column-wise
    numtime=math.ceil((nsamp-wlen+1)/timestep)
    tfmat=np.zeros((numfreq,numtime+1))+0j
    sind=None
    for i in range(0,numtime):
        sind=((i)*timestep)
        tfmat[:,i]=fft(x[sind:(sind+wlen)]*awin,numfreq)
    i=i+1
    sind=((i)*timestep)
    lasts = min(sind,x.size-1)
    laste=min((sind+wlen),x.size-1)
    tfmat[:,-1]=fft(np.hstack((x[lasts:laste],np.zeros(wlen-(laste-lasts))))*awin,numfreq)
    return tfmat