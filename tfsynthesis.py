from scipy.fftpack import rfft, irfft, fftfreq, fft, ifft
import numpy as np
import math

#time-frequency synthesis
#TIMEFREQMAT is the complex matrix time-freq representation
#SWIN is the synthesis window
#TIMESTEP is the # of samples between adjacent time windows.
#NUMFREQ is the # of frequency components per time point.
#
#X contains the reconstructed signal.

def tfsynthesis(timefreqmat,swin,timestep,numfreq):
	timefreqmat=np.asarray(timefreqmat)
	swin=np.reshape(swin,-1,'F')
	winlen=swin.size
	(numfreq, numtime)=timefreqmat.shape
	ind=np.fmod(np.array(range(0,winlen)),numfreq)
	x=np.zeros(((numtime-1)*timestep+winlen))
	for i in range(0,numtime):
		temp=numfreq*np.real(ifft(timefreqmat[:,i]))
		sind=((i)*timestep)
		for i in range(0,winlen):
			x[sind+i]=x[sind+i]+temp[ind[i]]*swin[i]
	return x	