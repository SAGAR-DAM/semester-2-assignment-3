''' Assignment 3 problem 6
fourier transform of sinc function using numpy
name: SAGAR DAM;  DNAP'''

import numpy as np
import matplotlib.pyplot as plt


a=1000
h=0.001
x=np.arange(-a,a,h)
g=np.ones(len(x))

sp = np.fft.fft(g)
freq = np.fft.fftfreq(x.size)*2*np.pi/h
sp*=h*np.exp(-complex(0,1)*freq*(a))/(np.sqrt(2*np.pi))

    
plt.plot(freq,sp,'r',label='FT of constant')
plt.legend()
plt.grid()
plt.gca().set_xlim(-10,10)
plt.show()