''' Assignment 3 problem 1
fourier transform of sinc function using numpy
name: SAGAR DAM;  DNAP'''

import numpy as np
import matplotlib.pyplot as plt

def f(t):
    if(t==0):
        z=1
    else:
        z=np.sin(t)/t
    return z

a=1000
h=0.001
x=np.arange(-a,a,h)
g=np.zeros(len(x))
for j in range(len(x)):
    g[j]=f(x[j])
sp = np.fft.fft(g)
freq = np.fft.fftfreq(x.size)*2*np.pi/h
sp*=h*np.exp(-complex(0,1)*freq*(a))/(np.sqrt(2*np.pi))

w=np.arange(-10,10,0.1)
F=np.zeros(len(w))
def true(t):
    if(abs(t)<=1):
        z=(np.pi/2)**0.5
    else:
        z=0
    return z
for j in range(len(w)):
    F[j]=true(w[j])
    
plt.plot(freq,sp,'r',label='numerical FT')
plt.plot(w,F,'go',markersize=1.5,label='analytic FT')
plt.title('FT of sinc with numpy')
plt.legend()
plt.grid()
plt.gca().set_xlim(-10,10)
plt.show()