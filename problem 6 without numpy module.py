''' Problem 6 without numpy module
making FT of constant function
Name: SAGAR DAM; DNAP'''

import numpy as np
import matplotlib.pyplot as plt

a=100
h=0.1

#set variables (the function is trivially zero here)
x=np.arange(-a,a+h,h)
X=np.ones(len(x))
W=np.zeros(len(x),dtype=complex)
N=int(len(x))
freq=np.zeros(N)
kmin=-np.pi*(N-1)/(h*N)

#taking k points
for j in range(len(x)):
    freq[j]=kmin+j*2*np.pi/(h*N)

#doing FT
for i in range(len(x)-1):
    for k in range(len(x)-1):
        W[i]=W[i]+(X[k]*np.exp(complex(0,1)*freq[i]*x[k]))
    W[i]=W[i]/np.sqrt(N/np.pi)

#Plotting the FT wrt k.
plt.plot(freq,W,'k-',label='numerical solution')
plt.gca().set_xlim(-10,10)
plt.legend()
plt.grid()
plt.show()