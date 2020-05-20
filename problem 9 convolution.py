''' Problem 9 self convolution 
convolution of a function
SAGAR DAM; DNAP'''

import numpy as np
from matplotlib import pyplot as plt

a=100
h=0.1
x=np.arange(-a,a+h,h)
T=np.arange(-10,10,h)
I=np.zeros(len(T))
box=np.zeros(len(T))
def f(t):
    if(abs(t)<=1):
        z=1
    else:
        z=0
    return z

for i in range(len(T)):
    S=0
    for j in range(len(x)):
        S=S+h*f(x[j])*f(T[i]-x[j])
    I[i]=S
    box[i]=f(T[i])


plt.plot(T,I,label='Convolution of Box')
plt.plot(T,box,label='given box function')
plt.gca().set_xlim(-5,5)
plt.legend()
plt.grid()
plt.show()