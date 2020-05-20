''' Problem 6 without numpy module
making FT of constant function
Name: SAGAR DAM; DNAP'''

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

a=10
h=0.1
x=np.arange(-10,10,0.1,dtype=complex)
y=np.arange(-10,10,0.1,dtype=complex)
a = complex(np.mgrid[:len(x), :len(y)][1])

def f(t,v):
    return np.exp(-t**2-v**2)

#print(a)
for i in range(len(x)):
    for j in range(len(y)):
        a[i][j]=f(x[i],y[j])
        
sp=np.fft.fft2(a)
fx = np.fft.fftfreq(x.size)*2*np.pi/h
fy = np.fft.fftfreq(y.size)*2*np.pi/h
#sp*=h*np.exp(-complex(0,1)*freq*(a))/(np.sqrt(2*np.pi))
FX,FY=np.meshgrid(fx,fy)


fig = plt.figure(figsize=(10,6))
ax1 = fig.add_subplot(111, projection='3d')

mycmap = plt.get_cmap('gist_earth')
ax1.set_title('gist_earth color map')
surf1 = ax1.plot_surface(FX, FY, sp, cmap=mycmap)
fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=5)
plt.show()
#print(sp)