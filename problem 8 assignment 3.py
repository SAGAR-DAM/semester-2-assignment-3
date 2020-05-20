''' Problem 8
2D Fourier transform with fft2
NAME: SAGAR DAM;  DNAP'''
import numpy as np
import matplotlib.pyplot as plt

unitcomplex=0.00+1.00j
xa=-15.0
xb=15.0
ya=-15.0
yb=15.0
h=0.4
k=h*(yb-ya)/(xb-xa)
x=np.arange(xa,xb+h,h,dtype=float)
y=np.arange(ya,yb+k,k,dtype=float)
g=np.zeros(len(x),dtype=float)   
ft=np.zeros(len(x),dtype=complex)   

def f(x,y):
    return np.exp(-(x**2+y**2))

X, Y = np.meshgrid(x,y)   
gaussian=f(X, Y)
wx=np.fft.fftshift(np.fft.fftfreq(len(x),h))
wy=np.fft.fftshift(np.fft.fftfreq(len(y),k))
WX,WY=np.meshgrid(wx,wy)
fk=np.fft.fftshift(np.fft.fft2(gaussian,norm='ortho'))
ft = abs(h*k*len(x)*np.exp(unitcomplex*(wx*xa-wy*yb))*fk/np.sqrt((2*np.pi)))

fig=plt.figure()
fourier=plt.axes(projection='3d')
fourier.plot_wireframe(WX,WY,ft, color='k')