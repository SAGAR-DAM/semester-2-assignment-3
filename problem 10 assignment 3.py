''' Problem 10 assignment 3
FT of noise data
SAGAR DAM'''
import numpy as np
import matplotlib.pyplot as plt

file = open("noise.txt","r")
data = np.loadtxt("noise.txt")
n=len(data)
file.close()
h=1
x=np.arange(0,n,1)

#plotting data points
plt.plot(x,data,color='orange',label='data point plot')
plt.xlabel('x')
plt.ylabel('noise value')
plt.legend()
plt.grid()
plt.title('data point plotting')
plt.show()

#Doing DFT of given data (using numpy package)
k=np.fft.fftshift(np.fft.fftfreq(n,h))
gk=abs(np.fft.fftshift(np.fft.fft(data,norm='ortho')))
plt.plot(k,gk,'r',label='DFT')
plt.suptitle('Plotting the DFT of given data')
plt.xlabel('w')
plt.ylabel('g(w)')
plt.legend()
plt.grid()
plt.show()

#power spectrum;  p~A**2
plt.plot(k,gk**2)
plt.xlabel('w')
plt.ylabel('Intensity(w)')
plt.title('power spectrum')
plt.legend()
plt.show()

#Binning the given spectrum
b=10
no=int(n/b)
ko=np.fft.fftshift(np.fft.fftfreq(no,h))
hk=np.ones(no*b,dtype=float)
hk=hk.reshape(b,no)
s=np.zeros(no,dtype=float)
h=data[0:no*b].reshape(b,no)
for i in range(10):
    hk[i]=np.fft.fftshift(abs(np.fft.fft(h[i],norm='ortho')))
    s=s+((hk[i])**2)
plt.plot(ko,s,color='violet',label='binned power spectrum')
plt.legend()
plt.grid()
plt.title('binned power spectrum')
plt.show()