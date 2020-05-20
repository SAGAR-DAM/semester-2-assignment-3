'''Problem 5 assignment 3
comparisn of timing between FFT and normal DFT
NAME: SAGAR DAM;  DNAP'''

#importing headers
import timeit 
import numpy as np
import matplotlib.pyplot as plt


N=1
wofft=[]
fft=[]
index=[]
for i in range(N):
    a0=500
    mysetup = "a=10"
    
#calculating time for without FFT module
# code snippet whose execution time is to be measured 
    mycodewithoutfft1 = ''' 
import numpy as np
import matplotlib.pyplot as plt

a=10
h=0.1

x=np.arange(-a,a+h,h)
X=np.ones(len(x))
W=np.zeros(len(x),dtype=complex)
    
N=int(len(x))
freq=np.zeros(N)
kmin=-np.pi*(N-1)/(h*N)

for j in range(len(x)):
    freq[j]=kmin+j*2*np.pi/(h*N)
    
for i in range(len(x)-1):
    for k in range(len(x)-1):
        W[i]=W[i]+(X[k]*np.exp(-complex(0,1)*freq[i]*x[k]))
    W[i]=W[i]/np.sqrt(N/np.pi)
'''
  
# timeit statement 
    t1=timeit.timeit(setup=mysetup,stmt = mycodewithoutfft1, number = 1)
    wofft.append(t1)
    
    
    mycodewithoutfft2 = ''' 
import numpy as np
import matplotlib.pyplot as plt

a=15
h=0.1

x=np.arange(-a,a+h,h)
X=np.ones(len(x))
W=np.zeros(len(x),dtype=complex)
    
N=int(len(x))
freq=np.zeros(N)
kmin=-np.pi*(N-1)/(h*N)

for j in range(len(x)):
    freq[j]=kmin+j*2*np.pi/(h*N)
    
for i in range(len(x)-1):
    for k in range(len(x)-1):
        W[i]=W[i]+(X[k]*np.exp(-complex(0,1)*freq[i]*x[k]))
    W[i]=W[i]/np.sqrt(N/np.pi)
'''
  
# timeit statement 
    t1=timeit.timeit(setup=mysetup,stmt = mycodewithoutfft2, number = 1)
    wofft.append(t1)
    
    
    mycodewithoutfft3 = ''' 
import numpy as np
import matplotlib.pyplot as plt

a=20
h=0.1

x=np.arange(-a,a+h,h)
X=np.ones(len(x))
W=np.zeros(len(x),dtype=complex)
    
N=int(len(x))
freq=np.zeros(N)
kmin=-np.pi*(N-1)/(h*N)

for j in range(len(x)):
    freq[j]=kmin+j*2*np.pi/(h*N)
    
for i in range(len(x)-1):
    for k in range(len(x)-1):
        W[i]=W[i]+(X[k]*np.exp(-complex(0,1)*freq[i]*x[k]))
    W[i]=W[i]/np.sqrt(N/np.pi)
'''
  
# timeit statement 
    t1=timeit.timeit(setup=mysetup,stmt = mycodewithoutfft3, number = 1)
    wofft.append(t1)
    
    
    mycodewithoutfft4 = ''' 
import numpy as np
import matplotlib.pyplot as plt

a=25
h=0.1

x=np.arange(-a,a+h,h)
X=np.ones(len(x))
W=np.zeros(len(x),dtype=complex)
    
N=int(len(x))
freq=np.zeros(N)
kmin=-np.pi*(N-1)/(h*N)

for j in range(len(x)):
    freq[j]=kmin+j*2*np.pi/(h*N)
    
for i in range(len(x)-1):
    for k in range(len(x)-1):
        W[i]=W[i]+(X[k]*np.exp(-complex(0,1)*freq[i]*x[k]))
    W[i]=W[i]/np.sqrt(N/np.pi)
'''
  
# timeit statement 
    t1=timeit.timeit(setup=mysetup,stmt = mycodewithoutfft4, number = 1)
    wofft.append(t1)
    
    
    mycodewithoutfft5 = ''' 
import numpy as np
import matplotlib.pyplot as plt

a=30
h=0.1

x=np.arange(-a,a+h,h)
X=np.ones(len(x))
W=np.zeros(len(x),dtype=complex)
    
N=int(len(x))
freq=np.zeros(N)
kmin=-np.pi*(N-1)/(h*N)

for j in range(len(x)):
    freq[j]=kmin+j*2*np.pi/(h*N)
    
for i in range(len(x)-1):
    for k in range(len(x)-1):
        W[i]=W[i]+(X[k]*np.exp(-complex(0,1)*freq[i]*x[k]))
    W[i]=W[i]/np.sqrt(N/np.pi)
'''
  
# timeit statement 
    t1=timeit.timeit(setup=mysetup,stmt = mycodewithoutfft5, number = 1)
    wofft.append(t1)
    
    
    mycodewithoutfft6 = ''' 
import numpy as np
import matplotlib.pyplot as plt

a=35
h=0.1

x=np.arange(-a,a+h,h)
X=np.ones(len(x))
W=np.zeros(len(x),dtype=complex)
    
N=int(len(x))
freq=np.zeros(N)
kmin=-np.pi*(N-1)/(h*N)

for j in range(len(x)):
    freq[j]=kmin+j*2*np.pi/(h*N)
    
for i in range(len(x)-1):
    for k in range(len(x)-1):
        W[i]=W[i]+(X[k]*np.exp(-complex(0,1)*freq[i]*x[k]))
    W[i]=W[i]/np.sqrt(N/np.pi)
'''
  
# timeit statement 
    t1=timeit.timeit(setup=mysetup,stmt = mycodewithoutfft6, number = 1)
    wofft.append(t1)
    
    
    mycodewithoutfft7 = ''' 
import numpy as np
import matplotlib.pyplot as plt

a=40
h=0.1

x=np.arange(-a,a+h,h)
X=np.ones(len(x))
W=np.zeros(len(x),dtype=complex)
    
N=int(len(x))
freq=np.zeros(N)
kmin=-np.pi*(N-1)/(h*N)

for j in range(len(x)):
    freq[j]=kmin+j*2*np.pi/(h*N)
    
for i in range(len(x)-1):
    for k in range(len(x)-1):
        W[i]=W[i]+(X[k]*np.exp(-complex(0,1)*freq[i]*x[k]))
    W[i]=W[i]/np.sqrt(N/np.pi)
'''
  
# timeit statement 
    t1=timeit.timeit(setup=mysetup,stmt = mycodewithoutfft7, number = 1)
    wofft.append(t1)
    
    
    mycodewithoutfft8 = ''' 
import numpy as np
import matplotlib.pyplot as plt

a=45
h=0.1

x=np.arange(-a,a+h,h)
X=np.ones(len(x))
W=np.zeros(len(x),dtype=complex)
    
N=int(len(x))
freq=np.zeros(N)
kmin=-np.pi*(N-1)/(h*N)

for j in range(len(x)):
    freq[j]=kmin+j*2*np.pi/(h*N)
    
for i in range(len(x)-1):
    for k in range(len(x)-1):
        W[i]=W[i]+(X[k]*np.exp(-complex(0,1)*freq[i]*x[k]))
    W[i]=W[i]/np.sqrt(N/np.pi)
'''
  
# timeit statement 
    t1=timeit.timeit(setup=mysetup,stmt = mycodewithoutfft8, number = 1)
    wofft.append(t1)
    
    mycodewithoutfft9 = ''' 
import numpy as np
import matplotlib.pyplot as plt

a=50
h=0.1

x=np.arange(-a,a+h,h)
X=np.ones(len(x))
W=np.zeros(len(x),dtype=complex)
    
N=int(len(x))
freq=np.zeros(N)
kmin=-np.pi*(N-1)/(h*N)

for j in range(len(x)):
    freq[j]=kmin+j*2*np.pi/(h*N)
    
for i in range(len(x)-1):
    for k in range(len(x)-1):
        W[i]=W[i]+(X[k]*np.exp(-complex(0,1)*freq[i]*x[k]))
    W[i]=W[i]/np.sqrt(N/np.pi)
'''
  
# timeit statement 
    t1=timeit.timeit(setup=mysetup,stmt = mycodewithoutfft9, number = 1)
    wofft.append(t1)
    
    mycodewithoutfft10 = ''' 
import numpy as np
import matplotlib.pyplot as plt

a=55
h=0.1

x=np.arange(-a,a+h,h)
X=np.ones(len(x))
W=np.zeros(len(x),dtype=complex)
    
N=int(len(x))
freq=np.zeros(N)
kmin=-np.pi*(N-1)/(h*N)

for j in range(len(x)):
    freq[j]=kmin+j*2*np.pi/(h*N)
    
for i in range(len(x)-1):
    for k in range(len(x)-1):
        W[i]=W[i]+(X[k]*np.exp(-complex(0,1)*freq[i]*x[k]))
    W[i]=W[i]/np.sqrt(N/np.pi)
'''
  
# timeit statement 
    t1=timeit.timeit(setup=mysetup,stmt = mycodewithoutfft10, number = 1)
    wofft.append(t1)
    
    mycodewithoutfft11 = ''' 
import numpy as np
import matplotlib.pyplot as plt

a=60
h=0.1

x=np.arange(-a,a+h,h)
X=np.ones(len(x))
W=np.zeros(len(x),dtype=complex)
    
N=int(len(x))
freq=np.zeros(N)
kmin=-np.pi*(N-1)/(h*N)

for j in range(len(x)):
    freq[j]=kmin+j*2*np.pi/(h*N)
    
for i in range(len(x)-1):
    for k in range(len(x)-1):
        W[i]=W[i]+(X[k]*np.exp(-complex(0,1)*freq[i]*x[k]))
    W[i]=W[i]/np.sqrt(N/np.pi)
'''
  
# timeit statement 
    t1=timeit.timeit(setup=mysetup,stmt = mycodewithoutfft11, number = 1)
    wofft.append(t1)
    
    
    
# Calculating time for with FFT module    
    
    
    
    mycodewithfft1 = ''' 
import numpy as np
import matplotlib.pyplot as plt
a=10
h=0.1
x=np.arange(-a,a,h)
g=np.ones(len(x))
sp = np.fft.fft(g)
freq = np.fft.fftfreq(x.size)*2*np.pi/h
sp*=h*np.exp(-complex(0,1)*freq*(a))/(np.sqrt(2*np.pi))
'''
# timeit statement 
    t1=timeit.timeit(setup=mysetup,stmt = mycodewithfft1, number = 100)
    fft.append(t1/100)
    
    
    mycodewithfft2 = ''' 
import numpy as np
import matplotlib.pyplot as plt
a=15
h=0.1
x=np.arange(-a,a,h)
g=np.ones(len(x))
sp = np.fft.fft(g)
freq = np.fft.fftfreq(x.size)*2*np.pi/h
sp*=h*np.exp(-complex(0,1)*freq*(a))/(np.sqrt(2*np.pi))
'''
# timeit statement 
    t1=timeit.timeit(setup=mysetup,stmt = mycodewithfft2, number = 100)
    fft.append(t1/100)
    
    
    mycodewithfft3 = ''' 
import numpy as np
import matplotlib.pyplot as plt
a=20
h=0.1
x=np.arange(-a,a,h)
g=np.ones(len(x))
sp = np.fft.fft(g)
freq = np.fft.fftfreq(x.size)*2*np.pi/h
sp*=h*np.exp(-complex(0,1)*freq*(a))/(np.sqrt(2*np.pi))
'''
# timeit statement 
    t1=timeit.timeit(setup=mysetup,stmt = mycodewithfft3, number = 100)
    fft.append(t1/100)
    
    
    mycodewithfft4 = ''' 
import numpy as np
import matplotlib.pyplot as plt
a=25
h=0.1
x=np.arange(-a,a,h)
g=np.ones(len(x))
sp = np.fft.fft(g)
freq = np.fft.fftfreq(x.size)*2*np.pi/h
sp*=h*np.exp(-complex(0,1)*freq*(a))/(np.sqrt(2*np.pi))
'''
# timeit statement 
    t1=timeit.timeit(setup=mysetup,stmt = mycodewithfft4, number = 100)
    fft.append(t1/100)
    
    
    mycodewithfft5 = ''' 
import numpy as np
import matplotlib.pyplot as plt
a=30
h=0.1
x=np.arange(-a,a,h)
g=np.ones(len(x))
sp = np.fft.fft(g)
freq = np.fft.fftfreq(x.size)*2*np.pi/h
sp*=h*np.exp(-complex(0,1)*freq*(a))/(np.sqrt(2*np.pi))
'''
# timeit statement 
    t1=timeit.timeit(setup=mysetup,stmt = mycodewithfft5, number = 100)
    fft.append(t1/100)
    
    
    mycodewithfft6 = ''' 
import numpy as np
import matplotlib.pyplot as plt
a=35
h=0.1
x=np.arange(-a,a,h)
g=np.ones(len(x))
sp = np.fft.fft(g)
freq = np.fft.fftfreq(x.size)*2*np.pi/h
sp*=h*np.exp(-complex(0,1)*freq*(a))/(np.sqrt(2*np.pi))
'''
# timeit statement 
    t1=timeit.timeit(setup=mysetup,stmt = mycodewithfft6, number = 100)
    fft.append(t1/100)
    
    
    mycodewithfft7 = ''' 
import numpy as np
import matplotlib.pyplot as plt
a=40
h=0.1
x=np.arange(-a,a,h)
g=np.ones(len(x))
sp = np.fft.fft(g)
freq = np.fft.fftfreq(x.size)*2*np.pi/h
sp*=h*np.exp(-complex(0,1)*freq*(a))/(np.sqrt(2*np.pi))
'''
# timeit statement 
    t1=timeit.timeit(setup=mysetup,stmt = mycodewithfft7, number = 100)
    fft.append(t1/100)
    
    mycodewithfft8 = ''' 
import numpy as np
import matplotlib.pyplot as plt
a=45
h=0.1
x=np.arange(-a,a,h)
g=np.ones(len(x))
sp = np.fft.fft(g)
freq = np.fft.fftfreq(x.size)*2*np.pi/h
sp*=h*np.exp(-complex(0,1)*freq*(a))/(np.sqrt(2*np.pi))
'''
# timeit statement 
    t1=timeit.timeit(setup=mysetup,stmt = mycodewithfft8, number = 100)
    fft.append(t1/100)
    
    
    mycodewithfft9 = ''' 
import numpy as np
import matplotlib.pyplot as plt
a=50
h=0.1
x=np.arange(-a,a,h)
g=np.ones(len(x))
sp = np.fft.fft(g)
freq = np.fft.fftfreq(x.size)*2*np.pi/h
sp*=h*np.exp(-complex(0,1)*freq*(a))/(np.sqrt(2*np.pi))
'''
# timeit statement 
    t1=timeit.timeit(setup=mysetup,stmt = mycodewithfft9, number = 100)
    fft.append(t1/100)
    
    
    mycodewithfft10 = ''' 
import numpy as np
import matplotlib.pyplot as plt
a=55
h=0.1
x=np.arange(-a,a,h)
g=np.ones(len(x))
sp = np.fft.fft(g)
freq = np.fft.fftfreq(x.size)*2*np.pi/h
sp*=h*np.exp(-complex(0,1)*freq*(a))/(np.sqrt(2*np.pi))
'''
# timeit statement 
    t1=timeit.timeit(setup=mysetup,stmt = mycodewithfft10, number = 100)
    fft.append(t1/100)
    
    
    mycodewithfft11 = ''' 
import numpy as np
import matplotlib.pyplot as plt
a=60
h=0.1
x=np.arange(-a,a,h)
g=np.ones(len(x))
sp = np.fft.fft(g)
freq = np.fft.fftfreq(x.size)*2*np.pi/h
sp*=h*np.exp(-complex(0,1)*freq*(a))/(np.sqrt(2*np.pi))
'''
# timeit statement 
    t1=timeit.timeit(setup=mysetup,stmt = mycodewithfft11, number = 100)
    fft.append(t1/100)
    
    
index=np.arange(10,65,5)
plt.plot(index,wofft,'r',label='without numpy')
plt.plot(index,fft,'k',label='with numpy')
plt.suptitle('comparisn of time for FFT algo:')
plt.title('(taking the FFT of a constant function)')
plt.xlabel('n value')
plt.ylabel('time taken for 1 iteration')
plt.legend()
plt.grid()
plt.show()