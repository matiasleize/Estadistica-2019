import numpy as np
from matplotlib import pyplot as plt
import seaborn as sb
from math import factorial
from funciones import ncr,binom,recursive_hiperg

#%% Ejercicio 3
p = 1/6
q = 1 - p

for n in range(11,90):
    a = ncr(n-1,10)*(1/q)
    b = ncr(n,10)
    c = ncr(n+1,10)*q
    if (b>=a) & (b>=c):
        print (n)   
print(a,b,c)
#%% Ejercicio 4e
#resultado = np.zeros(10)
#for n in range(3):
        
#    resultado[n] = 
#%% EJercicio 8
#Inciso a
resultado = np.zeros(10)
for k in range(0,10):
    resultado[i] = binom(500,0.1,k)
    
a = 1-resultado.sum()
print(a) 

#%%
#Inciso b

#Calculo a mano la hipergeometrica con H(k=0): 
h0 = 0.0050144117958

resultado = np.zeros(10)
for k in range(10):
    resultado[k] = recursive_hiperg(5000,50,500,k, h0 = h0, stirling=False)
a = 1-resultado.sum()
print(a)
#%%
#Inciso c
#Calculo a mano la hipergeometrica con H(k=0): 
h0 = 0.0050144117958
total = np.ones(10000)
for n in range(36400,36494):
    resultado = np.zeros(10)
    for k in range(10):
        resultado[k] = recursive_hiperg(n,50,500,k, h0 = h0, stirling=False) - binom(500,0.1,k)
    total[n-32000] = abs(resultado.sum())
    print(n)
print(min(total))
#%% EJercicio 10
#Inciso a
resultado = np.zeros(100)
for n in range(1,100):
    resultado[n] = 1-binom(n,0.182,1)-binom(n,0.182,0)
a = resultado
print(a)  

#%%    
#ejercicio 15

resultado = np.zeros(35)
for i in range(100,130):
    resultado[i-100] = ncr(i-1,99)*((0.8)**99)*(0.2**(i-100))

a = 1 - resultado.sum()
print(a)


#%%#Ejer 8:

#a)
import matplotlib.pyplot as plt
import numpy as np

def fa(k):
    if k==0:
        return (1-0.01)**(500)
    else:
        return fa(k-1)*(500-(k-1))*0.01*(k*(1-0.01))**(-1)

suma=0

for i in range(10):
    suma=suma+fa(i)
print(suma)    

rta = 1-suma

print(rta)

x=range(500)

ya=[]

for j in range(500):
    ya.append(fa(j))
    
plt.plot(x, ya, 'b.', linewidth = 2, label = 'binomial')
plt.legend(loc = 'best')
plt.grid(1)

plt.show

#b)
import matplotlib.pyplot as plt
import numpy as np

prod=1

for i in range(50):
    prod=prod*(4500-i)*((5000-i)**(-1))
print(prod)

#prod=0.005014411795816377
    
def fb(k):
    if k==0:
        return 0.005014411795816377
    else:
        return fb(k-1)*(50-(k-1))*(500-(k-1))*(k*(4451+k-1))**(-1)

sumb=0

for i in range(10):
    sumb=sumb+fb(i)
print(sumb)    

rtb=1-sumb

print(rtb)

xb=range(50)
yb=[]

for j in range(50):
    yb.append(fb(j))
    
plt.plot(xb, yb, 'r.', linewidth = 2, label = 'hipergeometrica')
plt.legend(loc = 'best')
plt.grid(1)

plt.show
