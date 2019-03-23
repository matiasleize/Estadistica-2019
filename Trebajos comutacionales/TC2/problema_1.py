# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
#from scipy.stats import chi2, norm, cauchy
from scipy.stats import cauchy
import scipy
from matplotlib import pyplot as plt
#import pandas as pd
#import numpy as np
#from matplotlib.ticker import StrMethodFormatter

#%%

def plot_histogram(data, bins, ticks=5, xlabel='X', ylabel='Histograma', density=True, ax = None,
                   color='', ecolor='red' , alpha = 1):      

    N = len(data)
    num_bins = len(bins)
    xmin = min(bins)
    xmax = max(bins)
    
    
    if ax is None:
        ax = plt.gca()
    
    hist, _ = np.histogram(data, bins=bins, density=density)
    
    normed = N if density else 1
    error =  np.sqrt(hist / normed)  # por qué es así el error?
    
    
    
    # Se le puede pasar un yerr y xerr, como a errorbar
    # ecolor es el color de los errores y capsize es el tamaño del fin de barra de error
    plt.bar(x=bins[:-1], height=hist, width=np.diff(bins), yerr=error, 
            edgecolor='black', ecolor=ecolor, capsize=2, alpha=alpha, align= 'edge')    

#    plt.xticks(
#            np.linspace(min(bins), max(bins) - 1, ticks), #Chequear y pensar esta linea
#               np.linspace(xmin, xmax, ticks), rotation='horizontal');
    plt.xlabel(xlabel, fontsize = 20);
    plt.ylabel(ylabel, fontsize = 20);
    return ax




#Otra idea: poner un f_m mayor al maximo y anda igual
def funciones_montecarlo(a, b, f, n):
    
    '''Dada una densidad de probabilidad, esta función devuelve un array de
    numpy con los datos necesarios para realizar el histograma de dicha 
    distribución mediante el método de montecarlo. Es necesario pasar como
    argumento el intervalo que queremos muestrar (a y b) y el número de puntos
    que queremos utilizar para el montercarlo, denotado como n.'''
    
    x_m= scipy.optimize.fmin(lambda x: -f(x), 0)[0]
    f_m = f(x_m)

    y = np.random.uniform(0, 1, n) 
    z = np.random.uniform(0, 1, n)
    
    u  = a + (b - a) * y
    v = f_m * z
    x = u[v<=h(u)]
    
    return x, v
#%% Punto a
n = int(1e4)

a = np.random.uniform(0,1,n)
u  = -np.pi/2 + np.pi * a
b = np.tan(u)


#plt.close()
plt.figure(figsize=(7 * 1.78, 7))
ax = plot_histogram(b, np.linspace(-25, 25, 50))
plt.grid(True, axis='y')


t = np.linspace(-25,25,int(1e4))
plt.plot(t, cauchy.pdf(t), 'g-')

plt.xlim(-25,25)
plt.show()
#%% Punto b
n = int(1e7)

#Cauchy
h = lambda x: 1/(np.pi*(1+x**2)) #chequeado que tiene el maximo en 0.3 y pico!

a,_ = funciones_montecarlo(-100,100, h, n)

long = np.sqrt(len(a))
#long = 10


plt.close()
plt.figure(figsize=(7 * 1.78, 7))
ax = plot_histogram(a, np.linspace(-100,100, int(long)))
plt.grid(True, axis='y')

t = np.linspace(-100,100,int(1e4))
plt.plot(t, cauchy.pdf(t) , 'g-') #Si pongo scale = np.pi se ve mejor, pensar porque..  (no tiene sentido)
plt.xlim(-25,25)
plt.show()

#%%


##%%
#n = int(1e7)
#
##Función cuadrática de pruebita
#f = lambda x: -x**2+6        
#
##Distribución Gaussiana
#g  = lambda x,mu,sigma: 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (x - mu)**2 / (2 * sigma**2))
#
#
##Distribución normal
#g_1 = lambda x: g(x,mu=0,sigma=1)
#
#a, v = funciones_montecarlo(-100,100, g_1, n)
#plt.close()
#plt.figure()
#plt.hist(a, color = 'g', bins = int(np.sqrt(len(a))),
#         density=True)
