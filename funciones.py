import numpy as np
from matplotlib import pyplot as plt
from math import factorial

import pandas as pd

#%%
#Funcion encontrada para el convinatorio
import operator as op
from functools import reduce


def numeral(n):
    '''Calcula el factorial de un número'''
    if n==0:
        fact = 0
    else:
        fact = numeral(n-1)*n
    return fact

#%% Convinatorios

def ncr(n,r):
    c = factorial(n)/(factorial(r)*factorial(n-r))
    return c

#%% Distribuciones  discretas

def binom(N,p,k):
    B = ncr(N,k) * p**k * (1-p)**(N-k)
    return B

def hiperg(N,K,n,k):
    H = ncr(K,k)*ncr(N-K,n-k)/ncr(N,n)
    return H

def recursive_binom(n,p,k):
    q = 1 - p
    if k == 0:
        return q**n
    else:     
        c = (p*(n-k+1))/(q*k)
        return  c * recursive_binom(n,k-1,p)

def recursive_hiperg(N, K, n, k, h0=None, stirling=False):
    if k == 0:
        if h0 != None: #si la hiperg con k=0 está definida, uso ese valor
            return h0
        else: #si no lo está puedo calcularlo en el momento posta o con aprox de Stirling
            if stirling == True: 
                return com_stirling((N-K),n)/com_stirling(N,n)
            else:
                return ncr((N-K),n)/ncr(N,n)
    else:
        c = (K-k+1)*(n-k+1)/(k*(N-K-n+k))
        return c*recursive_hiperg(N,K,n,k-1, h0, stirling)

#%% Metodos utilizando stirling

def log_num_stirling(n):
    s = n*np.log(n) - n + 0.5 *np.log(2*np.pi*n)+1/(12*n)
    return s

def com_stirling(a,b):
    c = np.exp(log_num_stirling(a)-log_num_stirling(b)-log_num_stirling(a-b))
    return c


#%% Ploteo de Histogramas

# Fomar vieja(con Pandas)

def plot_histogram(data, bins, ticks=5, xlabel='X', ylabel='Histograma', 
                   color='', ecolor='' ,ax = None, alpha = 1,
                   density = True):
    N = len(data)
    num_bins = len(bins)
    xmin = min(bins)
    xmax = max(bins)

    if ax is None:
        ax = plt.gca() #ax es un eje del plot

    hist, _ = np.histogram(data, bins=bins, density=density)

    df = pd.DataFrame({'histograma': hist, 'error': np.sqrt(hist/N)})

    #Para definir el ax de matplotlib usamos el data frame de pandas
    ax = df.plot.bar(y='histograma', yerr='error', width=1, edgecolor='k',
                     ecolor='red', capsize=3, label=ylabel, 
                     ax=ax, color ='blue', alpha=alpha)
    
    plt.grid(True, axis = 'y')
#    plt.xticks(np.linspace(0, num_bins - 2, ticks), np.linspace(xmin, xmax, ticks), rotation='horizontal');
    plt.xlabel(xlabel);
    plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
    return ax

# Forma nueva (sin Pandas)

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
            edgecolor='black', ecolor=ecolor, capsize=2, alpha = alpha)
    

#    plt.xticks(np.linspace(min(bins), max(num_bins) - 2, ticks), np.linspace(xmin, xmax, ticks), rotation='horizontal');
    plt.xlabel(xlabel);
    return ax




def binom_stirling(N,p,k):
    B = com_stirling(N,k) * p**k * (1-p)**(N-k)
    return B

def cuadrados_minimos(M,V,Y):
    pass


