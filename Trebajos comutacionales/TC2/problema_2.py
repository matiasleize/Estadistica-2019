#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 01:48:00 2019

@author: matias
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import norm
import random
#%%

def propagar_f (x_a):
    
    a_1 = popt[1]
    a_2 = popt[0]    
    var_a2 = pcov[0,0]
    var_a1 = pcov[1,1]
    cov_a = pcov[0,1]
    
    y_a = a_1 + a_2 * x_a
    var_ya = x_a**2 * var_a2 + var_a1 + 2 * x_a * cov_a
    mal_var_ya = x_a**2 * var_a2 + var_a1 
    return y_a, var_ya, mal_var_ya

f = lambda x, A, B: A * x + B


#%%

x = np.linspace(2,3,11)
y = np.array([2.78, 3.29, 3.29, 3.33, 3.23, 3.69, 3.46, 3.87, 3.62, 3.40, 3.99])
y_error = 0.3 * np.ones(len(y))



# Ajustamos, pero con las funciónes logaritmicas. Usamos propagación de errores
popt, pcov = curve_fit(f, x, y, sigma = y_error, 
                       absolute_sigma=True)


#plt.plot(x,y, '.')
plt.errorbar(x, y, yerr = y_error, fmt = 'r.')

t = np.linspace(0,5,100)
plt.plot(t,f(t, *popt))
plt.xlim(0,5)

#%% Ejercicio b
#FALTA: Encuentre qué valor de x a minimiza el error de y a ,
#e interprete la magnitud de este valor mı́nimo. Discuta por qué el error aumenta para
#valores de x a alejados de la región donde se hicieron las mediciones.


t = np.linspace(0,10,1000)


t_y = np.zeros(len(t))
t_ey = np.zeros(len(t))
for i in range (len(t)):
    t_y[i], t_ey[i], _ = propagar_f(t[i])

valor_buscado = t[t_ey==min(t_ey)]

print('El valor de x_a donde el error en y es mínimo es {}' .format(*valor_buscado))
print('El promedio de los datos en x es {}'.format(x.mean()))#El valor buscado coincide con el promedio de los datos en x!!


plt.plot(x,y, '.')
plt.errorbar(x, y, yerr = y_error, fmt = 'r.')

plt.fill_between(t, t_y-t_ey, t_y+t_ey,
    alpha=0.2, edgecolor='#1B2ACC', facecolor='#089FFF',
    linewidth=4, linestyle='dashdot', antialiased=True)
t = np.linspace(0,5,100)
plt.plot(t,f(t, *popt))
plt.xlim(0,5)
plt.ylim(0,8)


#%% Ejercicio c
# FALTA: discuta por qué ésta es claramente errónea



t = np.linspace(0,10,1000)


t_y = np.zeros(len(t))
t_ey_mal = np.zeros(len(t))
for i in range (len(t)):
    t_y[i], _, t_ey_mal[i] = propagar_f(t[i])



valor_buscado = t[t_ey_mal==min(t_ey)]

print('El valor de x_a donde el error en y es mínimo es {}' .format(*valor_buscado))


plt.plot(x,y, '.')
plt.errorbar(x, y, yerr = y_error, fmt = 'r.')

plt.fill_between(t, t_y-t_ey_mal, t_y+t_ey_mal,
    alpha=0.2, edgecolor='#1B2ACC', facecolor='#089FFF',
    linewidth=4, linestyle='dashdot', antialiased=True)
t = np.linspace(0,5,100)
plt.plot(t,f(t, *popt))
plt.xlim(0,5)
plt.ylim(0,8)
#%% Ejercicio d

sigma = 0.3
y_error = sigma * np.ones(len(x))

a1_0 = 1.452
a2_0 = 0.799
 

n = int(1e4)
total = np.zeros(n)
for k in range(n):
    g = np.zeros(len(x))
    for i in range(len(x)):
        g[i] = random.gauss(a2_0 * x[i] + a1_0, sigma)
    
    popt, pcov = curve_fit(f, x, g, sigma = y_error, 
                           absolute_sigma=True)
    
    x_a = 0.5
    a_1 = popt[1]
    a_2 = popt[0]    
    var_a2 = pcov[0,0]
    var_a1 = pcov[1,1]
    cov_a = pcov[0,1]
    
    y_a = a_1 + a_2 * x_a
#    var_ya = x_a**2 * var_a2 + var_a1 + 2 * x_a * cov_a
    #mal_var_ya = x_a**2 * var_a2 + var_a1 
    total[k] = y_a        
plt.hist(total, density=True)
t = np.linspace(0,6,1000)
t_y = norm.pdf(t,loc = 1.8575, scale = 0.57918)
plt.plot(t,t_y)
