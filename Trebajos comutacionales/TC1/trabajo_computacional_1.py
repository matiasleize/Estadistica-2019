import numpy as np
from scipy.stats import binom, poisson
from matplotlib import pyplot as plt
import pandas as pd
from matplotlib.ticker import StrMethodFormatter

#%%
def plot_histogram(data, bins, ticks=5, xlabel='X', ylabel='Histograma',
                   color='', ecolor='' ,ax = None, alpha = None,
                   density = True):
    N = len(data)
    num_bins = len(bins)
    xmin = min(bins)
    xmax = max(bins)

    if ax is None:
        ax = plt.gca() #ax es un eje del plot
        
    if alpha is None:
        alpha = 1

    hist, _ = np.histogram(data, bins=bins, density=density)

    df = pd.DataFrame({'histograma': hist, 'error': np.sqrt(hist/N)})

    #Para definir el ax de matplotlib usamos el data frame de pandas
    ax = df.plot.bar(y='histograma', yerr='error', width=1, edgecolor='k', ecolor='red', capsize=3, label=ylabel, 
                     ax=ax, color ='blue', alpha = alpha)
    
    plt.grid(True, axis = 'y')
#    plt.xticks(np.linspace(0, num_bins - 2, ticks), np.linspace(xmin, xmax, ticks), rotation='horizontal');
    plt.xlabel(xlabel);
    plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
    return ax

#%% Ejercicio 1

#Inciso a
    
def prob_uniforme(p, N=10e6):
    q = (1-p)
    n = np.random.uniform(0,1,int(N))
    n_p = n[n<=p]
    n_q = n[n<=q]
    prob_p = len(n_p)/len(n)
    prob_q = len(n_q)/len(n)
    return prob_p, prob_q


t = np.linspace(0,1,1000)

plt.figure(figsize=(9.7,6))
plt.title('Longitud del intervalo vs Probabilidad de un número aleatorio')
plt.xlabel('Longitud')
plt.ylabel('Probabilidad')
plt.l
plt.grid(True)

#Falta hacer varios subplots
for N in [1000,10000,100000]:
    ps = np.zeros(len(t))
    qs = np.zeros(len(t))
    for i in range(len(t)):
        ps[i], qs[i] = prob_uniforme(t[i],N)
    plt.plot(t,ps)
#%%
##Inciso b
def exp_bernoulli(N,p):
    '''Esta función toma como parametros el número de experimentos
    realizados y la probabilidad de tener éxito en cada uno de ellos.
    Devuelve el numero de exitos que hubo en el total de experimentos'''
    n = np.random.uniform(0,1,N)
    n_p = n[n<=p] 
    a = len(n_p) #La catidad de exitos
    return a, n_p

#A modo de ejemplo:
e_1 = exp_bernoulli(int(10e6),0.5)
print(e_1)


#%%
##Problema 2
e_2= exp_bernoulli(15, 0.75)

#Repito mil veces
tiradas = np.zeros(1000)
for i in range(0, 1000):
    tiradas[i],_ = exp_bernoulli(15, 0.75)

#La distribución teórica es la binomial
X = binom(15,0.75)

plt.figure(figsize=(9.7,6))
plt.title('Ejercicio 2')
plot_histogram(tiradas,np.arange(np.sqrt(len(tiradas))),alpha=0.6)

t = np.arange(0,200)
#plt.plot(t, X.pmf(t), 'm.-',ms=15, linewidth=4) #Puedo usar la PDF de la distribución así
plt.stem(t, X.pmf(t), 'k.-') #Puedo usar la PDF de la distribución así
plt.show()
#%%
#Ejercico 3
#Datos del problema
I = 15
m = 1000 #Lo que divido mi delta t
Dt = 1
dt = Dt/m

# Para un dt
n_exp = 1000
p=I*dt
n_dt = m #n de la binomial

data = np.zeros(n_exp) #Hago mil experimentos
for j in range(n_exp):  
    data[j],_ = exp_bernoulli(m,p) #Hago m tiradas de moneda donde puedo tener un exito o un fracaso.

plt.figure(figsize=(9.7,6))
plt.title('Ejercicio 3')
plot_histogram(data,bins=np.arange(np.sqrt(len(data))))

Y = poisson(15) #Distribución de Poisson con mu = 15
t = np.arange(0,200)
plt.plot(t, Y.pmf(t), 'm.-', linewidth=4, ms=15) #Puedo usar la PDF de la distribución así
plt.show()

#%%
#Ejercicio 4

#Datos del problema
I = 15
m = 1000 #Lo que divido mi delta t
Dt = 1
dt = Dt/m

# Para un dt
n_exp = 1000
p=I*dt
n_dt = m #n de la binomial

ev_detectados = np.zeros(n_exp)
data = np.zeros(n_exp) #Hago mil experimentos
for j in range(n_exp):  
    data[j],_ = exp_bernoulli(m,p) #Hago m tiradas de moneda donde puedo tener un exito o un fracaso.
    n_binom = int(data[j])
    ev_detectados[j],_ = exp_bernoulli(n_binom,0.75)
    


plt.figure(figsize=(9.7,6))
plt.title('Ejercicio 4')
ax = plot_histogram(ev_detectados, np.arange(np.sqrt(len(ev_detectados))))

Y = poisson(0.75*15) #Distribución de Poisson con mu = 15 por la eficiencia del detector
t = np.arange(0,200)
plt.plot(t, Y.pmf(t), 'g.-',ms=15, linewidth=4) #Puedo usar la PDF de la distribución así

plt.show()

#%%
#Ejercicio 5

#Datos del problema
I = 15
m = 1000 #Lo que divido mi delta t
Dt = 1
dt = Dt/m

# Para un dt
n_exp = 5000
p=I*dt
n_dt = m #n de la binomial

eficiencia_detector = 0.75

prob_total = p*eficiencia_detector #Son probabilidades independientes.

ev_detectados = np.zeros(n_exp) #Hago mil experimentos
for j in range(n_exp):  
    ev_detectados[j],_ = exp_bernoulli(m,p*0.75)


plt.figure(figsize=(9.7,6))
plt.title('Ejercicio 5')
ax = plot_histogram(ev_detectados, np.arange(np.sqrt(len(ev_detectados))))

Y = poisson(0.75*15) #Distribución de Poisson con mu = 15 por la eficiencia del detector
t = np.arange(np.sqrt(len(ev_detectados)))
#plt.xlim([0,100]) #Si ponemos muchos puntos el t se va muy lejos
plt.plot(t, Y.pmf(t), 'k.-',ms=15, linewidth=4) #Puedo usar la PDF de la distribución así
plt.show()


