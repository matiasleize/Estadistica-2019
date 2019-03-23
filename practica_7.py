#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:46:57 2019

@author: matias
"""

#%%
import numpy as np
from scipy.stats import beta
#%% EJercicio 8

x = beta(51,12292)

a = x.ppf(0.05)

b = x.ppf(0.9+0.05)
