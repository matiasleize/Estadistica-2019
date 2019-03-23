#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 10 02:11:16 2019

@author: matias
"""

import scipy
import numpy as np

result = scipy.special.erf(195) - scipy.special.erf(0.000001)
print(result)

