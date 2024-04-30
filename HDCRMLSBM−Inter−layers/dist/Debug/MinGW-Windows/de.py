__author__ = 'hamed'
import sys
import math

# import xlwt
import os
import glob
from math import log
import networkx as nx

import random

# import matplotlib.pyplot as plt
# import pandas as pd
# from scipy import stats
# import scipy.sparse as sp

import csv
import numpy as np
import numpy.random as rand

jk=np.zeros(100)
summ=0


for i in range(0,100):
    jk[i] = np.genfromtxt('result'+str(i)+'.csv',  delimiter=',')
    summ = summ+jk[i]



summ=summ/100
print(summ)
np.savetxt('result.csv', jk, fmt='%f', delimiter=',')
#np.savetxt('result.csv', summ, fmt='%d', delimiter=',')

