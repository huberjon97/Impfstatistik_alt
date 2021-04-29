# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 17:06:22 2021

@author: aidac
"""

from datetime import date
import matplotlib.pyplot as plt
today=date.today()
d4 = today.strftime("%d_%m_%Y")
print("d4 =", d4)
fig1=plt.figure(1)
plt.savefig('test_'+d4+'.pdf')