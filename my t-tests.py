# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 12:51:39 2019

@author: Jack
"""


def find_Ttest(y1,y2,s1,s2,N1,N2):
    nom=0
    nom = y1-y2
    d1=(s1*s1)/N1
    d2=(s2*s2)/N2

    denom = (d1+d2)**.5
    return  nom / denom

y1=5
y2=8
s1=1
s2=3
N1=200
N2=500
print(find_Ttest(y1, y2, s1, s2, N1, N2))


y1=1090
y2=999
s1=400
s2=30
N1=900
N2=100
print(find_Ttest(y1, y2, s1, s2, N1, N2))

y1=45
y2=40
s1=45
s2=40
N1=2000
N2=2000
print(find_Ttest(y1, y2, s1, s2, N1, N2))
