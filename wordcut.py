# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 15:39:55 2018

@author: zhanglisama    jxufe
"""

import jieba 
import jieba.analyse
import re 

f=open('test1.txt')
list1=f.readlines()
codes= {}

for i in list1:
    code=jieba.cut(i)
    code=' '.join(code)
    code=re.sub('[£¬¡£\n£¨£©£¿£»¡°¡±]',' ',code)
    code=code.split(' ')
    
    for j in code:
        
        if j not in codes:
            codes[j]=1
        else:
            codes[j]+=1


for key in codes:
    print(key,codes[key])         
