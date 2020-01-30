# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 13:33:43 2019

@author: belincoln
"""

import pandas as pd
import glob

#%%

# So I tried reading all 7 csvs into one data frame using some code I found on stack exchange. 
# I got a Memory Error. I think its to large. 
path = r'C:\Users\belincoln\Documents\ML\Budget_Initiative' # use your path
all_files = glob.glob(path + "\*.csv")

li = []

for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    li.append(df)

frame = pd.concat(li, axis=0, ignore_index=True)

#%%
df = pd.read_csv('2019_all_Contracts_Full_20191211_6.csv', header=0)



#%%

df2 = df.iloc[:,:79]
df3 = df2[df['federal_action_obligation']<-1000]
#%%
df2.to_excel('2019_contract_awards.xlsx')
df3.to_excel('significant_deobligations.xlsx')

