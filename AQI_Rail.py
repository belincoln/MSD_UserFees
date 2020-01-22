# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 15:03:34 2020

@author: belincoln
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os


#%% Load Data

# first set the working directory. This code will be changed based on the relative location of the data files 
# on the local drive of the computer executing the command. 
os.chdir('C:\\Users\\belincoln\\Documents\\! CBP\\!User Fees\\!! Goal 1 Dashboards')

# Remember to set the working directory to whatever folder you have saved the zip file to. 
# Works well for Jupyter Notebooks, can be configured in IDE using file explorer. 
data = pd.read_excel(os.path.join('Source Emails & Source Files','Files','Collections','AQI','AQI Collections.xlsx'),'Rail')
#%%

Rail_Collections = data[['Fiscal year', 'Fiscal period','Document Date', '$']]
Rail_Collections = Rail_Collections.rename(columns={'$': 'Collections'})
# All collection totlas came in as negative?
Rail_Collections['Collections'] = (Rail_Collections['Collections'] * -1).astype(float)

#%%

# Sum on Remittance Period
Rail_Collections = Rail_Collections.groupby(['Fiscal year','Fiscal period']).sum()

#%%

# Convert Fiscal Year and Month to Calendar Year and Month
fy = []
month = []
for i in Rail_Collections.index:
    fy.append(i[0])
    month.append(i[1])
    
Rail_Collections['Month'] = month 
Rail_Collections['Month'] = Rail_Collections['Month'] + 9
Rail_Collections['Month'][Rail_Collections['Month'] >12] = Rail_Collections['Month'][Rail_Collections['Month'] >12] - 12

Rail_Collections['Fiscal year'] = fy

cy = Rail_Collections['Fiscal year'][Rail_Collections['Month'] > 9] -1
cy = cy.append(Rail_Collections['Fiscal year'][Rail_Collections['Month'] < 10])
Rail_Collections['Calendar Year'] = cy

#%%
Rail_Collections.set_index(Rail_Collections['Month'].apply(str) + "/1/" + Rail_Collections['Calendar Year'].apply(str), inplace = True)

#%% Read and Clean Workload Data
workload = pd.read_excel(os.path.join('Source Emails','Files','Workload','COBRA','fy13-18 stats by_Month National.xlsx'))
# Select only rail rows
workload = workload.iloc[2:4,:]
#select data we want to work with
workload.iloc[0,3:] = workload.iloc[1,3:]
workload = workload.iloc[0,:]
workload.drop(['Line','Total', 'Data Id', 'Data Elements  - National'],inplace = True)
workload = workload.to_frame()
workload = workload.astype(float)
workload.rename(columns = {2:'Trains'}, inplace =True)


#%%
# merge workload and collections

workload_collections = Rail_Collections.merge(workload,left_index = True,right_index = True)
#%% Identify periods of fee rate change

workload_collections['old fee rate'] = 1
workload_collections.loc['1/1/2016':,'old fee rate'] = 0

#%%
# Shift collections to match with worklaod

workload_collections['Collections_Shift'] = workload_collections['Collections'].shift(-3)
workload_collections.drop(workload_collections.tail(3).index, inplace = True)

#%% Show Scatter

colors = np.where(workload_collections['old fee rate']==1,'b','r')
plt.scatter(workload_collections['Trains'], workload_collections['Collections_Shift'], c=colors)
plt.xlabel('Trains')
plt.ylabel('Collections')
plt.show()

#%%

# Perform R^2 Calculation to determine the % of change of collections that is explained by variation in workload
import statsmodels.api as sm

# set x and ys 
r1_c_array = workload_collections['Collections_Shift'][workload_collections['old fee rate']==1]
r2_c_array = workload_collections['Collections_Shift'][workload_collections['old fee rate']==0]

r1_w_array = workload_collections['Trains'][workload_collections['old fee rate']==1]
r2_w_array = workload_collections['Trains'][workload_collections['old fee rate']==0]

# run linear regression on workload and collections to get linear coefficent. 
X = r1_w_array.astype(float)
X = sm.add_constant(X)
model = sm.OLS(r1_c_array,X)
results = model.fit()

X2 = r2_w_array.astype(float)
X2 = sm.add_constant(X2)
model2 = sm.OLS(r2_c_array,X2)
results2 = model2.fit()

#%%
workload_collections['Remittance Period'] = workload_collections['Fiscal year']
workload_collections.drop(['Collections','Fiscal year', 'old fee rate','Month'], axis = 1, inplace = True)
workload_collections = workload_collections.rename(columns={'Collections_Shift': 'Collections','Trains':'Workload'})
workload_collections['Fee'] = 'AQI'

#%%
df1 = workload_collections.loc[:'1/1/2016',:]
df1['Environment'] = 'Rail_Pre_Rate_Change'
df1['R^2'] = results.rsquared
df1['Regression Coefficent'] = results.params.iloc[1]
df1['Expected Collections'] = results.params.iloc[0] + results.params.iloc[1] * df1['Workload']

# add coefficents and constants for confidence interval mapping
df1['lower conf_int constant'] = results.conf_int(alpha=.05, cols= None).iloc[0,0]
df1['lower conf_int coefficent'] = results.conf_int(alpha=.05, cols= None).iloc[1,0]

df1['upper conf_int constant'] = results.conf_int(alpha=.05, cols= None).iloc[0,1]
df1['upper conf_int coefficent'] = results.conf_int(alpha=.05, cols= None).iloc[1,1]

# Export as CSV for Power BI Visualization
df1.to_csv(os.path.join('Power_BI_Data_Files','Pre_Rate_Change_AQI_Rail_Workload_Collections_Period.csv'))

#%%
#%%
df2 = workload_collections.loc['1/1/2016':,:]
df2['Environment'] = 'Rail_Post_Rate_Change'
df2['R^2'] = results2.rsquared
df2['Regression Coefficent'] = results2.params.iloc[1]
df2['Expected Collections'] = results2.params.iloc[0] + results2.params.iloc[1] * df2['Workload']

# add coefficents and constants for confidence interval mapping
df2['lower conf_int constant'] = results2.conf_int(alpha=.05, cols= None).iloc[0,0]
df2['lower conf_int coefficent'] = results2.conf_int(alpha=.05, cols= None).iloc[1,0]

df2['upper conf_int constant'] = results2.conf_int(alpha=.05, cols= None).iloc[0,1]
df2['upper conf_int coefficent'] = results2.conf_int(alpha=.05, cols= None).iloc[1,1]

# Export as CSV for Power BI Visualization
df2.to_csv(os.path.join('Power_BI_Data_Files','Post_Rate_Change_AQI_Rail_Workload_Collections_Period.csv'))
# =============================================================================
# # this is the linear coefficent
# collection_per_workload = results.params.iloc[0]
# # Multiply linear coefficent with workload to graph a line of best fit. 
# workload_collections['Expected Collections'] = collection_per_workload * workload_collections['Total Workload']
# # Add regression coefficent and R^2 values for notecards
# workload_collections['Regression Coefficent'] = results.params.iloc[0]
# workload_collections['R^2'] = results.rsquared
# =============================================================================

