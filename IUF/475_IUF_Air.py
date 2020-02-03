# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 13:05:35 2019

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
collections_475 = pd.read_excel(os.path.join('Source Emails & Source Files','Files','Collections','IUF_Air','Collections cc475 - FY13 - FY18.xls'))

#%% Clean Data


# delete columns and rows that contain only na
collections_475 = collections_475.dropna(axis=0, how = 'all')
collections_475 = collections_475.dropna(axis=1, how = 'all')

# delete first two rows
collections_475 = collections_475.iloc[2:,:]
#%%
#Fill down the company column
collections_475.iloc[:,0] = collections_475.iloc[:,0].ffill(axis = 0)
# reset index
collections_475 = collections_475.reset_index(drop=True)

#%% drop columns
collections_475 = collections_475.iloc[:,[0,1,7,8,9,11]]

# Give column headers
#%%
collections_475.columns = collections_475.iloc[0]

#rename columns
collections_475.columns.values[0] = 'Company'
collections_475.columns.values[1] = 'Period'
collections_475.columns.values[2] = 'Class Code'

# delete first two rows
collections_475 = collections_475.iloc[2:,:]
#%%
# Remove excess rows 
collections_475 = collections_475.dropna(subset = ['Class Code'])

# Create total collections column
collections_475['Collections'] = collections_475.iloc[:,-3:].sum(axis = 1)
#delete excess columns
collections_475 = collections_475.loc[:,['Collections','Period']]

#%%
# Sum on Remittance Period
collections_475 = collections_475.groupby(collections_475['Period']).sum()
# Remove audit payments
collections_475 = collections_475[~collections_475.index.str.contains("\*")]

# Add an additional column that shows remittance period (independent of year)
collections_475['Remittance Period'] = collections_475.index.str.split('20').str[0]


# Create Calendar Year Column
collections_475['Calendar Year'] = collections_475.index.str.split(')').str[1]
# Turn Years into integers
collections_475['Calendar Year'] = collections_475['Calendar Year'].astype(int)

# Filter on years not a part of analysis
years = [2012,2013,2014,2015,2016,2017,2018]
collections_475 = collections_475[collections_475['Calendar Year'].isin(years)]

#%%

workload = pd.read_excel(os.path.join('Source Emails & Source Files','Files','Workload','IUF_Air','FY09-fy18-passenger data air and cruise.xlsx'))

#Clean Workload Df
#remove cruise data
workload = workload.iloc[0:6,:]
#select data we want to work with
workload2 = workload.iloc[::2,:2]
workload3 = workload.iloc[[1,3,5],2:]
#reset indices for merge
workload2.reset_index(drop = True, inplace = True)
workload3.reset_index(drop = True, inplace = True)

# concat dataframes to get cleaned workload df
workload = pd.concat([workload2, workload3], axis=1, sort=False)
#sum Data Element and Workload ID columns into one descriptor
workload['Workload Element'] = workload.iloc[:,0]+': '+workload.iloc[:,1]
workload = workload.iloc[:,::-1]
workload = workload.iloc[:,:-2]
#%%
workload = workload.transpose()
#promote first row to column headers and drop first row
workload.columns = workload.iloc[0]
workload = workload.iloc[1:]

# Create Calendar Year and Month Columns
workload['Month'] = workload.index.str.split('/').str[0]
workload['Calendar Year'] = workload.index.str.split('/').str[2] + " "
print(workload['Calendar Year'])
# Filter on years not a part of analysis
years = ['2012 ','2013 ','2014 ','2015 ','2016 ','2017 ','2018 ']
workload = workload[workload['Calendar Year'].isin(years)]

#%%
# Build out Remittance Period Columns
conditions = [(workload['Month'] == '1'), (workload['Month'] == '2'), (workload['Month'] == '3'), (workload['Month'] == '4'), (workload['Month'] == '5'), (workload['Month'] == '6'),(workload['Month'] == '7'),(workload['Month'] == '8'),(workload['Month'] == '9'),(workload['Month'] == '10'),(workload['Month'] == '11'),(workload['Month'] == '12')] 
choices = ['Qtr 01 (Jan-Mar)','Qtr 01 (Jan-Mar)','Qtr 01 (Jan-Mar)','Qtr 02 (Apr-Jun)','Qtr 02 (Apr-Jun)','Qtr 02 (Apr-Jun)','Qtr 03a (Jul-Aug)','Qtr 03a (Jul-Aug)', 'Qtr 03b (Sept)','Qtr 04 (Oct-Dec)','Qtr 04 (Oct-Dec)','Qtr 04 (Oct-Dec)']
workload['Period'] = np.select(conditions, choices, default='error')


#%%

# Match Period Column to Collections
workload['Period'] = workload['Period'] + ' ' + workload['Calendar Year']
# Set index to Remittance Period
workload = workload.set_index('Period')
# drop unnecssary columns
workload = workload.drop(['Calendar Year','Month'], axis = 1)
# Sum on Remittance Period
workload = workload.groupby(workload.index).sum()



#%%
#remove non FY2013-2018 data
workload = workload.iloc[1:,:]
collections_475 = collections_475.iloc[1:-1,:]

#%%
#remove non FY2013-2018 data
searchfor = ['Qtr 02 \(Apr-Jun\) 2012', 'Qtr 03a \(Jul-Aug\) 2012','Qtr 03b \(Sept\) 2012']
collections_475 = collections_475[~collections_475.index.str.contains('|'.join(searchfor))]
workload = workload[~workload.index.str.contains('|'.join(searchfor))]

#Add sum of workload columns
workload['Workload'] = workload.sum(axis = 1)

#%%
workload_collections = pd.merge(workload,collections_475,how = 'inner', left_index = True, right_index = True)
#%%
# create a sum of all relevant workload metrics
#workload_collections['Workload'] = workload_collections.iloc[:,:3].sum(axis = 1)
#test for correlation
corr = workload_collections.iloc[:,:-1].corr()
# Select Workload metric with highest correlation
#workload_collections = workload_collections.iloc[:,2:-1]
#%% Scatterplots of workload metrics and collection
for i in range(len(workload.columns)):
    plt.scatter(workload_collections.iloc[:,i],workload_collections['Collections'])
    plt.xlabel(workload.columns[i])
    plt.ylabel('Total Collections')
    plt.show()

#%%
import statsmodels.api as sm

# run linear regression on workload and collections to get linear coefficent. 
X = workload_collections['Workload']
X = sm.add_constant(X)
model = sm.OLS(workload_collections['Collections'],X)
results = model.fit()

# this is the Regression coefficent
collection_per_workload = results.params.iloc[1]

# Multiply linear coefficent with workload to graph a line of best fit. 
workload_collections['Expected Collections'] = collection_per_workload * workload_collections['Workload']
# Add regression coefficent and R^2 values for notecards
workload_collections['Regression Coefficent'] = results.params.iloc[1]
workload_collections['R^2'] = results.rsquared
workload_collections = workload_collections.iloc[:,3:]

#%%

# Show final Scatter Plot with Line of best fit. 
plt.scatter(workload_collections['Workload'], workload_collections['Collections'])
plt.xlabel('Workload')
plt.ylabel('Collections')
plt.plot(workload_collections['Workload'], results.params.iloc[0] + collection_per_workload * workload_collections['Workload'])
plt.show()

#%%
#Add Power BI column for filter
workload_collections['Fee'] = 'IUF'
workload_collections['Environment'] = 'Air'

# add coefficents and constants for confidence interval mapping
workload_collections['lower conf_int constant'] = results.conf_int(alpha=.05, cols= None).iloc[0,0]
workload_collections['lower conf_int coefficent'] = results.conf_int(alpha=.05, cols= None).iloc[1,0]

workload_collections['upper conf_int constant'] = results.conf_int(alpha=.05, cols= None).iloc[0,1]
workload_collections['upper conf_int coefficent'] = results.conf_int(alpha=.05, cols= None).iloc[1,1]

#Write csv file for power bi
workload_collections.to_csv(os.path.join('Power_BI_Data_Files','475_IUF_Air_Workload_Collections_Period.csv'))

#%%

# =============================================================================
# from scipy.stats import t
# 
# #Set confidence interval
# confidence = 0.95
# data = workload_collections[['Collections','Expected Collections','Workload']]
# data['squared_diff'] = (data['Collections'] - data['Expected Collections']) ** 2
# temp = np.sum(data['squared_diff'])
# temp = temp / (len(data)-2)
# se = temp ** .5
# #%%
# 
# denom = np.sum(data['Collections']**2)
# denom = denom - (np.sum(data['Collections'])**2 / len(data))
# num = (data['Workload'] - np.mean(data['Workload'])) ** 2
# temp = 1/len(data) + num/denom
# temp = temp ** .5
# 
# data['Upper Confidence Interval'] = data['Expected Collections'] + t.ppf(1-(1-confidence)/2,(len(data)-2))*se*temp
# data['Lower Confidence Interval'] = data['Expected Collections'] - t.ppf(1-(1-confidence)/2,(len(data)-2))*se*temp
# 
# 
# #%%
# #plt.scatter(workload_collections['Workload'],data['Collections'])
# plt.plot(workload_collections['Workload'],data['Expected Collections'])
# plt.scatter(workload_collections['Workload'],data['Collections'])
# plt.plot(workload_collections['Workload'],data['Upper Confidence Interval'])
# plt.plot(workload_collections['Workload'],data['Lower Confidence Interval'])
# plt.xlabel('Workload')
# plt.ylabel('Total Collections')
# plt.show()
# 
# =============================================================================
#%%




