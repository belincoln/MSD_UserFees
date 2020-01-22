# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 11:21:55 2019

@author: belincoln
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

#%%

# first set the working directory. This code will be changed based on the relative location of the data files 
# on the local drive of the computer executing the command. 
os.chdir('C:\\Users\\belincoln\\Documents\\! CBP\\!User Fees\\!! Goal 1 Dashboards')

# read in Collection Data
#collections_477 = pd.read_excel(r'C:\Users\belincoln\Documents\! CBP\!User Fees\ML_Applications\CC477 collections FY13-FY18.xlsx')
collections_477 = pd.read_excel(os.path.join('Source Emails & Source Files','Files','Collections','IUF_Cruise','CC477 collections FY13-FY18.xlsx'))

# format df, add header
collections_477 = collections_477.drop([0,1,2], axis=0) # 'axis=1' specifies columns; 'axis=0' is rows
header = collections_477.iloc[0]
collections_477 = collections_477[1:]
collections_477.columns = header

#%%
column_name = collections_477.iloc[0]['Class Code']

# Turn interest payments, principal payments, and late penalties into floats
collections_477.iloc[:,-3:].apply(pd.to_numeric)
#%%%
# Sum all columns
collections_477[column_name] = collections_477.iloc[:,-3] + collections_477.iloc[:,-2] + collections_477.iloc[:,-1]

#drop remaining columns
collections_477 = collections_477.loc[:, collections_477.columns.intersection([column_name,'Period'])]

#set index to period
collections_477 = collections_477.set_index('Period')

# Sum on Remittance Period
collections_477 = collections_477.groupby(collections_477.index).sum()

#%%

#collections_476 = pd.read_excel(r'C:\Users\belincoln\Documents\! CBP\!User Fees\ML_Applications\Collections cc476 for FY13-FY18.xlsx')
collections_476 = pd.read_excel(os.path.join('Source Emails','Files','Collections','IUF_Cruise','Collections cc476 for FY13-FY18.xlsx'))
# format df, add header
collections_476 = collections_476.drop([0,1,2], axis=0) # 'axis=1' specifies columns; 'axis=0' is rows
header = collections_476.iloc[0]
collections_476 = collections_476[1:]
collections_476.columns = header


column_name = collections_476.iloc[0]['Class Code']
# Turn interest payments, principal payments, and late penalties into floats
collections_476.iloc[:,-3:].apply(pd.to_numeric)

# Sum all columns
collections_476[column_name] = collections_476.iloc[:,-3] + collections_476.iloc[:,-2] + collections_476.iloc[:,-1]

#drop remaining columns
collections_476 = collections_476.loc[:, collections_476.columns.intersection([column_name,'Period'])]

#set index to period
collections_476 = collections_476.set_index('Period')

# Sum based on Remittance Period
collections_476 = collections_476.groupby(collections_476.index).sum()


#%%
#Merge both collection codes
collections = collections_476.merge(collections_477, left_index=True, right_index=True)
collections['Collections'] = collections.iloc[:,0] + collections.iloc[:,1]

#%%
#workload = pd.read_excel(r'C:\Users\belincoln\Documents\! CBP\!User Fees\ML_Applications\Workload_PPAE.xlsx')
workload = pd.read_excel(os.path.join('Source Emails','Files','Workload','IUF_Cruise','Workload_PPAE.xlsx'))
# Select workload Month and Year in order to sum on Remittance Period
workload['Year'] = workload['Date'].str[-4:]
workload['Month'] = workload['Date'].str.split('/').str[0]

#%%
# Create Conditional Column (Group workload metrics by remittance period). Jan - March -> Q1 ect...
conditions = [(workload['Month'] == '1'), (workload['Month'] == '2'), (workload['Month'] == '3'), (workload['Month'] == '4'), (workload['Month'] == '5'), (workload['Month'] == '6'),(workload['Month'] == '7'),(workload['Month'] == '8'),(workload['Month'] == '9'),(workload['Month'] == '10'),(workload['Month'] == '11'),(workload['Month'] == '12')] 
choices = ['Qtr 01 (Jan-Mar)','Qtr 01 (Jan-Mar)','Qtr 01 (Jan-Mar)','Qtr 02 (Apr-Jun)','Qtr 02 (Apr-Jun)','Qtr 02 (Apr-Jun)','Qtr 03a (Jul-Aug)','Qtr 03a (Jul-Aug)', 'Qtr 03b (Sept)','Qtr 04 (Oct-Dec)','Qtr 04 (Oct-Dec)','Qtr 04 (Oct-Dec)']
workload['Period'] = np.select(conditions, choices, default='error')

# Match Period Column to Collections
workload['Period'] = workload['Period'] + ' ' + workload['Year']
# Set index to Remittance Period
workload = workload.set_index('Period')
# drop unnecssary columns
workload = workload.drop(['Year','Month','Date'], axis = 1)
# Sum on Remittance Period
workload = workload.groupby(workload.index).sum()

#%%
# Merge Workload and Collection Data
workload_collections = pd.merge(workload,collections,how = 'inner', left_index = True, right_index = True)
# view Correlation Matrix to determine which Workload Metrics have a strong relationship with collections
# Use th variable explorer. 
corr = workload_collections.corr()
print(corr)

#%%
# Show scatter plots of workload and total collections
for i in range(len(workload.columns)):
    plt.scatter(workload_collections.iloc[:,i],workload_collections.iloc[:,-1])
    plt.xlabel(workload.columns[i])
    plt.ylabel('Collections')
    plt.show()
#for i in range(len(workload_collections.columns))

#%%
# Drop all columns, but collection and workload metric columns that have the highest correlation
workload_collections = workload_collections.loc[:, workload_collections.columns.intersection(['Foreign Psngrs (Cruise vessels)','Collections'])]
# rename workload column
workload_collections.rename(columns = {'Foreign Psngrs (Cruise vessels)': 'Workload'}, inplace = True)
#%%
# Show scatter plot of workload and collections. Shows strong linear relationship. 
plt.scatter(workload_collections.iloc[:,0], workload_collections.iloc[:,1])
plt.xlabel('Workload')
plt.ylabel('Collections')
plt.show()

#%%
import statsmodels.api as sm

# run linear regression on workload and collections to get linear coefficent. 
X = workload_collections['Workload']
X = sm.add_constant(X)
model = sm.OLS(workload_collections['Collections'],X)
results = model.fit()

# this is the linear coefficent
collection_per_workload = results.params.iloc[1]
const = results.params.iloc[0]
# Multiply linear coefficent with workload to graph a line of best fit. 
workload_collections['Expected Collections'] = collection_per_workload * workload_collections.iloc[:,0]
# Add regression coefficent and R^2 values for Power BI notecards
workload_collections['Regression Coefficent'] = results.params.iloc[1]
workload_collections['R^2'] = results.rsquared

# add coefficents and constants for confidence interval mapping
workload_collections['lower conf_int constant'] = results.conf_int(alpha=.05, cols= None).iloc[0,0]
workload_collections['lower conf_int coefficent'] = results.conf_int(alpha=.05, cols= None).iloc[1,0]

workload_collections['upper conf_int constant'] = results.conf_int(alpha=.05, cols= None).iloc[0,1]
workload_collections['upper conf_int coefficent'] = results.conf_int(alpha=.05, cols= None).iloc[1,1]


#%%

# Add an additional column that shows remittance period (independent of year)
workload_collections['Remittance Period'] = workload_collections.index.str.split('20').str[0]

# Show final Scatter Plot with Line of best fit. 
plt.scatter(workload_collections.iloc[:,0], workload_collections.iloc[:,1])
plt.xlabel('Foreign Passengers (Cruise)')
plt.ylabel('Collections')
plt.plot(workload_collections.iloc[:,0], const + collection_per_workload * workload_collections.iloc[:,0])
plt.plot(workload_collections.iloc[:,0], workload_collections['lower conf_int constant'][0] + workload_collections['lower conf_int coefficent'][0]*workload_collections.iloc[:,0])
plt.plot(workload_collections.iloc[:,0], workload_collections['upper conf_int constant'][0] + workload_collections['upper conf_int coefficent'][0]*workload_collections.iloc[:,0])
plt.show()

#%%
# Add back Calendar Year Column
workload_collections['Calendar Year'] = workload_collections.index.str.split('\)').str[1]
    #Add Power BI column for filter
workload_collections['Fee'] = 'IUF'
workload_collections['Environment'] = 'Cruise & Ferry'
# Export as CSV for Power BI Visualization
workload_collections.to_csv(os.path.join('Power_BI_Data_Files','476_477_IUF_Cruise_Ferry_Workload_Collections_Period.csv'))

