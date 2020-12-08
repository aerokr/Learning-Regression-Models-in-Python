import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as mp # plotting 
import statsmodels.api as sm 
import statsmodels.stats as sms #OLS model

# Creating the dataframe for the houseing data
Housedb = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/housetrain.csv')
#looking at the first few rows
Housedb.head()

#
Deeds = Housedb.loc[Housedb['SaleType'].isin(['WD','CWD','VWD','New'])]
print(Deeds)

Y = Deeds[['SalePrice']]
X = Deeds[['SaleType']]
House_lm = sm.OLS(Y,X,missing='drop').fit()