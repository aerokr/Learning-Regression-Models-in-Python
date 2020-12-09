import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as mp # plotting 
import statsmodels.api as sm 
import statsmodels.stats as sms #OLS model

# Creating the dataframe for the houseing data
Housedb = pd.read_csv('train.csv')
#looking at the first few rows
Housedb.head()

#pulling out a couple of variables to use in a simple model
LotFrontage=Housedb[['LotFrontage']]
LotArea = Housedb[['LotArea']]
SalePrice = Housedb[['SalePrice']]
#taking a quick look at the graphs of the data 
mp.scatter(np.log(LotArea),np.log(SalePrice))

mp.show()

mp.scatter(np.log(LotFrontage), np.log(SalePrice))

mp.show()

mp.scatter(np.log(LotFrontage), np.log(LotArea))

mp.show()

#OLS regression itself
Y = np.log(SalePrice)
X = np.log(Housedb[['LotArea','LotFrontage']])
X = sm.add_constant(X)
model = sm.OLS(Y,X,missing='drop')
results = model.fit()
print(results.summary(results))

#testing for heteroskedasticity
sms.diagnostic.het_breuschpagan(results.resid,X.dropna())