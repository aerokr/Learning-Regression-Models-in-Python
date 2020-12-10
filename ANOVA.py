import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as mp # plotting 
import statsmodels.formula.api as sm 
import statsmodels.stats as sms #OLS model

# Creating the dataframe for the houseing data
Housedb = pd.read_csv('train.csv')
#looking at the first few rows
Housedb.head()

#Gather
Deeds = Housedb.loc[Housedb['SaleType'].isin(['WD','New'])]
print(Deeds)

House_lm = sm.ols(formula="SalePrice ~ C(SaleType)",data=Deeds).fit()
table = sms.anova.anova_lm(House_lm,typ=1)
print(table)