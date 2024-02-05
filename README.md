# Boston Housing Dataset Analysis 
Project Description
This project aims to analyze the Boston Housing dataset to gain insights into housing prices and related factors.

# Libraries Used ** 
data tranform and statistical operation
import pandas as pd
import numpy as np
### visualization
import matplotlib.pyplot as plt
""""magic command after using this, no need to use plt.show()
Without %matplotlib inline, you would need to use plt.show() after creating a plot to display
it. However, with this magic command, plots are automatically displayed when the cell is run."""
%matplotlib inline
import seaborn as sns
from pandas.plotting import scatter_matrix
### sampling and split
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
# imputation
from sklearn.impute import SimpleImputer
# pipeline
from sklearn.pipeline import Pipeline
# model selection, regressors
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
# normalization and scaling technique
from sklearn.preprocessing import StandardScaler
# model evaluation
from sklearn.metrics import mean_squared_error
