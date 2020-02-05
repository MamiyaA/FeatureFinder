import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd


import matplotlib.pyplot as plt
import seaborn as sb
sb.set()
color = sb.color_palette()
import matplotlib as mpl
import pickle

from sklearn import preprocessing as pp 
#import the entire linear model
from sklearn import linear_model
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from scipy.stats import pearsonr 
from sklearn.decomposition import PCA
from sklearn.linear_model import LassoCV
#from sklearn.preprocessing import PolynomialFeatures

from scipy.stats import wilcoxon

from sklearn.linear_model import LogisticRegressionCV
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_auc_score



#title of the app
st.title('Predicting employee turnover')

st.write("Scatter plot of predicted and actual turnover rate")

# Getting back the objects:
with open('lasso_output.pickle', 'rb') as f:  
    test_r_squared, lasso_alpha, lasso_coef = pickle.load(f)

#loading the dataframe
feature_set = pd.read_pickle("./feature_set.pkl")
turnOverRate = pd.read_pickle("./turnOverRate.pkl")
LocationCodeComprehensive = pd.read_pickle("./LocationCodeComprehensive.pkl")
location_pay = pd.read_pickle("./location_pay.pkl")
location_sd = pd.read_pickle("./location_sdS_df.pkl")
lasso_r_squared_df = pd.read_pickle("./lasso_r_squared_1000.pkl")
lasso_alpha_df = pd.read_pickle("./lasso_alpha_1000.pkl")
lasso_coef_df = pd.read_pickle("./lasso_coef_1000.pkl")

#Rename the column
LocationCodeComprehensive.rename(columns = {'Unnamed: 2':'location code'}, inplace = True) 


#plot the ditribution of the coefficent of determination
test_r_squared_df=pd.DataFrame(data=test_r_squared)
test_r_squared_df.rename(columns = {0:'coefficient of determination'}, inplace = True) 

#figure out how to plot later.
#sb.distplot(test_r_squared_df)
#sb.despine()

#Calculate mean and plot
mean_coef=lasso_coef_df.mean()
plt.plot(mean_coef,alpha=0.7,linestyle='none',marker='*',markersize=5,color='red',label=r'Lasso; $\alpha = 0.05$',zorder=7)
st.pyplot()

#Chose top coefficients.
abs_mean_coef=abs(mean_coef)
top_coef=abs_mean_coef.sort_values(ascending=False)

all_features = pd.concat([feature_set,location_pay,location_sd], axis = 1)


#Side bar with choice of location
option = st.sidebar.selectbox(
    'Please select your location',
     LocationCodeComprehensive)
#Side bar with choice of number of features 
NofFeatures = st.sidebar.slider(
    'Select number of features',
    0, 65, (3))

#Side bar with choice of number of training 
NofTraining = st.sidebar.slider(
    'Select number of training',
    1, 1000, (10))

'Current location:', option

'Number of features to use:', NofFeatures

'Training number:', NofTraining

#run linear regression with the chosen number of figures and plot all the data point against the predicted values for that data.

#Choose the features with top n coefficients 
features_number = NofFeatures
#get the top n features and construct a new feature.
top_coef=abs_mean_coef.sort_values(ascending=False)

selected_features = all_features.iloc[:,top_coef.index[0]]

for features in range(features_number-1):
    selected_features = pd.concat([selected_features, all_features.iloc[:,top_coef.index[features+1]]], axis = 1)

#incorporate interaction terms
poly = pp.PolynomialFeatures(interaction_only=True,include_bias = False)
selected_features_interaction=poly.fit_transform(selected_features)
selected_features_interaction=pd.DataFrame(data=selected_features_interaction)
selected_features_interaction.head()

#make a model with interactions.
#Run N times and take the average.
#Try to use the shuffle split function => later
x_linear = selected_features_interaction
y_linear = turnOverRate #Turn over

train_number = NofTraining
#Run this part for selected number of times and get the values for r-squared, alpha, and the weights.
#reg.alpha_, reg.coef_, r2score.

#initialize
linear_r_squared = np.zeros((train_number,1))
linear_coef = np.zeros((train_number,x_linear.shape[1]))

for trialNo in range(train_number):
    
    #split 80-20
    Xl_train, Xl_test, Yl_train, Yl_test = train_test_split(x_linear, y_linear, test_size=0.2)
    

    linear_reg = linear_model.LinearRegression()
    linear_reg.fit(Xl_train,Yl_train)
    
    
    Ypredl = linear_reg.predict(Xl_test)
    
    linear_r_squared[trialNo] = r2_score(Yl_test, Ypredl)
    linear_coef[trialNo,:] = linear_reg.coef_

#Plot the distribution.
linear_r_squared_df=pd.DataFrame(data=linear_r_squared)
#linear_r_squared_df.hist()
sb.distplot(linear_r_squared_df)
sb.despine()
st.pyplot()

#Calculate the prediction with the average model and plot the actual vs the prediction.
#Highlight the current location.
linear_coef_df = pd.DataFrame(data=linear_coef)
mean_linear_coef=linear_coef_df.mean()
PredictionMatrix=x_linear.multiply(mean_linear_coef)
PredictionValue=PredictionMatrix.sum(axis=1)
#Put together into a single data frame so we can plot easier
frameMVP = { 'turnover rate': turnOverRate, 'Predicted turnover': PredictionValue} 
dataForPlotMVP = pd.DataFrame(frameMVP)
sb.lmplot(x='Predicted turnover', y='turnover rate', data=dataForPlotMVP)
st.pyplot()


#User chose how much to reduce.
#Split out the best number according to the equation. 4:30pm

#rest 30 min and write down the new script till 5:45 pm. 
