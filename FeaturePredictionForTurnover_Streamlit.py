#import necessary packages
#for the web app
import streamlit as st

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
sb.set()
color = sb.color_palette()
import matplotlib as mpl
import pickle
from sklearn import preprocessing as pp 
from sklearn import linear_model
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from scipy.stats import pearsonr 
from sklearn.decomposition import PCA
from sklearn.linear_model import LassoCV
from scipy.stats import wilcoxon

#title of the app
st.title('Top 3 features predicting employee turnover')

# Loading data
with open('lasso_output.pickle', 'rb') as f:  
    test_r_squared, lasso_alpha, lasso_coef = pickle.load(f)

with open('scaler.pickle', 'rb') as f:  
    scalerEmployeeResponse, scalerOccupancyTurnover = pickle.load(f)

with open('original_values.pickle', 'rb') as f:  
    TurnOver2, dataX2 = pickle.load(f)

with open('linear_output.pickle', 'rb') as f:  
    linear_r_squared_df, linear_coef_df = pickle.load(f)

#loading the dataframe
feature_set = pd.read_pickle("./feature_set.pkl")
turnOverRate = pd.read_pickle("./turnOverRate.pkl")
LocationCodeComprehensive = pd.read_pickle("./LocationCodeComprehensive.pkl")
location_pay = pd.read_pickle("./location_pay.pkl")
location_sd = pd.read_pickle("./location_sdS_df.pkl")
lasso_r_squared_df = pd.read_pickle("./lasso_r_squared_1000.pkl")
lasso_alpha_df = pd.read_pickle("./lasso_alpha_1000.pkl")
lasso_coef_df = pd.read_pickle("./lasso_coef_1000.pkl")
selected_features_interaction = pd.read_pickle("./selected_features_interaction.pkl")

#Rename the column
LocationCodeComprehensive.rename(columns = {'Unnamed: 2':'location code'}, inplace = True) 

#Calculate mean
mean_coef=lasso_coef_df.mean()

#Chose top coefficients.
abs_mean_coef=abs(mean_coef)
top_coef=abs_mean_coef.sort_values(ascending=False)

#Put features together
all_features = pd.concat([feature_set,location_pay,location_sd], axis = 1)

#Side bar with choice of location
location = st.sidebar.selectbox(
    'Select your location',
     LocationCodeComprehensive)

#Index for current location
index = LocationCodeComprehensive==location

selected_features_interaction2=pd.DataFrame(data=selected_features_interaction)
x_linear = selected_features_interaction

#Calculate the mean for the coefficients
mean_linear_coef=linear_coef_df.mean()

PredictionMatrix=x_linear.multiply(mean_linear_coef)
PredictionValue=PredictionMatrix.sum(axis=1)

#User chose how much to reduce, figure out how much do we need to reduce in the log transformed scaled unit.
TurnOver2=TurnOver2.reset_index(drop=True)
dataX2=dataX2.reset_index(drop=True)
dataX2.columns = range(dataX2.shape[1])
base_turnover=TurnOver2[index]

mean_coef_reduced = mean_linear_coef.apply(lambda x: round(x, 2 - int(np.floor(np.log10(abs(x))))))

st.subheader("Feature 55: Provide excellent service")
st.write(mean_coef_reduced[0])

st.subheader("Feature 62: Chef's tenure")
st.write(mean_coef_reduced[1])

st.subheader("Feature 21: Managers avoid playing favorites")
st.write(mean_coef_reduced[2])

st.markdown('***')

st.subheader("Current and target turnover rate for your location (%)")


#slider for choosing the goal turnover rate
new_turnover = st.slider('Goal turn over rate (%)', 0, 400, 100)
new_turnover = new_turnover/100

#Plot the current and target turnover rate
fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
sb.barplot(TurnOver2[index]*100, orient = 'v', ax =ax1)
ax1.set_ylabel('turnover rate (%)')
ax1.set_xlabel('current')
ax1.set_ylim((0,400)) 
sb.barplot(new_turnover*100, orient = 'v', color = 'red', ax = ax2)
ax2.set_xlabel('target')
st.pyplot()


#new_turnover = base_turnover*(1-(percent_turnover/100))
log_new_turnover = np.log(new_turnover+0.1)
scaled_turnover = (log_new_turnover-scalerOccupancyTurnover.mean_[1])/scalerOccupancyTurnover.scale_[1]

current_turnover = turnOverRate[index]
turnover_change = scaled_turnover-current_turnover
#currently assumes we use the top component and have selected 3 features.
feature_change = turnover_change/(mean_linear_coef[0]+mean_linear_coef[3]*selected_features_interaction2.loc[LocationCodeComprehensive==location,1]+mean_linear_coef[4]*selected_features_interaction2.loc[LocationCodeComprehensive==location,2])

#bring it back to what it means in the raw data scale. The raw value of the feature has to be between 0 and 1.
new_feature = selected_features_interaction2.loc[LocationCodeComprehensive==location,0]+feature_change

new_feature_before_scale = (new_feature*scalerEmployeeResponse.scale_[55])+scalerEmployeeResponse.mean_[55]
new_feature_before_transform = 1.1-np.exp(0.1-new_feature_before_scale)
current_feature = dataX2.loc[LocationCodeComprehensive==location,55]

st.markdown('***')

st.subheader("Current and target scores for the (Q55: Provide excellent service)")

#Plot the current and target score
fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
#current score
sb.barplot(current_feature.iloc[0], orient = 'v', ax =ax1)
ax1.set_ylabel('score (Q55) (%)')
ax1.set_xlabel('current')
ax1.set_ylim((0,1)) 

#TargetScore
#need to check if we reached the limit (score must be between 0 and 1)
TargetScore = new_feature_before_transform.iloc[0]
if TargetScore >= 0 and TargetScore <= 1:
   sb.barplot(TargetScore, orient = 'v', color = 'red', ax = ax2)
   ax2.set_xlabel('target')
   st.pyplot()
elif TargetScore < 0:
   sb.barplot(0, orient = 'v', color = 'red', ax = ax2)
   ax2.set_xlabel('target')
   st.pyplot()

   st.write('Target score = 0 reached score limit')
elif TargetScore > 1:
   sb.barplot(1, orient = 'v', color = 'red', ax = ax2)
   ax2.set_xlabel('target')
   st.pyplot()

   st.write('Target score = 1 reached score limit')



st.markdown('***')
#have option for the 2nd question.
feature_change2 = turnover_change/(mean_linear_coef[2]+mean_linear_coef[4]*selected_features_interaction2.loc[LocationCodeComprehensive==location,0]+mean_linear_coef[5]*selected_features_interaction2.loc[LocationCodeComprehensive==location,1])

#bring it back to what it means in the raw data scale. The raw value of the feature has to be between 0 and 1.
new_feature2 = selected_features_interaction2.loc[LocationCodeComprehensive==location,2]+feature_change2

new_feature_before_scale2 = (new_feature2*scalerEmployeeResponse.scale_[21])+scalerEmployeeResponse.mean_[21]
new_feature_before_transform2 = 1.1-np.exp(0.1-new_feature_before_scale2)
current_feature2 = dataX2.loc[LocationCodeComprehensive==location,21]


st.subheader("Current and target scores for the (Q21: Managers avoid playing favorites)")

#Plot the current and target score
fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
#current score
sb.barplot(current_feature2.iloc[0], orient = 'v', ax =ax1)
ax1.set_ylabel('score (Q21) (%)')
ax1.set_xlabel('current')
ax1.set_ylim((0,1)) 

#TargetScore
#need to check if we reached the limit (score must be between 0 and 1)
TargetScore2 = new_feature_before_transform2.iloc[0]
if TargetScore2 >= 0 and TargetScore2 <= 1:
   sb.barplot(TargetScore2, orient = 'v', color = 'red', ax = ax2)
   ax2.set_xlabel('target')
   st.pyplot()
elif TargetScore2 < 0:
   sb.barplot(0, orient = 'v', color = 'red', ax = ax2)
   ax2.set_xlabel('target')
   st.pyplot()

   st.write('Target score = 0 reached score limit')
elif TargetScore2 > 1:
   sb.barplot(1, orient = 'v', color = 'red', ax = ax2)
   ax2.set_xlabel('target')
   st.pyplot()

   st.write('Target score = 1 reached score limit')


