# FeatureFinder


Data exploration and pre-processing:

DataExplorationAndPreProcessing.ipynb: Load the employee turnover rate, employee’s answers to the questions, tenure of key positions, and median household income for the location (from US Census). Transform the data to make it normally distributed, standardize the data, save the data.

AddingEmployeeSalary.ipynb: Load the salary for each employee and calculate the median salary for each location. Standardize the data and save it.

AddingResponseStandardDeviation.ipynb: Load the individual employee’s answers to the questions and calculate how varied the responses are at each location. Standardize and save it.

Regression Analysis:

Run_LassoRegression_and_Average.ipynb: Load all the features saved by the notebooks above, and run Lasso regression (1000 times). Produce a model that predicts employee turnover rate using few selected features. Save the results for linear regression step.

LinearRegressionWithInteractions.ipynb: Load the results of the lasso regression. Choose the top 3 features and run linear regression using these features and their interactions. A model confirms the validity of the features and shows that interactions are not significantly big. Save the results for the use in Streamlit app. 

Interactive Web App:
The app will interactively show how much each senior care center has to improve on key features in order to meet their "target" emplyee turn over rate.

For running the Web App, please install “Streamlit” from (https://www.streamlit.io/), and run the file by typing: streamlit run FeaturePredictionForTurnover_Streamlit.py
