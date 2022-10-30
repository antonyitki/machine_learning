"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Coursework 1, Machine Learning (CMP020X303A), Year 3
YIT19488399, A R O
Deliver by 19 of October 2022, 25% of total mark of module

Version 01.02, "Analizer LASSO and more"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""


import numpy as np
import pandas as pd
# visualizations in matplotlib
from matplotlib import pyplot as plot
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import ols
from numpy import mean, std, absolute, arange
from sklearn.model_selection import cross_val_score, train_test_split, RepeatedKFold
#pip install sklearn (windows machine)
from sklearn import linear_model
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as mp
# pip install hvplot (holoviz, https://holoviz.org/)
import time
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import Ridge, RidgeCV, Lasso, LassoCV
# Lab3 from Dr. Gu
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
# to open some websites links for SL information theory
import webbrowser
# Necesary for Dr Gu code lab3
from sklearn import preprocessing


# input validation for read file. Use relative path and locate file in same folder as code (python).
print("Locate the file in the right directory.")
file = ''
while True:
    file = input("Enter filename with extension, plesae:\n")
    if file != "cw1data.csv":
        print('WRONG file name.')
        continue
    else:
        print(f"The file name entered matches! It is {file}.\n")
        break
# print some file information on screen
df = pd.read_csv(file)
# print on screen initial and end of rows and all columns
print(df)
#some analysis of data on file in text and numeric values
print(f"\nNumber of unique values in each column:\n {df.nunique()}\n")
print(f"First 5 rowas and all columns\n {df.head()}\n")
print(f"The total number of elements are: {df.size}, without headers of first row.\n")
# looking for empty cells in the .csv file
print(f"If 'True' everywhere, this means not empty values in the file\n {df.notnull()}\n")
# standard information about dataset
print(df.info(), "\n")
# analytical information for each column with all rows
print("Some useful information about dataset")
print(df.describe())
###############################################################################
# VISUALISATION
# all data in a same figure with 14 lines
plt.plot(df[:])
plt.title("All columns togheter")
plt.show()
df.plot(y=["x1", "x2", "x3"], x="y",  kind="bar", figsize=(9, 7))
mp.show()
# each of the columns in a separate subplot
axs = df.plot.area(figsize=(14, 9), subplots=True)
plt.show()
df.plot(x="y", y=["x3"], kind="bar",figsize=(9,7), title="x3 colums", xlabel="x3 values", ylabel="y values")
plt.show()
plt.pie(df['y'])
plt.show()
plt.pie(df['x1'])
plt.title("x1 pie chart")
plt.show()
# scatter the data
plt.scatter(df['y'], df['x1'], color = "green")
plt.title("x1 with y")
plt.xlabel("x1")
plt.ylabel("y")
plt.show()
# anoter scatter plot with x7
plt.scatter(df['y'], df['x7'], color = "purple")
plt.title("x7 with y")
plt.xlabel("x7")
plt.ylabel("y")
plt.show()
plt.scatter(df['x1'], df['y'], c=df['x2'], s=df['x3'])
plt.xlabel("x1")
plt.ylabel("y")
plt.title("y, x1, with x2 and x3 in a separate bar with colour")
plt.colorbar()
plt.show()
df.plot(x="x1", y=("x12"), kind="scatter", title="x1 with x12", xlabel= "x12", ylabel= "x1", figsize=(8,6))
plt.show()
# scatter plot with all columns in same figure with different colors
plt.scatter(y=(df['y']), x=(df['x1'] + df["x2"] + df['x3'] + df['x4'] + df["x5"] + df['x6'] + df['x7'] + df["x8"] + df['x9'] + df['x10'] + df["x11"] + df['x12']))
plt.title("all togheter points")
plt.show()
# histogram of x8
df.x8.plot(kind='hist', color='purple', edgecolor='black', figsize=(10,7))
plt.title('Distribution of x8', size=24)
plt.xlabel('x8', size=18)
plt.ylabel('Frequency', size=18);
plt.show()
# histogram of x1
df.x1.plot(kind='hist', color='red', edgecolor='green', figsize=(10,7))
plt.title('Distribution of x1', size=24)
plt.xlabel('x1', size=18)
plt.ylabel('Frequency', size=18)
plt.show()
#density plots of all columns
df.plot.density(figsize=(8,6), logx=True, linewidth=4)
plt.show()
df[["y", "x1", "x12", "x3", "x4","x5","x6","x7","x8","x9","x10","x11","x12"]].plot.kde()
plt.show()
# super advance figure with all columns and all rows in same figure
# https://medium.com/analytics-vidhya/applying-multiple-linear-regression-in-house-price-prediction-47dacb42942b
sns.pairplot(df)
plot.show()
plt.style.use('ggplot')
# even more advances figures, looking for ??? with printed information!!
#https://www.learndatasci.com/tutorials/predicting-housing-prices-linear-regression-using-python-pandas-statsmodels/
df_model = ols('y ~ x7', data = df).fit()
print(df_model)
print(df_model.summary())
#regression plots with previous link information
fig = plt.figure(figsize=(15,8))
# pass in the model as the first parameter, then specify the 
# predictor variable we want to analyze
fig = sm.graphics.plot_regress_exog(df_model, "x7", fig=fig)
fig.show()
input('Press ENTER to exit (image need to be closed manually)')
df_modeltwo = ols('y ~ x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9 + x10 + x11  + x12', data = df).fit()
df_modeltwo_summary = df_modeltwo.summary()
fig = plt.figure(figsize=(18,10))
fig = sm.graphics.plot_partregress_grid(df_modeltwo, fig=fig)
print(df_modeltwo)
print('\n')
print(df_modeltwo_summary)
fig.show()
input('Press ENTER to exit (image need to be closed manually)')
print("\n\n\n")
###############################################################################
# REGRESSIONS
print("\nNow is time to do some regression!!\n")
#splitting data intro training and testing. 70 % and 30 %.
y_data = df['y']
print(y_data)
print("\n")
# same as x_data = df['x1', x2, ..., x12, x13]
x_data = df.drop(["y"], axis=1)
print(x_data)
print("\n")
# predefined function to split 2 datasets randomly selecting rows
X_train, X_test, Y_train, Y_test = train_test_split(x_data, y_data, test_size=.30, random_state=35)
print(f"This is training data INPUT\n {X_train}\n")
print(f"This is training data OUTPUT\n {Y_train}\n")
print("########################################################################\n")
print(f"This is testing data INPUT\n {X_test}\n")
print(f"This is testing data OUTPUT\n {Y_test}\n")
print("\nFor ALL regressions methods will be used same initial splitting dataset\n")
# LASSO regression:
# Solution Lab2 by Dr Gu
lasso_model = Lasso(alpha=1.0)
lasso=lasso_model.fit(X_train , Y_train)
lasso_predict = lasso.predict(X_test)
def MAPE(Y_actual,Y_Predicted):
    mape = np.mean(np.abs((Y_actual - Y_Predicted)/Y_actual))*100
    return mape
Lasso_MAPE = MAPE(Y_test,lasso_predict)
print("MAPE value: ",Lasso_MAPE)
Accuracy = 100 - Lasso_MAPE
print('Accuracy of Lasso Regression: {:0.2f}%.'.format(Accuracy))
# end of Dr Gu lasso regression model code
# Finding best (optimal) alpha value for lasso regression:
# Lasso with 50 fold cross-validation
model = LassoCV(cv=50, random_state=0, max_iter=10000)
# Fit model
model.fit(X_train, Y_train)
print(f"\nThe best value for alpha is: {model.alpha_}. This values will be used for next LASSO models calculations.")
# Set best alpha
lasso_best = Lasso(alpha=model.alpha_)
print(lasso_best.fit(X_train, Y_train))
#Show model coefficients and names:
print((list(zip(lasso_best.coef_, x_data))))
print('R squared training set', round(lasso_best.score(X_train, Y_train)*100, 2))
print('R squared test set', round(lasso_best.score(X_test, Y_test)*100, 2))
# same Lasso regression model as Dr Gu, but with new alpha value. Accuracy of new lasso model calculated as well
print("\nAccuracy model with new best alpha value: ")
lasso_model = Lasso(alpha=model.alpha_)
lasso=lasso_model.fit(X_train , Y_train)
lasso_predict = lasso.predict(X_test)
def MAPE(Y_actual,Y_Predicted):
    mape = np.mean(np.abs((Y_actual - Y_Predicted)/Y_actual))*100
    return mape
Lasso_MAPE = MAPE(Y_test,lasso_predict)
print("MAPE value: ",Lasso_MAPE)
Accuracy = 100 - Lasso_MAPE
print('Accuracy of Lasso Regression: {:0.2f}%.'.format(Accuracy))
time.sleep(3.0)
# some other types of evaluation of the performance of the model
print("More lasso performance measures: \n")
print(lasso.score(X_test, Y_test))
print(lasso.score(X_train, Y_train))
print(lasso.coef_)
print(lasso.sparse_coef_)
# evaluate an lasso regression model on the dataset
# https://machinelearningmastery.com/lasso-regression-with-python/
# define model evaluation method
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate model
scores = cross_val_score(lasso, x_data, y_data, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
# force scores to be positive
scores = absolute(scores)
print('Mean MAE: %.3f (%.3f)' % (mean(scores), std(scores)))
# https://www.pluralsight.com/guides/linear-lasso-ridge-regression-scikit-learn
model_lasso = Lasso(alpha=model.alpha_)
model_lasso.fit(X_train, Y_train) 
pred_train_lasso= model_lasso.predict(X_train)
print(np.sqrt(mean_squared_error(Y_train,pred_train_lasso)))
print(r2_score(Y_train, pred_train_lasso))
pred_test_lasso= model_lasso.predict(X_test)
print(np.sqrt(mean_squared_error(Y_test,pred_test_lasso))) 
print(r2_score(Y_test, pred_test_lasso))
# for future research and investigation
print("Compare performance with alpha = 1 and with best value of alpha. If numbers have NO SENSE, do research!!!\n")
print("########################################################################\n")
print("Time for L2 regression:\n")
# https://www.pluralsight.com/guides/linear-lasso-ridge-regression-scikit-learn
rr = Ridge(alpha=0.01)
rr.fit(X_train, Y_train) 
pred_train_rr= rr.predict(X_train)
# RMSE and R-squared values for the Ridge Regression model on the training data. Second value is a %
print(np.sqrt(mean_squared_error(Y_train,pred_train_rr)))
print(r2_score(Y_train, pred_train_rr))
pred_test_rr= rr.predict(X_test)
# RMSE and R-squared values for the Ridge Regression model on the test data. Second value is a %
print(np.sqrt(mean_squared_error(Y_test,pred_test_rr))) 
print(r2_score(Y_test, pred_test_rr))
# other approach for L2 regression: https://vitalflux.com/ridge-regression-concepts-python-example/
print("\n")
# Fit a pipeline using Training dataset and related labels. Use L2 algorithm for training the model
pipeline = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
pipeline.fit(X_train, Y_train)
# Calculate the predicted value for training and test dataset
y_train_pred = pipeline.predict(X_train)
y_test_pred = pipeline.predict(X_test)
# Mean Squared Error
print('MSE train: %.3f, test: %.3f' % (mean_squared_error(Y_train, y_train_pred),
                mean_squared_error(Y_test, y_test_pred)))
# R-Squared
print('R^2 train: %.3f, test: %.3f' % (r2_score(Y_train, y_train_pred), r2_score(Y_test, y_test_pred)))
###initialize list to store coefficient values. https://jbhender.github.io/Stats506/F17/Projects/G13/Python.html
coef=[]
alphas = range(0,40)
for a in alphas:
  ridgereg=Ridge(alpha=a)
  ridgereg.fit(x_data,y_data)
  coef.append(ridgereg.coef_)
###Make plot of Beta as a function of Alpha
fig=plt.figure()
ax=fig.add_subplot(111)
ax.plot(alphas,coef)
ax.set_xlabel('Alpha (Regularization Parameter)')
ax.set_ylabel('Beta (Predictor Coefficients)')
ax.set_title('Ridge Coefficients vs Regularization Parameters')
ax.axis('tight')
### ax.legend(loc='best')
fig.savefig('coef_vs_alpha.png')
plt.show()
# define cross-validation method to evaluate model
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
#define model
model = RidgeCV(alphas=arange(0.1, 4, 0.01), cv=cv, scoring='neg_mean_absolute_error')
# fit model, necessary to display lambda value in next line of code
model.fit(x_data, y_data)
#display lambda that produced the lowest test MSE
print(model.alpha_)
print("Looks like lambda is better as lower. Because I selected 0.01 at the beginning, I assume is the best model possible!!\n")
print("########################################################################\n")
print("PREDICTIONS for both models, L1 and L2:\n")
# define new data(enter same number of data as columns, 14)
row = [0.00632,18.00,2.310,10,0.5380,6.5750,65.20,4.0900,1,296.0,15.30,396.90,4.98]
# make a prediction
yhat_lasso1 = lasso.predict([row])
yhat_lasso2 = lasso_model.predict([row])
yhat_lasso3 = lasso_best.predict([row])
# summarize prediction
print('Predicted: %.7f' % yhat_lasso1)
print('Predicted: %.7f' % yhat_lasso2)
print('Predicted: %.7f' % yhat_lasso3)
# We print the R2-score for the training and test set.
print('R squared training set', round(lasso.score(X_train, Y_train)*100, 2))
print('R squared test set', round(lasso.score(X_test, Y_test)*100, 2))
print('R squared...', round(lasso.score(x_data, y_data)*100, 2))
print('Now prediction for L2 model with same values:')
# make a prediction
yhat_rr1 = model.predict([row])
yhat_rr2 = pipeline.predict([row])
yhat_rr3 = rr.predict([row])
# summarize prediction
print('Predicted: %.7f' % yhat_rr3)
print('Predicted: %.7f' % yhat_rr1)
print('Predicted: %.7f' % yhat_rr2)
#We print the R2-score for the training and test set.
print('R squared training set', round(rr.score(X_train, Y_train)*100, 2))
print('R squared test set', round(rr.score(X_test, Y_test)*100, 2))
print('R squared...', round(rr.score(x_data, y_data)*100, 2))
# Make RESEARCH wgy ridge is no sense values compared to lasso and all three value are very different!!!
print("########################################################################\n")
print("Linear regression: ")
# Other regressions (linear): https://datatofish.com/multiple-linear-regression-python/
# with sklearn
regr = linear_model.LinearRegression()
regr.fit(x_data, y_data)
print('Intercept: \n', regr.intercept_)
print('Coefficients: \n', regr.coef_)
# with statsmodels
x_data = sm.add_constant(x_data) # adding a constant
model = sm.OLS(y_data, x_data).fit()
predictions = model.predict(x_data) 
print_model = model.summary()
print(print_model)
print("########################################################################\n")
print("Some other SL regressions:\n")
print("POLINOMIAL")
mymodel = np.poly1d(np.polyfit(x_data["x3"], y_data, 3))
myline = np.linspace(1, 88, 100)
plt.scatter(x_data["x3"], y_data)
plt.plot(myline, mymodel(myline))
plt.tight_layout()
plt.title('polinomial with grade 3')
plt.ylabel('y')
plt.xlabel('x3')
plt.savefig('polinomial3.png')
plt.show() 
print("How well does my data fit in a polynomial regression?")
print(r2_score(y_data, mymodel(x_data["x3"])))
# polinomial with grade 6
poly_fit = np.poly1d(np.polyfit(x_data["x3"], y_data, 6))
myline = np.linspace(1, 80, 100)
plt.scatter(x_data["x3"], y_data)
plt.plot(myline, poly_fit(myline))
plt.tight_layout()
plt.title('polinomial with grade 6')
plt.ylabel('y')
plt.xlabel('x3')
plt.show() 
print("How well does my data fit in a polynomial with grade 6 regression?")
print(r2_score(y_data, poly_fit(x_data["x3"])))
print("########################################################################\n")
# convert y values to categorical values
lab = preprocessing.LabelEncoder()
y_transformed = lab.fit_transform(Y_train)
Y_test_transformed = lab.fit_transform(Y_test)
to_predict_my_value = [2,3,4,5,6,7,8,9,11.1,12.2,3.2,5,56] #13 values
print("Time for Dr. Gu in lab3 regressions:")
classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_transformed)
y_pred = classifier.predict(X_test)
print(confusion_matrix(Y_test_transformed, y_pred))
print(classification_report(Y_test_transformed, y_pred))
#
clf = RandomForestClassifier(n_estimators=10)
clf = clf.fit(X_train, y_transformed)
# Prediction1
print (clf.predict([to_predict_my_value]))
print(confusion_matrix(Y_test_transformed, y_pred))
print(classification_report(Y_test_transformed, y_pred))
#
clf1 = tree.DecisionTreeClassifier()
clf1 = clf1.fit(X_train, y_transformed)
# Prediction
print (clf1.predict([to_predict_my_value]))
print(confusion_matrix(Y_test_transformed, y_pred))
print(classification_report(Y_test_transformed, y_pred))
#
clf2 = GaussianNB()
clf2 = clf2.fit(X_train, y_transformed)
# Prediction
print (clf2.predict([to_predict_my_value]))
print(confusion_matrix(Y_test_transformed, y_pred))
print(classification_report(Y_test_transformed, y_pred))
print("########################################################################\n")
print("\nThis is the END. Some useful theory are in these websites: \n")
# https://www.projectpro.io/article/types-of-regression-analysis-in-machine-learning/410
webbrowser.open('https://www.projectpro.io/article/types-of-regression-analysis-in-machine-learning/410')
time.sleep(5.00)
# https://www.listendata.com/2018/03/regression-analysis.html
webbrowser.open('https://www.listendata.com/2018/03/regression-analysis.html')
##############################END#################################################
'''
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import BaseDecisionTree
from sklearn.tree import ExtraTreeRegressor
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import ComplementNB
from sklearn.ensemble import GaussianNB
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import RandomForestClassifier
from sklearn.tree import RandomForestRegressor
from sklearn.naive_bayes import RandomForestClassifier
from sklearn.naive_bayes import RandomForestRegressor

from sklearn.linear_model import LinearRegression, ElasticNet, LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
'''