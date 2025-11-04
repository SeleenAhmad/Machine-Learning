import numpy as np
import pandas as pd
from pandas.core.dtypes.cast import np_can_hold_element

df= pd.read_csv(r"C:\Users\DELL\Downloads\biomass_large_sample.csv")
print(df.head())
print(df.info())
print(df.describe())
X=df[['rainfall','temperature','NDVI']]
Y=df[['biomass']].values.ravel()
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
linear_model= LinearRegression()
rf_model= RandomForestRegressor(n_estimators=250, random_state=99)
X_Train,X_TEST,Y_Train,Y_Test=train_test_split(
    X,Y ,test_size=.4, random_state=99
)
print("Train shapes for x and y: ", X_Train.shape,Y_Train.shape)
print("Test shapes of x and y :" , X_TEST.shape,Y_Test.shape)
linear_model.fit(X_Train, Y_Train)
rf_model.fit(X_Train, Y_Train)
linear_pred1=linear_model.predict(X_TEST)
rf_pred1= rf_model.predict(X_TEST)
print("R2 score for Linear regression:",r2_score(Y_Test,linear_pred1))
print("R2 score for Regression forest: ", r2_score(Y_Test,rf_pred1))
print("MAE via linear regression : ",mean_absolute_error(Y_Test, linear_pred1))
print("MAE via Regression Forest : ",mean_absolute_error(Y_Test, rf_pred1))
from xgboost import XGBRegressor
xgb= XGBRegressor(n_estimators=250, learning_rate=.05,max_depth=5, random_state=99,subsample=.8,colsample_bytree=.8)
xgb.fit(X_Train,Y_Train)
Y_pred_xgb=xgb.predict(X_TEST)
print("R2 score XGBOOST :" , r2_score(Y_Test,Y_pred_xgb))
print("MAE via XGBOOST : ",mean_absolute_error(Y_Test,Y_pred_xgb))
import numpy as np
Y_Test=np.array(Y_Test).flatten()
linear_pred1 = np.array(linear_pred1).flatten()
rf_pred1 = np.array(rf_pred1).flatten()
Y_pred_xgb=np.array(Y_pred_xgb).flatten()
x_axis=np.arange(len(Y_Test))
plt.figure(figsize=(8.5,6))
plt.plot(x_axis,Y_Test,label="Actual Biomass",marker='o')
plt.plot(x_axis, linear_pred1, label=" linear regression prediction",marker='x')
plt.plot(x_axis, rf_pred1, label= "Regression forest", marker='s')
plt.plot(x_axis, Y_pred_xgb, label= "XGBOOST", marker='d')
plt.xlabel("Test Sample")
plt.ylabel("Biomass")
plt.title("Actual vs predicted Biomass via three Machine Learning Methods")
plt.legend()
plt.grid(True)
plt.show()
plt.figure(figsize=(12,6))
plt.scatter(x_axis,Y_Test,label="Actual Biomass",color='black',s=20,alpha=.7 )
plt.scatter(x_axis, linear_pred1, label=" linear regression prediction", color='blue',s=15,alpha=.6)
plt.scatter(x_axis, rf_pred1, label= "Regression forest",color= 'green',s=15,alpha=.6)
plt.scatter(x_axis,Y_pred_xgb, label= "XGBOOST",color='red',s=15,alpha=.6)
plt.xlabel("Test Sample")
plt.ylabel("Biomass")
plt.title("Actual vs Predicted Biomass via Three Machine Learning Methods")
plt.legend()
plt.grid(True)
plt.show()
plt.figure(figsize=(14, 6))
plt.scatter(x_axis, Y_Test, label="Actual Biomass", color='black', s=20, alpha=0.7)
plt.scatter(x_axis, linear_pred1, label="Linear Regression", color='blue', s=15, alpha=0.5)
plt.scatter(x_axis, rf_pred1, label="Random Forest", color='green', s=15, alpha=0.5)
plt.scatter(x_axis, Y_pred_xgb, label="XGBoost", color='red', s=15, alpha=0.5)
plt.plot(x_axis, linear_pred1, color='blue', linestyle='-', linewidth=1)
plt.plot(x_axis, rf_pred1, color='green', linestyle='-', linewidth=1)
plt.plot(x_axis, Y_pred_xgb, color='red', linestyle='-', linewidth=1)
plt.xlabel("Test Sample")
plt.ylabel("Biomass")
plt.title("Actual vs Predicted Biomass: Scatter + Trend Lines")
plt.legend()
plt.grid(True)
plt.show()
from statsmodels.nonparametric.smoothers_lowess import lowess
Y_Test=np.array(Y_Test).flatten()
linear_pred1 = np.array(linear_pred1).flatten()
rf_pred1 = np.array(rf_pred1).flatten()
Y_pred_xgb=np.array(Y_pred_xgb).flatten()
x_axis=np.arange(len(Y_Test))
plt.figure(figsize=(14,6))
plt.plot(x_axis,Y_Test,label="Actual Biomass",marker='o')
plt.plot(x_axis, linear_pred1, label=" linear regression prediction",marker='x')
plt.plot(x_axis, rf_pred1, label= "Regression forest", marker='s')
plt.plot(x_axis, Y_pred_xgb, label= "XGBOOST", marker='d')
smooth_linear=lowess(linear_pred1,x_axis,frac=.05)
smooth_rf=lowess(rf_pred1,x_axis,frac=.05)
smooth_xgb=lowess(Y_pred_xgb,x_axis,frac=0.05)
smooth_actual=lowess(Y_Test,x_axis,frac=0.05)
plt.plot(smooth_linear[:,0], smooth_linear[:,1], color='blue', linewidth=2)
plt.plot(smooth_rf[:,0], smooth_rf[:,1], color='green', linewidth=2)
plt.plot(smooth_xgb[:,0], smooth_xgb[:,1], color='red', linewidth=2)
plt.plot(smooth_actual[:,0],smooth_actual[0:,1], color= 'black',linewidth=2)
plt.xlabel("Test Sample")
plt.ylabel("Biomass")
plt.title("Actual vs predicted Biomass via three Machine Learning Methods")
plt.legend()
plt.grid(True)
plt.show()
xgb= XGBRegressor(objective='reg:squarederror', random_state=99)
param_grid={
    'n_estimator':[100,200,400],
    'max_depth':[3,5,7],
    'learning_rate':[.01,.05,.1],
    'subsample':[.7,.8,.9],
    'colsample_bytree':[.7,.8,1]
}
grid=GridSearchCV(estimator=xgb, param_grid=param_grid,scoring='r2',cv=3, verbose=1, n_jobs=-1)
grid.fit(X_Train,Y_Train)
print("Best Parameters : ",grid.best_params_)
Y_pred=grid.predict(X_TEST)
print("R2 Score : ", r2_score(Y_Test,Y_pred))
print("MAE Score : ",mean_absolute_error(Y_Test,Y_pred))
