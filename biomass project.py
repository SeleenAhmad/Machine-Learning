import pandas as pd
from envs.qm1.DLLs.pyexpat import features

df=pd.read_csv(r"C:\Users\DELL\Downloads\pasture_biomass_10000.csv")
print(df.head())
print("\n_____Info_____")
print(df.info())
print("\n_____Describe_____")
print(df.describe())
print("\n_____missing details_____")
print(df.isnull().sum())
X=df[['rainfall','temperature','NDVI','soil_moisture']]
Y=df['biomass']
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import numpy as np
X_tr,X_tst,Y_tr,Y_tst=train_test_split(X,Y, test_size=.2, random_state=42)
print("training samples to X and Y :" , X_tr.shape[0],Y_tr.shape[0])
print("test samples to X and Y :" , X_tst.shape[0],Y_tst.shape[0])
print(X_tr.shape, Y_tr.shape)
print(X_tst.shape,Y_tst .shape)
l_model=LinearRegression()
l_model.fit(X_tr,Y_tr)
y_lin=l_model.predict(X_tst)
RF_model=RandomForestRegressor(n_estimators=200, max_depth=7,random_state=42)
RF_model.fit(X_tr,Y_tr)
Y_rf=RF_model.predict(X_tst)
XGB_model= XGBRegressor(objective='reg:squarederror',n_estimators=200, max_depth=5, learning_rate=.1, random_state=42)
XGB_model=XGB_model.fit(X_tr,Y_tr)
Y_xgb=XGB_model.predict(X_tst)
def evaluate_model(y_true, y_pred, model_name):
    r2=r2_score(y_true,y_pred)
    MAE=mean_absolute_error(y_true,y_pred)
    print(f"{model_name}, R2_score:{r2:.4f}" )
    print(f"{model_name}, MAE:{MAE:.4f}\n")
evaluate_model(Y_tst,y_lin,"linear Regression" )
evaluate_model(Y_tst,Y_rf, "Random Forest")
evaluate_model(Y_tst,Y_xgb, "XGBOOST")
features = ["rainfall","temperature","NDVI","soil_moisture"]
rf_importance= RF_model.feature_importances_
rf_df= pd.DataFrame({"Feature":features, "Importance":rf_importance}).sort_values(by="Importance", ascending=False)
xgb_importance= XGB_model.feature_importances_
xgb_df=pd.DataFrame({"Feature" : features, "importance" : xgb_importance}).sort_values(by="importance", ascending=False)
#the required plot of RF
plt.figure(figsize=(8,4))
plt.bar(rf_df["Feature"],rf_df["Importance"],color="skyblue")
plt.title("Random Forest Feature Importance")
plt.ylabel("Importance")
plt.show()
plt.figure(figsize=(8,4))
plt.bar(xgb_df["Feature"],xgb_df["importance"], color="salmon")
plt.title("XGB Feature Importance")
plt.ylabel("Importance")
plt.show()
