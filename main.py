from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
import pandas as pd
import joblib

dftest=pd.read_csv(r"C:\Users\hp\Downloads\archive\Weather Test Data.csv")
dftrain=pd.read_csv(r"C:\Users\hp\Downloads\archive\Weather Training Data.csv")

dftrain=dftrain.dropna()
features = [
    "MinTemp",
    "MaxTemp",
    "Rainfall",
    "WindSpeed9am",
    "WindSpeed3pm",
    "Humidity9am",
    "Humidity3pm",
    "Pressure9am",
    "Pressure3pm",
    "Cloud9am",
    "Cloud3pm",
    "Temp9am",
    "Temp3pm",
    "RainToday"
]
X = dftrain[features]
X["RainToday"]=X["RainToday"].map({"No":0,"Yes":1})
Y=dftrain["RainTomorrow"]
X_train, X_test, Y_train, Y_test= train_test_split(X,Y,test_size=0.2, random_state=42)
model=LogisticRegression(max_iter=10000)
model.fit(X_train,Y_train)
pred=model.predict_proba(X_test)
'''print(model.classes_)
print(pred[0])'''
rain=pred[0,1]*100
print(f"Probability of Rain Tomorrow: {rain:.2f}%")
loss=log_loss(Y_test,pred)
print(f"Log Loss of the model: {loss:.2f}")
joblib.dump(model, "rain_model.pkl")
joblib.dump(list(X.columns), "feature_columns.pkl")
