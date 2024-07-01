import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor as rfr
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import warnings


def warn(*args, **kwargs):
    pass


warnings.warn = warn

try:
    with open('ins.dat', 'rb') as file:
        data = pickle.load(file)
except FileNotFoundError:
    data = 'https://raw.githubusercontent.com/4GeeksAcademy/linear-regression-project-tutorial/main/medical_insurance_cost.csv'
    data = pd.read_csv(data)
    with open('ins.dat', 'wb') as file:
        pickle.dump(data, file)
finally:
    data.drop_duplicates(inplace=True)

for i in data.columns:
    if data[i].nunique() == 2:
        data[i] = data[i].factorize()[0]
# sns.heatmap(data.drop('region', axis=1).corr(), annot=True, cbar=True, fmt='.2f')
# plt.show()
data = data.drop(['sex', 'children', 'region'], axis=1)

x = data.drop('charges', axis=1)
y = data['charges']
scaler = StandardScaler()

x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.2, random_state=42)
scaler.fit(x_tr)
x_tr_s = scaler.transform(x_tr)
x_tr_s = pd.DataFrame(x_tr_s, index=x_tr.index, columns=x_tr.columns)
x_te_s = scaler.transform(x_te)
x_te_s = pd.DataFrame(x_te_s, index=x_te.index, columns=x_te.columns)

model = LinearRegression()
model.fit(x_tr_s, y_tr)
predict = model.predict(x_te_s)
print('Mean Squared Error:', mean_squared_error(y_te, predict))
print('Root Mean Squared Error:', np.sqrt(mean_squared_error(y_te, predict)))
print('R**2 Score:', r2_score(y_te, predict))
clf = rfr()
clf.fit(x_tr_s, y_tr)
print('Random Forest Regression:', clf.score(x_te_s, y_te))