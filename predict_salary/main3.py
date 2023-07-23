import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import math

data = pd.read_csv('ML/project/salary_data.csv')

x = data.iloc[:, :-1]
y = data.iloc[:, 1]

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0, test_size=1/3)

l_regression = LinearRegression()
l_regression.fit(x,y)

year = 5
salary_predict = l_regression.predict([[year]])
print(f'The Salary for {year} years is $ {math.floor(salary_predict[0])}')