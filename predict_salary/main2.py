import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

data = pd.read_csv('ML/project/salary_data.csv')

x = data.iloc[:, :-1]
y = data.iloc[:, 1]


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1/3, random_state=0)

regression = LinearRegression()

regression.fit(x_train, y_train)

year = 5
salary_predict = regression.predict([[year]])

print(f'The Salary for {year} years is {salary_predict[0]}')