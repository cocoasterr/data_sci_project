import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data_csv = pd.read_csv('ML/project/salary_data.csv')

x = data_csv.iloc[:, :-1]
y = data_csv.iloc[:, 1]

#split data menjadi 2 bagian 

#1/3 atau 30% akan dijadikan data testing dan 70 % data akan di jadikan train
#random state 0 maksudnya data nya tetap, tidak random ketika dipanggil ulang
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1/3, random_state=0)

# masukan x tran dan y train sebagai data training
regression = LinearRegression()
regression.fit(x_train, y_train)

year_experiences = 5 
salary_predict = regression.predict([[year_experiences]])
print(f'The salary for {year_experiences} experience is : {salary_predict}')