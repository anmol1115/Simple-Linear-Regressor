import pandas as pd
X_train=pd.read_csv('C:\\Users\Anmol\Desktop\ML Masters\Linear_X_train.csv')
X_test=pd.read_csv('C:\\Users\Anmol\Desktop\ML Masters\Linear_X_test.csv')
y_train=pd.read_csv('C:\\Users\Anmol\Desktop\ML Masters\Linear_Y_train.csv')

from sklearn.linear_model import LinearRegression
linreg=LinearRegression()
linreg.fit(X_train,y_train)
y_pred=linreg.predict(X_test)
