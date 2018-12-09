from sklearn.datasets import load_boston
import matplotlib.pyplot as plt
import numpy as np

# 1. Boston Data Load
# 보스톤데이터를 sklearn 패키지 안에서 불러올수 있다. 그리고 dict로 되어있음.
boston = load_boston()
boston['data']
# dict는 각각 data/target/feature_names/DESCR(설명)/filename으로 되어있음.
boston

# 2. Data 정의
x_data = boston.data
y_data = boston.target.reshape(boston.target.size, 1)
# x_data는 data와 target은 506개의 row/ 13개의 feature로 되어있다.
# y_data는 506*1의 결과물있음.
x_data.shape
y_data.shape

# 3. Data Scaler
from sklearn import preprocessing
minmax_scale = preprocessing.MinMaxScaler(feature_range=(0,5)).fit(x_data)
# standard_scale = preprocessing.StandardScaler().fit(x_data)
minmax_scale
x_scaled_data = minmax_scale.transform(x_data)
x_scaled_data[:3]


# 4. Test/Train Data 나누기.
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x_scaled_data, y_data, test_size=0.33)
X_train.shape, X_test.shape, y_train.shape, y_test.shape

# 5. sklern의 linear regression 적용하기.
from sklearn import linear_model
regr = linear_model.LinearRegression(fit_intercept=True, normalize=False, copy_X=True, n_jobs=8)
regr.fit(X_train,y_train)
regr
regr.coef_, regr.intercept_ # 각 feature의 w와 intercept 출력

# # The coefficients
print('Coefficients: ', regr.coef_)
print('intercept: ', regr.intercept_)

regr.predict(x_data[:5]) # 5개를 predict함수로 예측하기. 아래는 직접 coef_와 intercept_의 식에 넣은것. 둘다 같음.
x_data[:5].dot(regr.coef_.T) + regr.intercept_


# 6. 검증하기.
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

y_true = y_test
y_hat = regr.predict(X_test)
r2_score(y_true, y_hat), mean_absolute_error(y_true, y_hat), mean_squared_error(y_true, y_hat)
y_true = y_train
y_hat = regr.predict(X_train)

r2_score(y_true, y_hat), mean_absolute_error(y_true, y_hat), mean_squared_error(y_true, y_hat)
