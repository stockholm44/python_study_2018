# Linear Regression with Normal equation

%matplotlib inline
# numpy, pandas, matplotlib의 pyplot(그래프) import하기
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# boston데이터셋을 sklearn에서 불러오기.
from sklearn.datasets import load_boston
boston = load_boston()

# 보스톤 데이타셋은 dict 형태. key들을 불러오자.
boston.keys()
#위 키중 feature이름들이 있는 feature_names를 로드해보자.
boston.feature_names
# 보스토 데이타셋에서 data로 data를, 컬럼은 feature_names로 DataFrame을 만들자.
df = pd.DataFrame(boston.data, columns=boston.feature_names)
df.head()
# X는 values로, y는 target으로 지정하자. 각각 506*13, 506벡터 이다.
X = df.values
y = boston.target
X.shape
y.shape

# sklearn의 linear regression 불러오기.
from sklearn.linear_model import LinearRegression

# lr_ne로 linear regreesion 불러오기. 절편 오키.
lr_ne = LinearRegression(fit_intercept=True)

from sklearn.model_selection import train_test_split
# 테스트셋 나누기.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
# train data로 피팅하기.
lr_ne.fit(X_train,y_train)
# 위 결과 y_hat에 저장.
y_hat = lr_ne.predict(X_test)
y_hat[:10]
# y_true에 y_test저장. 정확도 비교를 위함.
y_true = y_test

# 정확도 확인.rmse를 식으로
rmse = np.sqrt((((y_hat-y_true)**2).sum() / len(y_true)))
rmse

# 정확도 확인.sklearn의 mean_squared_error함수로.
import sklearn
mse = sklearn.metrics.mean_squared_error(y_hat, y_true)
mse

# y_true와 y_hat으로 scatter그래프 그리고 라벨 대충 붙여넣기
plt.scatter(y_true, y_hat, s=10)
plt.xlabel("Prices: $Y_i$")
plt.ylabel("Predicted prices: $\hat{Y}_i$")
plt.title("Prices vs Predicted prices: $Y_i$ vs $\hat{Y}_i$")

lr_ne.coef_
boston.feature_names

# Linear Regression with SGD
# SGD regressor 로드
from sklearn.linear_model import SGDRegressor
lr_SGD = SGDRegressor()

# sklearn의 standard scaler 로드
from sklearn.preprocessing import StandardScaler
# std_scaler에 스케일러 할당
std_scaler = StandardScaler()
# X 어레이 피팅.
std_scaler.fit(X)
# ??
X_scaled = std_scaler.transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

lr_SGD.fit(X_train, y_train)
y_hat = lr_SGD.predict(X_test)
y_true = y_test
mse = sklearn.metrics.mean_squared_error(y_hat, y_true)
rmse = np.sqrt((((y_hat - y_true)**2).sum() / len(y_true)))
rmse, mse

plt.scatter(y_true, y_hat, s=10)
plt.xlabel("Prices: $Y_i$")
plt.ylabel("Predicted prices: $\hat{Y}_i$")
plt.title


# Linear Regression with Ridge & Lasso regression
# 1. 라쏘, 릿지 모듈불러오기
from sklearn.linear_model import Lasso, Ridge
# X_train, X_test, y_train, y_test 데이타셋 나누기, 테스트크기 33%, 랜덤스테이트 42
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
# 릿지 regression 하기. 절편 있고 alpha 0.5(?)
ridge = Ridge(fit_intercept=True, alpha=0.5)
# 트레인 데이터로 피팅하기.
ridge.fit(X_train,y_train)
# ridge로 predict해서 y_hat에 결과값 저장.
y_hat = ridge.predict(X_test)
y_true = y_test
# sklearn 내장 mse함수써서 mse에 저장.
mse = sklearn.metrics.mean_squared_error(y_hat, y_true)
# rmse를 식에 그대로 대입해서 rmse에 저장.
rmse = np.sqrt((((y_hat - y_true)**2).sum() / len(y_true)))
# 출력
rmse, mse

# scatter graph로 출력. xlabel, ylabel 지정하고 title넣기.
plt.scatter(y_true, y_hat, s=10)
plt.xlabel("Prices: $Y_i$")
plt.ylabel("Predicted prices: $\hat{Y}_i$")
plt.title("Prices vs Predicted prices: $Y_i$ vs $\hat{Y}_i$")

# # 3. Kfold??
# from sklearn.model_selection import KFold
#
# print('Ridge Regression')
# print('alpha\t RMSE_train\t RMSE_10cv\n')
# alpha = np.linspace(.01,20,50)
# t_rmse = np.array([])
# cv_rmse = np.array([])
#
# for a in alpha:
#     ridge = Ridge(fit_intercept=True, alpha=a)
#
#     # computing the RMSE on training data
#     ridge.fit(X_train,y_train)
#     p = ridge.predict(X_test)
#     err = p-y_test
#     total_error = np.dot(err,err)
#     rmse_train = np.sqrt(total_error/len(p))
#
#     # computing RMSE using 10-fold cross validation
#     kf = KFold(10)
#     xval_err = 0
#     for train, test in kf.split(X):
#         ridge.fit(X[train], y[train])
#         p = ridge.predict(X[test])
#         err = p - y[test]
#         xval_err += np.dot(err,err)
#     rmse_10cv = np.sqrt(xval_err/len(X))
#
#     t_rmse = np.append(t_rmse, [rmse_train])
#     cv_rmse = np.append(cv_rmse, [rmse_10cv])
#     print('{:.3f}\t {:.4f}\t\t {:.4f}'.format(a,rmse_train,rmse_10cv))
#
#
# plt.plot(alpha, t_rmse, label='RMSE-Train')
# plt.plot(alpha, cv_rmse, label='RMSE_XVal')
# plt.legend( ('RMSE-Train', 'RMSE_XVal') )
# plt.ylabel('RMSE')
# plt.xlabel('Alpha')
# plt.show()
#
# a = 0.3
# for name,met in [
#         ('linear regression', LinearRegression()),
#         ('lasso', Lasso(fit_intercept=True, alpha=a)),
#         ('ridge', Ridge(fit_intercept=True, alpha=a)),
#         ]:
#     met.fit(X_train,y_train)
#     # p = np.array([met.predict(xi) for xi in x])
#     p = met.predict(X_test)
#     e = p-y_test
#     total_error = np.dot(e,e)
#     rmse_train = np.sqrt(total_error/len(p))
#
#     kf = KFold(10)
#     err = 0
#     for train,test in kf.split(X):
#         met.fit(X[train],y[train])
#         p = met.predict(X[test])
#         e = p-y[test]
#         err += np.dot(e,e)
#
#     rmse_10cv = np.sqrt(err/len(X))
#     print('Method: %s' %name)
#     print('RMSE on training: %.4f' %rmse_train)
#     print('RMSE on 10-fold CV: %.4f' %rmse_10cv)
#
