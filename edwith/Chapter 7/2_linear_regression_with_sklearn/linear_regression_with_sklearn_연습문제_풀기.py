# 1.보스톤 데이터, 2. pyplot 데이터, 3. numpy 데이터 로드
from sklearn.datasets import load_boston
import matplotlib.pyplot as plt
import numpy as np

# 1) Boston Data Load
# 보스톤데이터를 sklearn 패키지 안에서 불러올수 있다. 그리고 dict로 되어있음.
# 4.boston 변수에 보스톤데이터함수 할당.5. Boston dict에 data를 key로 한번 확인.
boston = load_boston()
boston["data"]
# dict는 각각 data/target/feature_names/DESCR(설명)/filename으로 되어있음.





# 2) Data 정의
# 5. x_data에 data, 6. y_data에 타겟을 사이즈 * 1로 reshape
x_data = boston["data"]
y_data = boston["target"].reshape(-1, 1)
# x_data는 data와 target은 506개의 row/ 13개의 feature로 되어있다.
# y_data는 506*1의 결과물있음.
# 6,7 x_data와 y_date 모양체크
x_data.shape, y_data.shape





# 3) Data Scaler
# 8. sklearn에서 preprocessing 열기. (목적은 minmax 스케일러.)
from sklearn import preprocessing
# 9.minmax_scale에 minmax스케일러를 할당. feature range 0,5, 하고 그줄에서 바로 x_data를 피팅.
minmax_scale = preprocessing.MinMaxScaler(feature_range=(0,5)).fit(x_data) ###### feature_range????
minmax_scale
# 참고 MinMaxScaler의 Method
'''
Methods

fit(X[, y])	Compute the minimum and maximum to be used for later scaling.
fit_transform(X[, y])	Fit to data, then transform it.
get_params([deep])	Get parameters for this estimator.
inverse_transform(X)	Undo the scaling of X according to feature_range.
partial_fit(X[, y])	Online computation of min and max on X for later scaling.
set_params(**params)	Set the parameters of this estimator.
transform(X)	Scaling features of X according to feature_range.
'''

## standard_scale = preprocessing.StandardScaler().fit(x_data)
# 10.minmax scale 확인.
minmax_scale

# 11.transform 해서 스케일 데이터를 x_scaled_data에 할당.
x_scaled_data = minmax_scale.transform(x_data)
x_data
x_scaled_data

# 12.3줄까지 보여줘라.
x_scaled_data[:3]





# 4) Test/Train Data 나누기.
# 13. train_test_split 함수 로드
from sklearn.model_selection import train_test_split

# 14. X, y에 트레인이랑 테스트 각각 나눠 할당하기.
X_train, X_test, y_train, y_test = train_test_split(x_scaled_data, y_data, test_size=0.33)


X_test[:5]





# 5) sklern의 linear regression 적용하기.
# 15. linear model 로드하기
from sklearn import linear_model
# 16. regr에다가 저위에 로드한 것을 이용해서 LinearRegression 할당. 인터셉트 있고 노말라이즈 없고, copy_X있고, n_jobs 8개
regr = linear_model.LinearRegression(fit_intercept=True, normalize=False, copy_X=True, n_jobs=8) ##### n_jobs???


 ''' 참고 n_jobs
# n_jobs : int or None, optional (default=None)
# The number of jobs to use for the computation. This will only provide speedup for n_targets > 1
# and sufficient large problems. None means 1 unless in a joblib.parallel_backend context.
# -1 means using all processors. See Glossary for more details.
 '''


# 17. train data로 피팅.
regr.fit(X_train, y_train)
regr
# 18.각각의 feature와 coef 출력해보기.
print("coef: ", regr.coef_)
print("intercept: ", regr.intercept_)
# # The coefficients

# 상위 5개만 결과 예측하기.
# 19. 1) predict함수로 예측
X_test.shape
regr.coef_.shape
regr.intercept_.shape
regr.predict(X_test)[:5]
# 20. 2) 아래는 직접 coef_와 intercept_를 활용하여 식을 써서 도출
(X_test.dot(regr.coef_.T) + regr.intercept_)[:5]


# 6) 검증하기.
# 21. r2_score, mean_absolute_error, mean_squared_error 로들하기
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

# 22.y_ture를 테스트데이터롤 할당.
y_true = y_test
y_true[:10]
# 23.y_hat은 예측데이터를 할당.
y_hat = regr.predict(X_test)
y_hat[:10]
# 24.r2_score, mean_absolute_error, mean_squared_error 로 y_true와 y_hat데이터 검증
r2_score(y_true, y_hat)
mean_absolute_error(y_true, y_hat)
mean_squared_error(y_true, y_hat)

# 25. 위와 동일하게 y_train에 대해서도 해보자. 트레이닝 자체를 y_train으로 했으니 아마 더 잘 맞을 듯.
y_true = y_train
# 26. y_hat은 예측데이터를 할당.
y_hat = regr.predict(X_train)
# 27.r2_score, mean_absolute_error, mean_squared_error 로 y_true와 y_hat데이터 검증
r2_score(y_true, y_hat)
mean_absolute_error(y_true, y_hat)
mean_squared_error(y_true, y_hat)


''' r2_score
R^2 (coefficient of determination) regression score function.

Best possible score is 1.0 and it can be negative (because the model can be arbitrarily worse).
A constant model that always predicts the expected value of y,
disregarding the input features, would get a R^2 score of 0.0.
'''

''' mean_absolute_error
https://scikit-learn.org/stable/modules/model_evaluation.html#mean-absolute-error
'''

''' mean_squared_error
https://scikit-learn.org/stable/modules/model_evaluation.html#mean_squared_error
'''
