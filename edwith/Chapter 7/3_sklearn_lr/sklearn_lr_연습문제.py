# Linear Regression with Normal equation

%matplotlib inline
# 1. numpy, pandas, matplotlib의 pyplot(그래프) import하기

# 2. boston데이터셋을 sklearn에서 불러오고 boston에 인스턴스 할당.

# 3. 보스톤 데이타셋은 dict 형태. key들을 불러오자.

# 4. 위 키중 feature이름들이 있는 feature_names를 로드해보자.

# 5. 보스톤 데이타셋에서 data로 data를, 컬럼은 feature_names로 DataFrame을 만들자.
df = pd.DataFrame(boston.data, columns=boston.feature_names)

# 6. head 데이터 봐보자.

# 7. X는 values로, y는 target으로 지정하자. 모양을 확인해보자.(각각 506*13, 506벡터 이다.)

# 8. sklearn의 linear regression 불러오기.

# 9. lr_ne로 linear regreesion 불러오기. 절편 오키.
lr_ne = LinearRegression(fit_intercept=True)

# 10. train_test_split 모듈 불러오기.

# 11. 테스트셋 나누기. 테스트비율은 33%, random_states=0으로 -> 요것의 의미는 아직 모름.


# 12. train data로 피팅하기.

# 13. X_test로 예측한 결과값을 y_hat에 저장.

# 14. 10행까지 출력.

# 15. y_true에 y_test저장. 정확도 비교를 위함.

# 16. 정확도 확인을 위한 rmse를 구현(알고 있는식으로 그대로 구현)


# 17. 정확도 확인.sklearn의 mean_squared_error함수로.


# 18. y_true와 y_hat으로 scatter그래프 그리고 19. xlabel과 20. ylabel라벨 대충 붙여넣기
# 21. 타이틀 붙이기. 약간의 기호를 사용해서. 라벨때.Yi와 Y_hat을 기호로 표시해보자.
plt.scatter(y_true, y_hat, s=20)
plt.xlabel("Prices: $Y_i$")
plt.ylabel("Predicted prices: $\hat{Y}_i$")
plt.title("Prices vs Predicted prices: $Y_i$ vs $\hat{Y}_i$")

# 22. lr_ned의 w값들을 표시해바
lr_ne.coef_
# 23. feature의 이름들을 출력해보자. 참고로 boston데이터안에 key로 존재함.
boston.feature_names

# Linear Regression with SGD
# 24. SGD regressor 로드

# 25. lr_SGD에 해당 SGDRegressor 인스턴스 할당


# 26. sklearn의 standard scaler 로드

# 27. std_scaler에 스케일러 할당

# 28. 위의 것으로 X 어레이 피팅.

# 29. std_scaler를 이용해서 스케일된 데이터를 X_scaled에 할당.

# 30. train_test_split 모듈 로드

# 31. train, test 나누기. test 비율 0.33, random_state=42로할 것,

# 32. train data로 fit.

# 33. y_hat에 test 데이터로 예측한 값들을 할당.

# 34. y_true에 실제 test결과값 할당

# 35. mse에 mean_squared_error로 예측값과 실제값 평가결과 비교.

# 36. rmse를 식을 세워서 예측값과 실제값 비교

# 37. 위의 두개 평가결과 출력

# 38. 위의 결과를 scatter로 plot하되 size 10으로. 39. 40. xlabel과 ylabel 삽입. 41. title도


# Linear Regression with Ridge & Lasso regression
# 39. 라쏘, 릿지 모듈불러오기

# 40. X_train, X_test, y_train, y_test 데이타셋 나누기, 테스트크기 33%, 랜덤스테이트 42

# 41. 릿지 regression 하기. 절편 있고 alpha 0.5(?)

# 42. 트레인 데이터로 피팅하기.

# 43. ridge로 predict해서 y_hat에 결과값 저장.

# 44. y_test를 y_true에 할당

# 45. sklearn 내장 mse함수써서 mse에 저장.

# 46. rmse를 식에 그대로 대입해서 rmse에 저장.

# 47. 출력


# 48. scatter graph로 출력. xlabel, ylabel 지정하고 title넣기.





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
