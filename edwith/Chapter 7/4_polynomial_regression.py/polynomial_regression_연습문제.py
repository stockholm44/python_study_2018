# '''이번 강의에서는 Regression 모델 중 X, Y 의 관계가 곡선 형태(비선형)일 경우에 사용할 수 있는 Polynomial
# Regression에 대한 개념과, scikit-learn 모듈을 활용해서 polynomial regression 을 수행하는 방법에 대해서
# 공부합니다.'''

# 1. numpy, pyplot 로드


# 2. f함수 작성. parameter는 size, retrun은 (x,y)
# x 는 0, 5사이의 size로 등간격.
# y 는 y = sin(x) +1


# 3. sample 함수 작성. 위와 동일하나 y에 sin(x) + 1 + x크기의 0.5배에 해당하는 standard distribution으로 rand number 더하기.



# np.linspace
# """
# Return evenly spaced numbers over a specified interval.
# Returns num evenly spaced samples, calculated over the interval [start, stop].
# The endpoint of the interval can optionally be excluded.
# """

# numpy.random.rand(d0, d1, ..., dn)
# Random values in a given shape.
# Create an array of the given shape and populate it with random samples from a uniform distribution over [0, 1)

# numpy.random.randn(d0, d1, ..., dn)
# Return a sample (or samples) from the “standard normal” distribution.
# If positive, int_like or int-convertible arguments are provided, randn generates an array of shape (d0, d1, ..., dn), filled with random floats sampled from a univariate “normal” (Gaussian) distribution of mean 0 and variance 1 (if any of the d_i are floats, they are first converted to integers by truncation). A single float randomly sampled from the distribution is returned if no argument is provided.
# This is a convenience function. If you want an interface that takes a tuple as the first argument, use numpy.random.standard_normal instead.

# 4. f_x, f_y에 f함수로 size 1000 결과값저장.
# 5. 해당 것을 plot
# 6. X, y에 sample함수 size는 1000으로 저장.
# 7. 6번도 scatter로 size는 3, color는 검정으로 plt
# 8. 한꺼번에 plot들 보여줘.


# 9. X, y 모냥함 보자. -> 현재는 벡터다.
# 10, 11. X, y를 각각 array로 바꾸자.-> 1열짜리로 바꿔
# 12. X, y 모냥확인.




# 13. sklearn의 LinearRegression 로드


# 14. lr에 LinearRegression 인스턴스 할당
# 15. X, y로 피팅



# 이번엔 LR사용한것과 기존 plot들을 비교해보자.

# 16. f_x, f_y에 f함수로 size 1000 결과값저장.
# 17. 해당 것을 plot
# 18. X와 y를 평탄화시킨것을 scatter로 plt. size는 3, color는 검정으로 plt.
# 19. X평탄화와 아까 lr로 predict한 결과도 평탄화해서 plot.
# 20. 한꺼번에 show



# 이번에는 폴리노미얼로 해보자.
# sklearn.preprocessing.PolynomialFeatures
# 1차 방정식을 고차다항식으로 변경하는 기법.
# 첫컬럼은 1, 그 뒤들은 원래 것들 그다음은 교차로.
# 예를들면
# y = x1 + x2
# -> y = x0 + x1 + x2 +  x1^2 + x1*x2 + x2^2

# 21. PolynomialFeatures 로드


# 22. poly_features에  PolynomialFeatures를 degree 2로 인스턴스 생성
# 23. X_poly에 poly_features 인스턴스를 활용하여 fit_transform 해라.
# 24. X_poly를 10줄까지 출력


# 25. lr에 LR 인스턴스할당.
# 26. X_poly와 y로 fit


# 27, 28. 모양확인. X랑 X_poly


# 29. f_x, f_y에 f함수로 size 1000 결과값저장.
# 30. 해당 것을 plot
# 31. X와 y를 평탄화시킨것을 scatter로 plt. size는 3, color는 검정으로 plt.
# 32. X평탄화와 아까 lr로 X_poly predict한 결과도 평탄화해서 plot.
# 33. 한꺼번에 show


# 34. PolynomialFeatures인스턴스를 poly_features에 할당하되 degree 16으로 바짝늘려보자.
# 35. X_poly에 fit_transform
# 36. X_poly 2개행 출력



# 37. lr인스턴스 할당. 38. fit 39. f_x, f_y 1000으로 f 출력. 40.f_x, f_y plot. 41 scatter로 X, y를 flatten해서 plot
# 42. X랑 X_poly를 predict해서 plot 해서 위와 한꺼번에 show



# 숫자를 늘리면 늘릴수록 되긴하니까 3가지의 regressiong으로 검증해보자.
# 제일 좋은 건 rmse가 최소화되는 조건으로 한다.
# 43. rmse 함수 만들기. 인수는 predictions, targets. return은 (pre-tar)^2의 평균의 루트.


# 44. 10~49의 숫자들 list화해서 poly_range에 할당.
# 45, 46, 47. rmse_lr_list rmse_lasso_list, rmse_ridge_list를 각각 빈 리스트로 할당.



# 48,49. Lasso와 Ridge 로드


# 50. 반복문작성. poly_range내에서 각각 3개의 회귀 모델을 반복시키고 각각의 rmse를 각각의 list에 append.


# 51. pd와 DataFrame 로드


# 52. data라는 dict를 만들되 poly_range lr_rmse lasso_rmse ridge_rmse에 적절한 list들을 할당하기.


# 53. df를 data를 이용하여 만들고 index는 poly_range로 하기.


# 54~56. 각각의 rmse 3개를 poly_range를 x축으로 하여 plot하고 label도 넣기.
# 57. 위의 것을 show



# 58. 각각의 rmse의 min을 출력(한번에.)


# 59. 위의 각각의 최소값을 보면 일단 ridge_rmse의 최소값이 전체 최소값이다. ridge_rmse 칼람을 sort하고 head출력


# 60. 위에서 나온 poly_range로 PolynomialFeatures 인스턴스 다시 만들고
# 61. X_poly에 fit_tranform해서 저장.
# 62. Ridge 인스턴스 ridge에 할당
# 63. fit



# 64. f_x, f_y에 f함수로 size=1000 결과값 저장.
# 65. plot
# 66. scatter로 X, y 의 flatten plot
# 67. X와 ridge의 predict값을 plot
# 68. 한꺼번에 show
