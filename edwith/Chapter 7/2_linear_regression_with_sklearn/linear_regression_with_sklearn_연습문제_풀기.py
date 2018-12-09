# 1.보스톤 데이터, 2. pyplot 데이터, 3. numpy 데이터 로드



# 1) Boston Data Load
# 보스톤데이터를 sklearn 패키지 안에서 불러올수 있다. 그리고 dict로 되어있음.
# 4.boston 변수에 보스톤데이터함수 할당.5. Boston dict에 data를 key로 한번 확인.

# dict는 각각 data/target/feature_names/DESCR(설명)/filename으로 되어있음.





# 2) Data 정의
# 5. x_data에 data, 6. y_data에 타겟을 사이즈 * 1로 reshape

# x_data는 data와 target은 506개의 row/ 13개의 feature로 되어있다.
# y_data는 506*1의 결과물있음.
# 6,7 x_data와 y_date 모양체크






# 3) Data Scaler
# 8. sklearn에서 preprocessing 열기. (목적은 minmax 스케일러.)

# 9.minmax_scale에 minmax스케일러를 할당. feature range 0.5, 하고 그줄에서 바로 x_data를 피팅.

## standard_scale = preprocessing.StandardScaler().fit(x_data)
# 10.minmax scale 확인.

# 11.transform 해서 스케일 데이터를 x_scaled_data에 할당.

# 12.3줄까지 보여줘라.






# 4) Test/Train Data 나누기.
# 13. train_test_split 함수 로드


# 14. X, y에 트레인이랑 테스트 각각 나눠 할당하기.









# 5) sklern의 linear regression 적용하기.
# 15. linear model 로드하기

# 16. regr에다가 저위에 로드한 것을 이용해서 LinearRegression 할당. 인터셉트 있고 노말라이즈 없고, copy_X있고, n_jobs 8개

# 17. train data로 피팅.

# 18.각각의 feature와 coef 출력해보기.

# # The coefficients

# 상위 5개만 결과 예측하기.
# 19. 1) predict함수로 예측
# 20. 2) 아래는 직접 coef_와 intercept_를 활용하여 식을 써서 도출






# 6) 검증하기.
# 21. r2_score, mean_absolute_error, mean_squared_error 로들하기


# 22.y_ture를 테스트데이터롤 할당.

# 23.y_hat은 예측데이터를 할당.

# 24.r2_score, mean_absolute_error, mean_squared_error 로 y_true와 y_hat데이터 검증


# 25. 위와 동일하게 y_train에 대해서도 해보자. 트레이닝 자체를 y_train으로 했으니 아마 더 잘 맞을 듯.

# 26. y_hat은 예측데이터를 할당.

# 27.r2_score, mean_absolute_error, mean_squared_error 로 y_true와 y_hat데이터 검증
