# Linear regression with one variable, implemented by numpy.
# 1. pyplot, pandas, numpy 로드

# 2. %matplotlib inline ?? 입력





# 1) LOAD DATASET
# 3. xls 데이터 불려들어오기.  C:/study_2018/python_study_2018/edwith/Chapter 7/1_linear_regression/slr06.xls

# 4. head데이터 로드


# 5. raw_X에 X컬럼을 1개짜리 컬럼으로에서 array로써 변수할당.


# 6. Y 컬럼을 y에 할당


# 한번 plot해보자.
# 7. figsize 10,5로 figure만들기
# 8. raw_X와 y로 plot, o모양, 투명도 0.5(alpha써라.)
# 9. raw_X와 y 5행까지 로드
# 10. raw_X의 길이 * 1만큼 one행렬만들어주기. 하되 3줄까지만 출력.(W0의 계산을 위함.)
# 11. raw_X행렬의 0 컬럼에 위의 10.에서 만든 one 행렬 붙여주되 열로(열방향) 붙이고 X이름으로 변수로 저장.
# 12. X의 5줄까지 출력

# 첫 W벡터를 normal(가우시안) distribution범위로 2와 1을 랜덤숫자로 만들기.(W0, W1)
# Draw random samples from a normal (Gaussian) distribution.
# numpy.random.normal(loc=0.0, scale=1.0, size=None)
# 13. W벡터를 가우시안 dist로 2*1로 만들기. 14. 출력
# 15. 10*5로 plot을 하기전 figure만들기
# 16. X와 w를 dot product 해서 y_predict에 할당
# 17. raw_X, y로 plot하되 o모양으로 투명도 0.5
# 18. raw_X와 y_predict로 plot







# 2) HYPOTHESIS AND COST FUNCTION
# 19 hypothesis_function 함수 만들기. 2개 array받아서 dot product하는 .
# 20. cost_function 함수만들기. return 값은 (1/2*갯수) * 시그마(h-y)^2
# 21. h에 hypothesis_function 할당.
# 22. cost_function 실행하되 위에서 만든 h와 y로 해보자.






# 3) GRADIENT DESCENT  
# 23. gradient_descent 함수 만들기. 변수 X, y, w, alpha, iterations
# return 값은 theta (최종값), theta_list, cost_list,-->list 두개는 10번째 반복시마다 리스트에 append하게.


# DO Linear regression with GD
# 24. iteration 10000, alpha 0.001로 지정.

# 25 theta, theta_list, cost_list를 gradient_descent로 받기.변수는 변수들그대로,

# 26. cost에 cost_function으로 만든걸로 값넣기.

# 27. theta와 cost 프린트하기.

# 28. theta_list 5줄 출력
# 29. theta_list 어레이화하고 30. 모양보기. 31. X의 모양은?

# 30. figure 만들기. 사이즈는 10*5
# 31. 이건 왜있는지 모르겠다.. X와 theta_list를 dot-product해서

# 32. y_predict_step 출력.
# 아. 쉐입을 출력해보니 이해가능.
# 1001번반복한 결과들이 한번 하면 행당 1001개 생성.
# 63은? 기존 x_data들에 대한 예측치가 어떻게 변했는지를 보여주는 것.


# 33. plot 하자. 기존X데이터와 y로 o모양ㅇ로 투명도 0.5

# 34. 범례넣기.33이랑 같이 아래 반복문으로 fitting한 그래프들을 한번에 그리기.

# 35. cost_list를 이용해서 cost가 어떻게 줄어드는지 graph를 그리기.
