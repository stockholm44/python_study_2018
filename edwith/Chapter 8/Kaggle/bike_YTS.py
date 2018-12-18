import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

test_df = pd.read_csv("C:/study_2018/python_study_2018/edwith/Chapter 8/Kaggle/data/test.csv", parse_dates=["datetime"])
train_df = pd.read_csv("C:/study_2018/python_study_2018/edwith/Chapter 8/Kaggle/data/train.csv", parse_dates=["datetime"])

all_df = pd.concat((test_df, train_df), axis=0).reset_index()
all_df.head(2)

all_df.tail(2)
train_index = list(range(len(train_df)))
test_index = list(range(len(test_df)))

all_df.isnull().sum()

x = np.array([np.inf, -np.inf, np.nan, -128, 128])
x
np.nan_to_num(x)
def rmsle(y, y_):
    log1 = np.nan_to_num(np.log(y+1))
    log2 = np.nan_to_num(np.log(y_+1))
    calc = (log1-log2) ** 2
    return np.sqrt(np.mean(calc))

submission_df = pd.read_csv("C:/study_2018/python_study_2018/edwith/Chapter 8/Kaggle/data/sampleSubmission.csv")
submission_df.head()

rmsle(submission_df["count"].values, np.random.randint(0,100, size=len(submission_df)))

del all_df["casual"]
del all_df["registered"]
del all_df["index"]

pre_df = all_df.merge(pd.get_dummies(all_df["season"], prefix="season"), left_index=True, right_index=True)
pre_df.head()
pre_df = pre_df.merge(pd.get_dummies(all_df["weather"], prefix="weather"), left_index=True, right_index=True)
pre_df.head()

pre_df["datetime"].unique()
pre_df["year"] = pre_df["datetime"].dt.year
pre_df["month"] = pre_df["datetime"].dt.month
pre_df["day"] = pre_df["datetime"].dt.day
pre_df["hour"] = pre_df["datetime"].dt.hour
pre_df["weekday"] = pre_df["datetime"].dt.weekday

pre_df = pre_df.merge(pd.get_dummies(pre_df["weekday"], prefix="weekday"), left_index=True, right_index=True)
pre_df.head()
pre_df.dtypes
# 아래목록들은 type을 category로 만들어. 반복문없이 가능하지않나?
category_variable_list = ["season","weather","workingday","season_1","season_2","season_3","season_4","weather_1","weather_2","weather_3","weather_4","year","month","day","hour","weekday","weekday_0","weekday_1","weekday_2","weekday_3","weekday_4","weekday_5","weekday_6"]
for var_name in category_variable_list:
    pre_df[var_name] = pre_df[var_name].astype("category")

pre_df.dtypes
# train_index에 있는 pre_df의 정보들만 train_df로 뽑아.
train_df = pre_df.iloc[train_index]
train_df.head(2)
# 3*3 총 9개의 bar차트를 각각 count를 x축으로 그린다.
fig, axes = plt.subplots(nrows=3,ncols=3)
fig.set_size_inches(12,5)
axes[0][0].bar(train_df["year"], train_df["count"])
axes[0][1].bar(train_df["weather"], train_df["count"])
axes[0][2].bar(train_df["workingday"], train_df["count"])
axes[1][0].bar(train_df["holiday"], train_df["count"])
axes[1][1].bar(train_df["weekday"], train_df["count"])
axes[1][2].bar(train_df["month"], train_df["count"])
axes[2][0].bar(train_df["day"], train_df["count"])
axes[2][1].bar(train_df["hour"], train_df["count"])
plt.show()

# 달별로 groupby해서 평균보자. 근데 왜 6달(0~5)까지만 보는지??
series_data = train_df.groupby(["month"])["count"].mean()
series_data.index.tolist()[:5]
series_data

fig, axes = plt.subplots()
axes.bar(range(len(series_data)), series_data)
fig.set_size_inches(12,5)
plt.show()

# seaborn이라고 하는 데이터 분포 시각화 패키지 로드.
import seaborn as sn

fig,(ax1,ax2,ax3) = plt.subplots(ncols=3)
fig.set_size_inches(12, 5)
sn.regplot(x="temp", y="count", data=train_df,ax=ax1)
sn.regplot(x="windspeed", y="count", data=train_df,ax=ax2)
sn.regplot(x="humidity", y="count", data=train_df,ax=ax3)
plt.show()

category_variable_list

# 여기부터는 아직 못봄.
# 아래 5개 칼람들의 서로의 상관계수 확인. corr()함수는 상관관계를 DataFrame으로 return 함.
corrMatt = train_df[["temp","atemp","humidity","windspeed","count"]].corr()
corrMatt
mask = np.array(corrMatt)
mask[np.tril_indices_from(mask)] = False
fig,ax= plt.subplots()
fig.set_size_inches(20,10)
sn.heatmap(corrMatt, mask=mask,vmax=.8, square=True,annot=True)
plt.show()
category_variable_list[:5]
continuous_variable_list = ["temp","humidity","windspeed","atemp"]

season_list = ['season_1', 'season_2', 'season_3', 'season_4']
weather_list = ['weather_1', 'weather_2', 'weather_3', 'weather_4']
weekday_list = ['weekday_0','weekday_1','weekday_2','weekday_3','weekday_4','weekday_5','weekday_6']

category_varialbe_list = ["season","holiday","workingday","weather","weekday","month","year","hour"]

all_variable_list = continuous_variable_list + category_varialbe_list

all_variable_list.append(season_list)
all_variable_list.append(weather_list)
all_variable_list.append(weekday_list)

all_variable_list

number_of_variables = len(all_variable_list)
number_of_variables

variable_combinations = []

# itertools의 combination관련 method.
# Combinatoric generators:
#
# Iterator	Arguments	Results
# product()	p, q, … [repeat=1]	cartesian product, equivalent to a nested for-loop
# permutations()	p[, r]	r-length tuples, all possible orderings, no repeated elements
# combinations()	p, r	r-length tuples, in sorted order, no repeated elements
# combinations_with_replacement()	p, r	r-length tuples, in sorted order, with repeated elements
# 예제
# product('ABCD', repeat=2)	 	AA AB AC AD BA BB BC BD CA CB CC CD DA DB DC DD
# permutations('ABCD', 2)	 	AB AC AD BA BC BD CA CB CD DA DB DC
# combinations('ABCD', 2)	 	AB AC AD BC BD CD
# combinations_with_replacement('ABCD', 2)	 	AA AB AC AD BB BC BD CC CD DD

import itertools
for L in range(8, number_of_variables+1):
    for subset in itertools.combinations(all_variable_list, L):
        temp = []
        for variable in subset:
            if isinstance(variable, list): # 이건 예를들면 week가 들어갈떄 week0~6이 다 들어가게 꾸민 트릭이라는데 뭔소린지..->이제 알겠다. combination의 갯수에는 week만 포함되고 만약 week가 있다면 category data니까 나머지들을 한꺼번에 등록시키는 기능.임.
                for value in variable:
                    temp.append(value)
            else:
                temp.append(variable)
        variable_combinations.append(temp)

len(variable_combinations)

del pre_df["count"]


# 여기서부터 linear_model 적용해보자.
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import datetime

# KFold 클래스로 kf 인스턴스만듬.
kf= KFold(n_splits=10)

# 최종결과물 제출을 위한 준비
y = train_df["count"].values
final_output = []
models = []

print(len(variable_combinations))
ts = datetime.datetime.now()   # 걸리는 시간 계산을 위한 시작시간 설정

# # 아래 반복문 테스트용 임시 코드
# lr = LinearRegression(n_jobs=8)
# target_df = pre_df[['temp', 'humidity', 'windspeed']]
# ALL = target_df.values
# std = StandardScaler()
# std.fit(ALL)
# ALL_scaled = std.transform(ALL)
# X = ALL_scaled[train_index]
# X
for i, combination in enumerate(variable_combinations):
    lr = LinearRegression(n_jobs=8)
    ridge = Ridge()
    lasso = Lasso()

    lr_result = []
    ridge_result = []
    lasso_result = []

    target_df = pre_df[combination]
    ALL = target_df.values
    std = StandardScaler()
    std.fit(ALL)
    ALL_scaled = std.transform(ALL)
    X = ALL_scaled[train_index]

    for train_data_index, test_data_index in kf.split(X):
        X_train = X[train_data_index]
        X_test = X[test_data_index]
        y_train = y[train_data_index]
        y_test = y[test_data_index]

        lr.fit(X_train, y_train)
        result = rmsle(y_test, lr.predict(X_test))
        lr_result.append(result)

        ridge.fit(X_train, y_train)
        result = rmsle(y_test, ridge.predict(X_test))
        ridge_result.append(result)

        lasso.fit(X_train, y_train)
        result = rmsle(y_test, lasso.predict(X_test))
        lasso_result.append(result)

    final_output.append([i, np.mean(lr_result), np.mean(ridge_result), np.mean(lasso_result)])
    models.append([lr, ridge, lasso])
    if i % 100 == 0:
        tf = datetime.datetime.now()
        te = tf - ts
        print(i, te)
        ts = datetime.datetime.now()
        
