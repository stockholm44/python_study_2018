import pandas as pd

filename = 'C:/study_2018/python_study_2018/edwith/Chapter 6/Case Study#1/AirPassengers.csv'
df_time_series = pd.read_csv(filename)
df_time_series.head()

# 누적합을 위함인듯. step 컬럼만들어서 각 컬럼의 길이까지의 정수를 반환
df_time_series["step"] = range(len(df_time_series))

# 누적계산을 위한 cumsum과 누적 최대인원/최소인원 컬럼을 추가.
df_time_series["cum_sum"] = df_time_series["#Passengers"].cumsum()
df_time_series["cum_max"] = df_time_series["#Passengers"].cummax()
df_time_series["cum_min"] = df_time_series["#Passengers"].cummin()
df_time_series.head()

import numpy as np

# Month 컬럼의 년-월 을 나눠서 컬럼에 저장하기 위한 mapping
# 만약 아래의 tolist를 하지 않게 되면 어떻게 되나? --> Series가 되어 각 0, 1칸을 지정하기가 좀 까다롭긴한데..
# 가능한지 스터디때 해볼지 말지 생각해보자.
temp_date = df_time_series["Month"].map(lambda x: x.split('-'))
temp_date[:5]
# temp_date[:5][1] -> 이런식으로 해볼라고 했지만 컬럼별 call이 안됨.
temp_date = np.array(temp_date.values.tolist())
temp_date[:5]
df_time_series["year"] = temp_date[:,0]
df_time_series["month"] = temp_date[:,1]
df_time_series.head()

# 전달과의 변화량 %를 계산하기 위한 diff 컬럼 추가.
# 아래 것에 fillna(0)이 없었으면 NaN값이 포함되는데 이것이 대세에 영향을 주나? %계산에 영향을 주나?
  # -> 해보니까 여기서 fillna(0)을 하든 말든 다음 pct_change때 첫 행에 nan으로 저장됨. 쓰잘데없는듯.
df_time_series["diff"] = df_time_series["#Passengers"].diff().fillna(0)
df_time_series.head()

# pct_change()라는 pandas 함수로 이전 데이터와의 %차이를 계산.(매우편리하네. offset값도 매겨도됨.)
# Series.pct_change(periods=1, fill_method='pad', limit=None, freq=None, **kwargs) -> 요렇게 사용.
# 자세한것은 https://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.pct_change.html#pandas.Series.pct_change 참고
df_time_series["pct"] = df_time_series["#Passengers"].pct_change().map(lambda x : x * 100).map(lambda x : "%.2f" % x).fillna(0)
df_time_series.head()
