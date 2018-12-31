#data cleansing
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import matplotlib.pyplot as plt
# 1. Data FIlling
# Nan값이 있을떄

raw_data = {'first_name': ['Jason', np.nan, 'Tina', 'Jake', 'Amy'],
        'last_name': ['Miller', np.nan, 'Ali', 'Milner', 'Cooze'],
        'age': [42, np.nan, 36, 24, 73],
        'sex': ['m', np.nan, 'f', 'm', 'f'],
        'preTestScore': [4, np.nan, np.nan, 2, 3],
        'postTestScore': [25, np.nan, np.nan, 62, 70]}
1. df라는 변수에 raw_data, 컬럼은 ['first_name', 'last_name', 'age', 'sex', 'preTestScore', 'postTestScore']로 만들기

df = DataFrame(data=raw_data, columns=['first_name', 'last_name', 'age', 'sex', 'preTestScore', 'postTestScore'])
df
2. 각 컬럼별 Nan 갯수
df.isnull().sum()
3. 위 2번을 이용하여 각 컬럼별 NaN의 비율
df.isnull().sum()/len(df)

4. df_no_missing에 NaN이 한개도 없는 row만 저장하기
df_no_missing = df.dropna()
df_no_missing
5. df_cleaned에 NaN으로만 된 행들 제거하여 저장하기
df_cleaned = df.dropna(how='all')
df_cleaned
6. NaN이 3개이상 있는 row만 drop하기.
df.dropna(thresh=3)
7. df에서 NaN을 0으로 바꾸기
df.fillna(0)
8. preTestScore 컬럼의 평균구하기
pre_mean = df.preTestScore.mean()
9. preTestScore의 NaN에 preTestScore의 평균 집어넣기. 단 df는 원본그대로 보존
df.preTestScore.fillna(pre_mean)
df
10. preTestScore의 NaN에 preTestScore의 평균 집어넣기. 단 df data자체에 저장
df["preTestScore"].fillna(pre_mean, inplace=True)
df
11. sex컬럼을 그룹으로 하여 총합구하기.(그룹바이)
df.groupby("sex")["postTestScore"].sum()
12. sex컬럼을 기준으로 해당 성별 평균으로 값 대체하기.(트렌스폼)
df.groupby("sex")["postTestScore"].transform("mean")
13. age, sex 컬럼 모두 null이 아닌 row만 반환
df[df.age.notnull() & df.sex.notnull()]
14. age가 25이상인 row만 반환
df[df.age>25]
15. sex가 'f'인 row만 반환
df[df.sex=='f']


# 2. Category Data
edges = pd.DataFrame({'source': [0, 1, 2],
                      'target': [2, 2, 3],
                      'weight': [3, 4, 5],
                      'color': ['red', 'blue', 'blue']})

16. 위 edges data의 category data를 one-hot encoding해라.(이산형 데이터의 index로 column 만들기.)(겟더미)
pd.get_dummies(edges)


17. 위에건 edges안에 추가한거고 color만 따로 Dataframe에서 꺼내라.
pd.get_dummies(edges.color)
# Ordinary Data

18. 위 edges data의 weight 칼람의 3, 4, 5에 각각 M L, XL를 대입한 weight_sign column을 넣어라.
   (먼저 dict만들고, 그담에 맵으로 연결)
weight_dict = {3:'M', 4:'L', 5:'XL'}
edges["weight_sign"] = edges['weight'].map(weight_dict)
edges
19. 위 edges data의 weight_sign을 one-hot-encoding해라.
edges = pd.get_dummies(edges)
edges

20. 그렇게 만들어진 edges를 matrix로 만들어라.
np.array(edges)
edges.values
# Data Binning
raw_data = {'regiment': ['Nighthawks', 'Nighthawks', 'Nighthawks', 'Nighthawks', 'Dragoons', 'Dragoons', 'Dragoons', 'Dragoons', 'Scouts', 'Scouts', 'Scouts', 'Scouts'],
        'company': ['1st', '1st', '2nd', '2nd', '1st', '1st', '2nd', '2nd','1st', '1st', '2nd', '2nd'],
        'name': ['Miller', 'Jacobson', 'Ali', 'Milner', 'Cooze', 'Jacon', 'Ryaner', 'Sone', 'Sloan', 'Piger', 'Riani', 'Ali'],
        'preTestScore': [4, 24, 31, 2, 3, 4, 24, 31, 2, 3, 2, 3],
        'postTestScore': [25, 94, 57, 62, 70, 25, 94, 57, 62, 70, 62, 70]}
21. 위의 raw_data를 ['regiment', 'company', 'name', 'preTestScore', 'postTestScore']를 컬럼으로 하여 dataframe으로 맨들어라,
df = DataFrame(raw_data, columns=['regiment', 'company', 'name', 'preTestScore', 'postTestScore'])
df
22. Define bins as 0 to 25, 25 to 50, 60 to 75, 75 to 100 -> 구간나누기 준비 list만들어라
bins = [0,25,50,75,100]
23. 구간이름을 만들어라.(4개구간으로.) group_names
group_names = ["Low", "Okay", "Good", "Great"]

24. 위 df의 postTestScore 를 bins로 만든 구간으로 나누고 라벨은 위의 group_names로 넣어라.
categories = pd.cut(df.postTestScore, bins=bins, labels=group_names)
categories
25. df에 categories 이름의 컬럼으로 postTestScore를 bins 구간으로 나누고 라벨이름은 group_names로 해줘라.
df["categories"] = pd.cut(df.postTestScore, bins=bins, labels=group_names)
df
26. 위 df의 categoris의 항목들의 빈도를 나타내라
pd.value_counts(df["categories"])
