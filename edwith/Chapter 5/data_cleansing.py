#data cleansing
https://www.edwith.org/aipython/lecture/24076/
지금까지는 도구를 배웠음.
이제 뭘할지를 배우는과정

# Data quality problems
1. 데이터의 최대/최소가 다름 -> scale에 따른 y값에 영향
2. Ordinary 또는 Nominal 한값들의 표현을 어떻게?
3. 잘못 기입된 값들에 대한 처리
4. 값이없을 경우?
5. 극단적으로 큰 값 또는 작은값들은 그대로 놔두는가?

Data preprocessign issues
- 데이터가빠진경우
- 라벨링된 데이터 처리
- 스케일차이가 클때.

데이터가 없을때 전략
- sample을 drop
- 데이터가 없는 최소개수를 정해서 sample을 drop
- 데이터가 거의 없는 feature는 feature 자체를 drop
- 다른값으로 데이터 채우기.

# 1. Data FIlling
# Nan값이 있을떄

import pandas as pd
import numpy as np

# Eaxmple from - https://chrisalbon.com/python/pandas_missing_data.html
raw_data = {'first_name': ['Jason', np.nan, 'Tina', 'Jake', 'Amy'],
        'last_name': ['Miller', np.nan, 'Ali', 'Milner', 'Cooze'],
        'age': [42, np.nan, 36, 24, 73],
        'sex': ['m', np.nan, 'f', 'm', 'f'],
        'preTestScore': [4, np.nan, np.nan, 2, 3],
        'postTestScore': [25, np.nan, np.nan, 62, 70]}
df = pd.DataFrame(raw_data, columns = ['first_name', 'last_name', 'age', 'sex', 'preTestScore', 'postTestScore'])
df

# Nan값을 찾는 제일 쉬운 방법. 얼마나 비워져있는지 찾고 이를 길이로 나눠서 Nan값의 비율을 확인.
df.isnull().sum() / len(df)

# Nan이 있는 행을 Drop
df_no_missing = df.dropna()
df_no_missing
# Nan이 어떻게 있을때 drop? 이 예제는 전부다 Nan일때임.
df_cleaned = df.dropna(how="all")
df_cleaned

# Nan이 3개이상 있을떄 Drop -> thresh (쓰레스홀드)
df.dropna(axis=0, thresh=3)
#-> 너무 Data가 없으면 좀 써야겠지만 지저분한 Data를 drop 시키는게 제일 처음 할 일.

# Data filling
df.fillna(0) # 이건 당연히 하면 안되지.

# 보통 -> 평균(mean), 중위값median 최빈값mode을 활용함.
# 일단 평균값.
df["preTestScore"].mean()
# 그것을 넣어보자.
df["preTestScore"].fillna(df["preTestScore"].mean(), inplace=True) # inplace는 데이터를 아예바꿔라.
df

# 성별마다 다를때. groupby활용.
df.groupby("sex")["postTestScore"].sum()
# 각 값들을 성별별  그룹의 평균으로채울떄. 위와 달리 Nan만 채운게 아님.(즉 index를 한개 column 기준으로 할때)
df.groupby("sex")["postTestScore"].transform("mean")
# Boolean index를 써서 값들을 반환하는것도 가능
df[df['age'].notnull() & df['sex'].notnull()]
# 통계적인 값을 넣어 줄수도 있지만 보통은 평균등을 넣음.
# 성별등의 기준으로 넣어줌.

# 2. Category Data
# 이산형 데이터 어떻게 처리? one-hot encoding
# {g, b, y}
# g -> 1,0,0
# b -> 0,1,0
# y -> 0,0,1

edges = pd.DataFrame({'source': [0, 1, 2],
                   'target': [2, 2, 3],
                       'weight': [3, 4, 5],
                       'color': ['red', 'blue', 'blue']})

edges
# one-hot encoding 2개가 있다.
# 1) pandas, 2) 사이킷런에서.. 교슈는 판다스 선호. 데이터가 너무 크면 사이킥쓸때도.
# get_dummies(xx)
pd.get_dummies(edges)
# 알아서 카테고리 갯수까지 고려해서 칼럼으로 만들어줌

# 따로 꺼내서 만들어줄때도.
pd.get_dummies(edges["color"])

# ordinary data(서수형?데이터_순서잇는 데이터.)
# 저 3,4,5순서는 있지만 단지 비교용.
#아래와 같이 딕트타입으로 만들어줌.
# 딕트를 map에 연결하여
weight_dict = {3:"M", 4:"L", 5:"XL"}
edges["weight_sign"] = edges["weight"].map(weight_dict)
edges

edges = pd.get_dummies(edges)
edges
edges.as_matrix()
# 위의 경우 color와 weight 모두 각각 one-hot encoding 됨.




# Data Binning -> 퍼져있는 데이터들을 category화 시킬때. 데이터의 구간을 나눠보자.
# Example from - https://chrisalbon.com/python/pandas_binning_data.html
raw_data = {'regiment': ['Nighthawks', 'Nighthawks', 'Nighthawks', 'Nighthawks', 'Dragoons', 'Dragoons', 'Dragoons', 'Dragoons', 'Scouts', 'Scouts', 'Scouts', 'Scouts'],
        'company': ['1st', '1st', '2nd', '2nd', '1st', '1st', '2nd', '2nd','1st', '1st', '2nd', '2nd'],
        'name': ['Miller', 'Jacobson', 'Ali', 'Milner', 'Cooze', 'Jacon', 'Ryaner', 'Sone', 'Sloan', 'Piger', 'Riani', 'Ali'],
        'preTestScore': [4, 24, 31, 2, 3, 4, 24, 31, 2, 3, 2, 3],
        'postTestScore': [25, 94, 57, 62, 70, 25, 94, 57, 62, 70, 62, 70]}
df = pd.DataFrame(raw_data, columns = ['regiment', 'company', 'name', 'preTestScore', 'postTestScore'])
df
#Postscore를 학점으로 나눠보자.


bins = [0, 25, 50, 75, 100] # Define bins as 0 to 25, 25 to 50, 60 to 75, 75 to 100
group_names = ['Low', 'Okay', 'Good', 'Great'] # 구간이름
categories = pd.cut(df['postTestScore'], bins, labels=group_names) #cut 이용.
categories
# 기존 postTestScore를 빼주고(cut) category열에 너어쥐.
df['categories'] = pd.cut(df['postTestScore'], bins, labels=group_names)
df
pd.value_counts(df['categories'])


# Label encoding by sklearn -> 이것도 있다는 거 알아만 둬라.
# one-hot encoding을 기본적으로 지원함.
raw_example = df.as_matrix()
raw_example[:3]
data = raw_example.copy()
raw_example[:,0]

from sklearn import preprocessing
le = preprocessing.LabelEncoder() # -> encoding 생성
le.classes_ # -> labelling 이렇게 할거다라는거
le.fit(raw_example[:,0]) # -> data에 맞게 encoding fitting
raw_example[:,0] # -> 요넘을 OHE해준거
le.transform(raw_example[:,0]) # -> 실제 데이터 -> labelling data
#fit과 transform의 과정이 있는이유는 새로운 data입력시 기존 labelling 규칙을 그대로 적용할 필요가 있어서 함.
# fit은 규칙생성
# transform은 규칙을 저용하는 과정,
# 그래서 기존 인코더를 따로저장해서 살려둬야한다는거.
label_column = [0,1,2,5]
label_enconder_list = []
for column_index in  label_column:
    le = preprocessing.LabelEncoder()
    le.fit(raw_example[:,column_index])
    data[:,column_index] = le.transform(raw_example[:,column_index])
    label_enconder_list.append(le) # ->여기가 기존인코더 저장하는
    del le
data[:3]
label_enconder_list[0].transform(raw_example[:10,0])
# -> 저장된 le로 새로운 데이터에 적용
one_hot_enc = preprocessing.OneHotEncoder()
data[:,0].reshape(-1,1)
one_hot_enc.fit(data[:,0].reshape(-1,1))
data[:,0].reshape(-1,1)
onehotlabels = one_hot_enc.transform(data[:,0].reshape(-1,1)).toarray()
onehotlabels




# Feature Scaling -> feature간의 최대최소값의 차이를 맞춘다.
# Max-min 방법
# z score Normalization.-> 정규분포를 활용.
# 판다스보다 사이킷런을 더 씀.
#MinMaxScaler와 StandartScaler 사용
# -> NN을 쓰면서 FC를 안써도 된다는 말이있는데 그보다는 속도를 빨리하기 위함.
from sklearn import preprocessing

df = pd.io.parsers.read_csv(
    'https://raw.githubusercontent.com/rasbt/pattern_classification/master/data/wine_data.csv',
     header=None,
     usecols=[0,1,2]
    )
df.columns=['Class label', 'Alcohol', 'Malic acid']
df

from sklearn import preprocessing
std_scaler = preprocessing.StandardScaler().fit(df[['Alcohol', 'Malic acid']])
# 두개 열을 sklearn을 통해 std시킨
df_std = std_scaler.transform(df[['Alcohol', 'Malic acid']])
df_std
