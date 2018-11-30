# pandas#2 Data Handling
# Toy data는 깔끔하지만 서버의 데이타는 드럽.
# 그거에 대한 전처리를 위해 pandas를 쓰는거지요.

# 1. groupby
# -SQL groupby 명령어와 같음.
# -split->apply->combine 과정을 거쳐 연산


import pandas as pd
# data from:
ipl_data = {'Team': ['Riders', 'Riders', 'Devils', 'Devils', 'Kings',
         'kings', 'Kings', 'Kings', 'Riders', 'Royals', 'Royals', 'Riders'],
         'Rank': [1, 2, 2, 3, 3,4 ,1 ,1,2 , 4,1,2],
         'Year': [2014,2015,2014,2015,2014,2015,2016,2017,2016,2014,2015,2017],
         'Points':[876,789,863,673,741,812,756,788,694,701,804,690]}

df = pd.DataFrame(ipl_data)
df
df.groupby("Team")["Points"].sum() #한개로도 모을수 있고
df.groupby(["Team","Year"])["Points"].sum()  #여러개로도 모을수 있다.

Hierarchical index
- Groupby 명령의 결과물도 결국은 dataframe
- 두 개의 column으로 groupby를 할 경우, index가 두개 생성


h_index = df.groupby(["Team", "Year"])["Points"].sum()
h_index
h_index.index

h_index["Devils":"Kings"]
# 이렇게 matrix형태로 굉장히 많이 표현하므로 매우 사용이 많음.
h_index.unstack()
# Hierarchical index – swaplevel
# - Index level을 변경할 수 있음
h_index.swaplevel()

# Hierarchical index – operations
# - Index level을 기준으로 기본 연산 수행 가능 -> 중요하진않으니 가능은하구나정도..
h_index.swaplevel().sortlevel(0)


# 2. Groupby2 -> 중요하진않아보이니 이런게 있다정도..

# Groupby – gropued
# - Groupby에 의해 Split된 상태를 추출 가능함

grouped = df.groupby("Team")
for name,group in grouped:
    print (name)
    print (group)
# -> 각 그룹별로 정보 추출했네여.엑셀대용으로는쓰지만 그거이외에는 잘 안씀.

# 다양한 형태로 transform 가능.

grouped.agg(min)
import numpy as np
grouped.agg(np.mean)

grouped['Points'].agg([np.sum, np.mean, np.std])
# lambda도 지정가능
score = lambda x: (x - x.mean()) / x.std()
grouped.transform(score)
# -> 엑셀처럼 여러개의 통계치를 한꺼번에 볼수 있다.. 연습문제 잠깐해볼만. 피벗테이블대안정도로 기억.

# Groupby – transformation
# - Aggregation과 달리 key값 별로 요약된 정보가 아님
# - 개별 데이터의 변환을 지원함
# -> 조금재밌기도 하고 웃기다.
# 같은항목들끼리 뭉쳐서 그걸 시리즈데이터로 만들어서 적용시킬수 있다.
df
# 요 시리즈 데이터에서 맥스값을 뽑아라.
# 단 max나 min 처럼 Series 데이터에 적용되는 데이터 들은
# Key값을 기준으로 Grouped된 데이터 기준
score = lambda x: (x.max())
grouped.transform(score) #팀별 그룹끼리 맥스값으로 들어감.

# 각각별로 nomalization-> 뭐 많이 안씀.
score = lambda x: (x - x.mean())/x.std()
grouped.transform(score)


# Groupby - filter ->얘가 더 많이쓰겠다. 엑셀대체기능정도만.
# - filter안에는 boolean 조건이 존재해야함
# - len(x)는 grouped된 dataframe 개수
df.groupby('Team').filter(lambda x: len(x)>=3) # Team에 대한 row가 3개이상인것.
# 추가 3개 예제
df.groupby('Team').filter(lambda x: x['Rank'].sum() > 2)
df.groupby('Team').filter(lambda x: x['Points'].max() > 800) #각그룹의 max가 800이상인것.
df.groupby('Team').filter(lambda x: x['Rank'].mean() < 2)


# 4. Case Study

# !wget https://www.shanelynn.ie/wp-content/uploads/2015/06/phone_data.csv
df_phone = pd.read_csv("C:/djangocym/study_2018/lab_bla/data/phone_data.csv")
df_phone.head()

import dateutil
df_phone['date'] = df_phone['date'].apply(dateutil.parser.parse, dayfirst=True)
df_phone.head()

df_phone.groupby('month')['duration'].sum()
df_phone[df_phone['item'] == 'call'].groupby('month')['duration'].sum()

df_phone.groupby(['month', 'item'])['duration'].sum()

df_phone.groupby(['month', 'item'])['date'].count().unstack()
df_phone.groupby('month', as_index=False).agg({"duration": "sum"})
# 요렇게 피벗테이블처럼 여러개의 통계 데이터를 보여줄때 agg를 많이쓴다.
df_phone.groupby(['month', 'item']).agg({'duration':sum,      # find the sum of the durations for each group
                                     'network_type': "count", # find the number of network type entries
                                     'date': 'first'})    # get the first date per group

# 한개의 column에 대한 다영한 data를 한번에 받아올수도 있다.
df_phone.groupby(['month', 'item']).agg({'duration': [min],      # find the min, max, and sum of the duration column
                                     'network_type': "count", # find the number of network type entries
                                     'date': [min, 'first', 'nunique']})    # get the min, first, and number of unique dates

# -> 따로 Pandas를 공부하는 사람은 딱히 없다. 검색해보거나 연습할때 많이 쓰면 된다.


# Pivot table / Crosstab
# 2개 컬럼으로 데이터 행렬만드는걸 그룹바이가 아닌 피벗으로 간단하게 구현가능
# Pivot Table
# - 우리가 Excel에서 보던 그 것!
# - Index 축은 groupby와 동일함
# - Column에 추가로 labelling 값을 추가하여,
# - Value에 numeric type 값을 aggregation 하는 형태
df_phone = pd.read_csv("C:/djangocym/study_2018/lab_bla/data/phone_data.csv")
df_phone['date'] = df_phone['date'].apply(dateutil.parser.parse, dayfirst=True)
df_phone.head()
df_phone.pivot_table(["duration"],
                     index=[df_phone.month,df_phone.item],
                     columns=df_phone.network, aggfunc="sum", fill_value=0)

# Crosstab
# relation table을 groupby가 아닌 크로스탭으로 가능.
# Crosstab
# - 특허 두 칼럼에 교차 빈도, 비율, 덧셈 등을 구할 때 사용
# - Pivot table의 특수한 형태
# - User-Item Rating Matrix 등을 만들 때 사용가능함
df_movie = pd.read_csv("C:/djangocym/study_2018/lab_bla/data/movie_rating.csv")
df_movie.head()

df_movie.pivot_table(["rating"], index=df_movie.critic, columns=df_movie.title,
                     aggfunc="sum", fill_value=0)
pd.crosstab(index=df_movie.critic,columns=df_movie.title,values=df_movie.rating,
            aggfunc="first").fillna(0)
            #-> 하지만 교수는 groupby를 많이 쓴다고 함.
            #-> 피벗, 크로스로 갈수록 특수한 형태로 변하는것 뿐. 나머지는 동일.



# 특허 data -> 예제가 없어서 그냥 대충함.연습문제 따로 따보자.
import pandas as pd

df_ipcr = pd.read_csv("./data/ipcr.tsv", delimiter="\t")
# NaN값들이 많고 지저분.. 전처리 필수.
df_ipcr["section"].isnull().sum()
df_ipcr["ipc_class"].isnull().sum()
df_ipcr["subclass"].isnull().sum() # -> 5개가 나옴. 그러면 없에주고 다시 만들다;.
df_ipcr = df_ipcr[ df_ipcr["subclass"].isnull() == False] # ->무슨의미지?? null이 아닌경우에만 모아라 란다.

# 각각의 데이터들을 str으로 바꿔줌.
df_ipcr["ipc_class"] = df_ipcr["ipc_class"].map(str)
df_ipcr = df_ipcr[df_ipcr["ipc_class"].map(str.isdigit)]  .
df_ipcr["ipc_class"] = df_ipcr["ipc_class"].astype(int)

# 자리수를 맞춰줬어
two_digit_f = lambda x : '{0:02d}'.format(x)
two_digit_f(3) # 1->01로 바꾸고 그것을 str으로 바꿈,

# 알파벳이 아닌 숫자가 있으면 제거.
df_ipcr["subclass"] = df_ipcr["subclass"].map(str.upper)
df_ipcr = df_ipcr[df_ipcr["subclass"].isin(list("ABCEDFGHIJKLMNOPQRSTUVWXYZ"))]

# 4자리수 만들어줌.
df_ipcr["4digit"] = df_ipcr["section"] + df_ipcr["ipc_class"] + df_ipcr["subclass"]

# Patent ID 와 4digit으로 행렬구성해서 봄.
df_data = df_ipcr[["patent_id", "4digit"]]
df_data.describe()


# Merge & Concat
# Merge
# - SQL에서 많이 사용하는 Merge와 같은 기능
# - 두 개의 데이터를 하나로 합침
# feature가 기존 data에 추가로 더해야할 feature가 더 있있을 경우. 이를 합쳐줌.
# 토이데이터에는 별로 필요없다. 하지만 db에 있는데이터는 지저분하고 맘대로 되어있어서 전처리가 필요하여 merge같은거 많이 씀.

import pandas as pd
raw_data = {
        'subject_id': ['1', '2', '3', '4', '5', '7', '8', '9', '10', '11'],
        'test_score': [51, 15, 15, 61, 16, 14, 15, 1, 61, 16]}
df_a = pd.DataFrame(raw_data, columns = ['subject_id', 'test_score'])
df_a
raw_data = {
        'subject_id': ['4', '5', '6', '7', '8'],
        'first_name': ['Billy', 'Brian', 'Bran', 'Bryce', 'Betty'],
        'last_name': ['Bonder', 'Black', 'Balwner', 'Brice', 'Btisan']}
df_b = pd.DataFrame(raw_data, columns = ['subject_id', 'first_name', 'last_name'])
df_b

# on 인수로 붙이게 될 기준을 정함.
pd.merge(df_a, df_b, on='subject_id')

# 두 dataframe의 column이름이 다를때.
pd.merge(df_a, df_b, left_on='subject_id', right_on='subject_id')

# how에 join method를 정할 수 있음.
pd.merge(df_a, df_b, on='subject_id', how='left')
pd.merge(df_a, df_b, on='subject_id', how='right')
pd.merge(df_a, df_b, on='subject_id', how='outer')
pd.merge(df_a, df_b, on='subject_id', how='inner')

# 양쪽의 index를 살려야할 때.
pd.merge(df_a, df_b, right_index=True, left_index=True)

# Concat -> numpy와 동일하므로 패스

# Database connection
import sqlite3 #pymysql <- 설치

conn = sqlite3.connect("C:/djangocym/study_2018/lab_bla/data/flights.db") # db연결을 위한 개채
cur = conn.cursor()
cur.execute("select * from airlines limit 5;")
results = cur.fetchall()
results


df_airplines = pd.read_sql_query("select * from airlines;", conn)
df_airplines

# 속도를 위해 파일 형태로 저장할때가 잇지용.
# 그래서 엑셀.

# XLS persistence -> 코드만 보자. 엑셀로 변환시켜주는거.
writer = pd.ExcelWriter('C:/djangocym/study_2018/lab_bla/data/df_routes.xlsx', engine='xlsxwriter')
df_routes.to_excel(writer, sheet_name='Sheet1')


# Pickle persistence
# - 가장 일반적인 python 파일 persistence
# - to_pickle, read_pickle 함수 사용.
df_routes.to_pickle("C:/djangocym/study_2018/lab_bla/data/df_routes.pickle")
df_routes_pickle = pd.read_pickle("./data/df_routes.pickle")
df_routes_pickle.head()
df_routes_pickle.describe()
