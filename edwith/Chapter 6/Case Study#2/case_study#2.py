import pandas as pd
import numpy as np

filename = 'C:/study_2018/python_study_2018/edwith/Chapter 6/Case Study#2/sea_managing_raw.csv'
# encoding="cp949"안하니까 못읽음.
patent = pd.read_csv(filename, encoding="cp949")
patent.head(2)

# 출원번호에 nan값 있는지 체크. 0이면 없다. >0이면 그 숫자만큼 잇다. 왜냐면 True=1 False=0이므로
patent["출원번호"].isnull().sum()

# 출원번호/Original xxx만으로만 dataframe 구성.
# 출원번호별로 class 구분을 one-hot encoding으로 구분하기 위한 전처리.
df_patent = patent[["출원번호","Original US Class All[US]"]]
df_patent.head(2)

# 특허 class 구분이 여러개인 것들이 있어 이를 각 row별 리스트로 저장.
df_patent["Original US Class All[US]"].map(lambda x : x.split("|")).tolist()

df_patent["출원번호"].tolist()
# 각각의 출원번호와 해당 특허 class 리스트를 리스트로 만들어줌.
# 요기 좀 헷갈리므로 이 이후 부분 스터디떄 해라.원핫 인코딩도 있으니 한번 직접 해보는 시간 갖자.
edge_list = []
for data in zip(df_patent["출원번호"].tolist(), df_patent["Original US Class All[US]"].map(lambda x : x.split("|")).tolist()):
    for value in data[1]:
        edge_list.append([data[0],value.strip()])
edge_list
