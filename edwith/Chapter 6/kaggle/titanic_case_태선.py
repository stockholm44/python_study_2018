import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 순서.
# About Dataset
# Data Summary
# Feature Engineering
# Missing Value Analysis
# Outlier Analysis
# Correlation Analysis
# Visualizing Distribution Of Data
# Visualizing Count Vs (Month,Season,Hour,Weekday,Usertype)
# Filling 0's In Windspeed Using Random Forest
# Linear Regression Model
# Regularization Models
# Ensemble Models
# '''

# About dataset

# Variable	Definition	Key
# survival	Survival	0 = No, 1 = Yes
# pclass	Ticket class	1 = 1st, 2 = 2nd, 3 = 3rd
# sex	Sex
# Age	Age in years
# sibsp	# of siblings / spouses aboard the Titanic
# parch	# of parents / children aboard the Titanic
# ticket	Ticket number
# fare	Passenger fare
# cabin	Cabin number
# embarked	Port of Embarkation	C = Cherbourg, Q = Queenstown, S = Southampton

test_df = pd.read_csv('C:/study_2018/python_study_2018/edwith/Chapter 5/kaggle/test.csv')
train_df = pd.read_csv('C:/study_2018/python_study_2018/edwith/Chapter 5/kaggle/train.csv')

y_train_df = train_df.pop("Survived")
y_train_df

train_df.Cabin

pd.set_option('display.float_format', lambda x: '%.2f' % x)
test_df.head()
# Data Summary
# 1. 일단 데이터 불러오고
# 2. 불러온데이터의 head데인터 함 보고
# 3. 각각 컬럼의 type을 보고

test_df.head(2)
train_df.head(2)
train_df.dtypes
test_df.dtypes
train_df.shape
test_df.shape





# Feature Engineering
# 1. 일단 두개의 데이터를 합치고 passenger id로 index만들자.
# 2. train_df의 인덱스와 test_df의 인덱스를 각각 인덱스로 만들어놔서 나중에 filtering 할 수 있게 하자.
# 3. 추가로 뺼데이터 빼고 넣어야할 데이터는 넣되 one-hot encoding을 하면 될 듯.
# 4. 전체적으로 데이터에 대한 흠...타이타닉케이스 어려운가. 쉬운가..
# 5. 그냥하자..
# 6. 그래 간단하게 int data나 sex age등등에 대한 survive 회귀를 좀 보자...
train_index = train_df.index
train_index
test_df.index
all_df = pd.concat((train_df, test_df), axis=0)


# train_df -> training DataFrame -> survived 데이타 있음. fitting용 데이터
# test_df -> test DataFrame -> survived 없다.
# all_df -> training + test DataFrame -> survived 없다. feature를 한꺼번에 바꾸기 위해 합침.
         # ->  나중에 train_index, test_index로 나눠서 따로 저장할 수 있을듯.

all_df.head()
all_df = all_df.set_index(all_df["PassengerId"])
all_df.head()
del all_df["PassengerId"]
del all_df["Survived"]
del all_df["Cabin"]
all_df.head()

def name_filter(x):
    list = ['Mr','Mrs','Mister','Ms','Miss','Master']
    return x in list



df_name = all_df["Name"].map(filter(name_filter, all_df["Name"]))
# df_name.map(lambda x: )
# df_name
df_name

# PassengerId      int64 이걸 일단 인덱스로...
# Survived         int64 요건 겨로가데이터.
# Pclass           int64 숫자별 설바이브 데이터.
# Name            object - 이름에는 ms랑 mr랑 등등이 있다. 이것정도는 나눌수 잇다
# Sex             object
# Age            float64
# SibSp            int64
# Parch            int64
# Ticket          object
# Fare           float64
# Cabin           object
# Embarked        object
# dtype: object


# 나이와 cabin에 결측치가 개많음... 그래서 age를 빼버렸나...아니면
train_df.isnull().sum()/len(train_df) # null의 퍼센테이지

Pclass_age = all_df.groupby(["Pclass"])["Age"].mean()
Pclass_age
age_dict = dict(Pclass_age)
all_df["Age"] = all_df["Age"].fillna(all_df["Pclass"].replace(age_dict))

all_df.isnull().sum()/len(all_df)

import seaborn as sn
# Pclass_sur = train_df.groupby(["Pclass"])["Survived"].mean()
# Pclass_sur
# Sex_sur = train_df.groupby(["Sex"])["Survived"].mean()
# Age_sur = train_df.groupby(["Age"])["Survived"].mean()
# SibSp_sur = train_df.groupby(["SibSp"])["Survived"].mean()
# Parch_sur = train_df.groupby(["Parch"])["Survived"].mean()
# Ticket_sur = train_df.groupby(["Ticket"])["Survived"].mean()
# Fare_sur = train_df.groupby(["Fare"])["Survived"].mean()
# Embarked_sur = train_df.groupby(["Embarked"])["Survived"].mean()

# sn.set_style(style='whitegrid')
sn.set_style(style='whitegrid')
sn.countplot(x='Survived',data=train_df,hue='Sex')
sn.countplot(x='Survived', data=train_df, hue='Pclass')
sn.distplot(train_df['Age'].dropna(),kde=False,bins=30,color='darkred')

train_df.SibSp.unique()
train_df.Parch.unique()
train_df.Fare
sn.countplot(x='Survived', data=train_df, hue='SibSp')
sn.countplot(x='Survived', data=train_df, hue='Parch')


fig, axes = plt.subplots(nrows=3,ncols=3)
fig.set_size_inches(12,5)
axes[0][0].bar(train_df["Pclass"], train_df["Pclass_sur"])
axes[0][1].bar(train_df["Sex"], train_df["Survived"])
axes[0][2].bar(train_df["Age"], train_df["Survived"])
axes[1][0].bar(train_df["SibSp"], train_df["Survived"])
axes[1][1].bar(train_df["Parch"], train_df["Survived"])
axes[1][2].bar(train_df["Ticket"], train_df["Survived"])
axes[2][0].bar(train_df["Fare"], train_df["Survived"])
axes[2][1].bar(train_df["Embarked"], train_df["Survived"])
plt.show()
