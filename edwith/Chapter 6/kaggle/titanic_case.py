import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# 1) Load dataset
test_df = pd.read_csv("C:/study_2018/python_study_2018/edwith/Chapter 6/kaggle/test.csv")
train_df = pd.read_csv("C:/study_2018/python_study_2018/edwith/Chapter 6/kaggle/train.csv")

train_df.head(1)
train_df.index = train_df.PassengerId

train_df.index = train_df.PassengerId # 요렇게 하고 해당인덱스로 사용한 컬럼을 del해도 되지만
train_df.set_index('PassengerId', inplace=True) # 요렇게 하면 inplace속성만 True하면 됨.
train_df.head(1)
test_df.set_index('PassengerId', inplace=True) # 요렇게 하면 inplace속성만 True하면 됨.
test_df.head(1)

# 인덱스를 따로 꺼낸다.
train_index = train_df.index
test_index = test_df.index


# 생존여부를 따로 빼네서 y_train_df로 구성. 답변 제출을 위함.
y_train_df = train_df.pop("Survived")


y_train_df.head(3)

# 2) Data preprocessing
pd.set_option('display.float_format', lambda x: '%.2f' % x)
test_df.head()

pd.set_option('display.float_format', lambda x: '%.4f' % x)
train_df.head()
# NaN 갯수 세서 그것의 비율을 column 별로.
test_df.isnull().sum() / len(test_df) * 100

del test_df["Cabin"]
del train_df["Cabin"]

# test_df랑 train_df의 컬럼이 같기 때문에 append로 붙여도되는듯.
 #->concat이 아니고 append인 이유는 잘 모르겠다.
all_df = train_df.append(test_df)
all_df

(all_df.isnull().sum() / len(all_df)).plot(kind='bar')
len(all_df)
all_df.head(1)
del all_df["Name"]
del all_df["Ticket"]
all_df.head()

all_df["Sex"] = all_df["Sex"].replace({"male":0,"female":1})
all_df.head()
all_df.Embarked.unique()
all_df["Embarked"] = all_df["Embarked"].replace({"S":0,"C":1,"Q":2, np.nan:99})
all_df["Embarked"].unique()
all_df.head()
pd.get_dummies(all_df["Embarked"], prefix="embarked")
matrix_df = pd.merge(all_df, pd.get_dummies(all_df["Embarked"], prefix="embarked"), left_index=True, right_index=True)
matrix_df.head()
matrix_df.corr() # DataFrame.corr() Compute pairwise correlation of columns, excluding NA/null values
all_df.groupby("Pclass")["Age"].mean()
all_df.Age
age_dict = dict(all_df.groupby("Pclass")["Age"].mean())
age_dict

age_dict.keys()

all_df['Age_a'] = all_df['Age'].fillna(all_df["Pclass"].replace(age_dict))

# for key in age_dict.keys():
#     all_df.loc[(all_df["Pclass"] == key) & (all_df["Age"].isnull()), "Age"] = age_dict[key]
all_df['Age_a']



all_df.Age

all_df.isnull().sum()

all_df.groupby("Pclass")["Fare"].mean()

all_df[all_df["Fare"].isnull()]
all_df.loc[all_df["Fare"].isnull(), "Fare"] = 13.30 # 3등급 빈게 이거밖에 없어서 비움.

all_df.Embarked

del all_df["Embarked"]

all_df["Pclass"] = all_df["Pclass"].replace({1:"A",2:"B",3:"C"})
all_df = pd.get_dummies(all_df)
all_df.head()
all_df = pd.merge(
    all_df, matrix_df[["embarked_0", "embarked_1", "embarked_2", "embarked_99"]],
    left_index=True, right_index=True)
train_df = all_df[all_df.index.isin(train_index)]
test_df = all_df[all_df.index.isin(test_index)]
train_df.head()
test_df.head()


x_data = train_df.as_matrix()
y_data = y_train_df.as_matrix()
x_data.shape, y_data.shape

y_data

from sklearn.linear_model import LogisticRegression
cls = LogisticRegression()
cls.fit(x_data,y_data)

cls.intercept_

cls.coef_

cls.predict(test_df.values)
test_df.index

x_test = test_df.as_matrix()
y_test =cls.predict(x_test)
y_test

result = np.concatenate( (test_index.values.reshape(-1,1), cls.predict(x_test).reshape(-1,1)  ) , axis=1)
result[:5]

df_submssion =pd.DataFrame(result, columns=["PassengerId","Survived"])
df_submssion

df_submssion.to_csv("submission_result.csv",index=False)
