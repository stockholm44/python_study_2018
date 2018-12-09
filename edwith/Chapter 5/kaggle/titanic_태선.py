import numpy as np
import pandas as pd

test = 'C:/study_2018/python_study_2018/edwith/Chapter 5/kaggle/test.csv'
train = 'C:/study_2018/python_study_2018/edwith/Chapter 5/kaggle/train.csv'

test_df = pd.read_csv(test, sep=',')
train_df = pd.read_csv(train, sep=',')

train_df.head()

# 승객번호로 indexing

train_df.index = train_df.PassengerId
del train_df["PassengerId"]
train_df.head()

test_df.index = test_df.PassengerId
del test_df["PassengerId"]
test_df.head()
# 답안제출을 위한 생존데이터만 꺼내기.
 # -> pop을 꼭 써야하나? 아래처럼 안써도 되는거 아닌가?
y_train_df = train_df.Survived
y_train_df.head()
# pandas 내 전체
pd.set_option('display.float_format', lambda x: " %.2f" % x)
test_df.head()


test_df.isnull().sum()/len(test_df)
