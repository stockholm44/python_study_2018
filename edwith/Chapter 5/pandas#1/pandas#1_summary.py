# pandas#1

# Pandas는 python 계의 엑셀! 하지만 엑셀이 더쉽긴하다 ㅋㅋ
# numpy의 ref.이므로 numpy기능을 그대로 제공.
# indexing 전처리등에 많이씀
# 필요없는데이타 날리기, 머지등을 하기 쉽다.

import pandas as pd
import numpy as np

data_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data' #Data URL
# data_url = './housing.data' #Data URL
df_data = pd.read_csv(data_url, sep='\s+', header = None) #csv 타입 데이터 로드, separate는 빈공간으로 지정하고, Column은 없음
df_data.head()

# pandas는 series, dataframe으로 구성.
# series는 한개 column에 해당
# series를 모은게 dataframe


# 1. Series
# Series도 index가 있긴한데 별로 신경안쓰는편.
from pandas import Series, DataFrame
import pandas as pd
import numpy as np

list_data = [1,2,3,4,5]
example_obj = Series(data = list_data)
example_obj

# 요것 처럼 index 지정가능
list_data = [1,2,3,4,5]
list_name = list('abced')
example_obj = Series(data = list_data, index=list_name)
example_obj

# dict타입으로도 만들수 있다. 이때는 index는 알아서
dict_data = {"a":1, "b":2, "c":3, "d":4, "e":5}
example_obj = Series(dict_data, dtype=np.float32, name="example_data")
example_obj

# Data접근하기할때는 index로 접근. 하지만 dataframe에서는 좀 쓰고 여기선 좀 불편
example_obj["a"]

# 2. Dataframe
# 기본적으로 matrix를 가정. row, column으로 접근가능하나 좀 귀찮...
# numpy의 subclass라 그쪽기능 거의다 사용가능.
# Example from - https://chrisalbon.com/python/pandas_map_values_to_values.html
raw_data = {'first_name': ['Jason', 'Molly', 'Tina', 'Jake', 'Amy'],
        'last_name': ['Miller', 'Jacobson', 'Ali', 'Milner', 'Cooze'],
        'age': [42, 52, 36, 24, 73],
        'city': ['San Francisco', 'Baltimore', 'Miami', 'Douglas', 'Boston']}
df = pd.DataFrame(raw_data, columns = ['first_name', 'last_name', 'age', 'city'])
df

# 기존 column 바꾸기
df_data = pd.DataFrame(raw_data).head(5)
df_data
df_data.columns = ["First_Name", "Family_Name", "Age", "Born"]
df_data
#요기부터 집중!
#특별 column 만가져올수 있다. 이게 numpy와 다른점.
pd.DataFrame(raw_data, columns = ['age', 'city'])

# 기존에 없던 새로운 column을 넣으면 Nan으로 들어감,
DataFrame(raw_data, columns = ['first_name', 'last_name', 'age', 'city', 'debt'])

# Column을 선택하여 Series로 추출가능.
df.first_name
df['first_name']



# Dataframe의 값 접근법
# 1) loc - index location -> 인덱스 이름으로 반환
# 2) iloc - index position -> 인덱스 숫자위#로 반환
# 아랫것으로 구분해보자.


# Example from - https://stackoverflow.com/questions/31593201/pandas-iloc-vs-ix-vs-loc-explanation
s = pd.Series(np.nan, index=[49,48,47,46,45, 1, 2, 3, 4, 5])
s
s.iloc[:3] # 3번째 위치 전까지를 뽑아줌.
s.loc[:3] # 인덱스 이름이 3인놈 까지를 뽑아줌.


# Column 에 새로운 데이터 할당 -> 이것 굉장히 많이씀.
# 있는 data를 갖고 새로운 feature를 만들때 많이 사용함.
df = DataFrame(raw_data, columns = ['first_name', 'last_name', 'age', 'city', 'debt'])
df.debt = df.age>40
df

# Transpose, 값출력, csv 변환 기능 제공.
df.T
df.values
df.to_csv()

# Column 삭제
del df["debt"]

# 3. Selection & Drop
# numpy떄의 indexing과 같은 기능.

# head하면 적힌 수만큼 반환
import pandas as pd
import numpy as np
df = pd.read_excel("C:/djangocym/study_2018/lab_bla/data/excel-comp-data.xlsx")
df.head()
df
df["account"].head(3) # 한개의 column가져오기
df[["account", "street", "state"]].head(3) # 여러개의 column 가져오기

# Selection with index number
df[:3] # str을 넣으면 column을 가져오는데 숫자를 넣으면 row? 좀 일관성이 없긴함.
df["account"][:3]

# series selection - 웃기다면 웃기고 재미있다면 재밌는 기능
 #-> index값을 넣으면 index값들의 data를 가져옴.
account_series = df["account"]
account_series[:3]
account_series[[1,5,2]] #헷갈리는 문법이지만. 일단 row index기준으로 뽑는거다.

# Boolean index - 먹힘.
account_series[account_series<250000]


# index 변경
# 숫자가 아닌 다른 예를들면 주민번호, 학번으로 하고 싶으면 index로 해줄수 있다.
df.index = df["account"]
del df['account'] # 남아있으니까 Del해주자
df.head()

# Basic, loc, iloc Selection
df[["name","street"]][:2] # column과 index number -> 실제로 해볼때는 데이터접근이 좀 어렵ㄴ다. 컬럼이 많을떄는 쓴다.
df.loc[[211829,320563],["name","street"]] # column과 index name
df[["name", "street"]].iloc[:10] # column number와 index number -> 교수는 이걸많이쓴다.col이 몇개 없을때는 편한데 데이터 엄청많은면 iloc으로만은 힘드므로 위의 2개사용함.

# index 재설정 -> 머지가 없을경우에는 제일편함.
df.index=list(range(0,15))
df.head()

# data drop
#column단위면 del인데 row단위면 drop괴 row Num적어줌.
df.drop(1) # index1 row 없에기
df.drop([0,1, 2,3]) # 여러개도 가능
 #-> 결측치가 많거나 할때 사용. drop 자체를 사용할일이 없을수 있다.

# axis 축을 기준으로 drop -> 별로랜다.

df.drop("city",axis=1).head() # 축을기준으로 없에라.
# -> 재밌는건..보여주는것만 없어지는거지 원래 df를 열어보면 그대로 있당.
# -> 왜냐면 pandas에서는 데이터 핸들링할때 원본데이터를 쉽게 삭제하지 않음.
# -> 원본데이터를 바꾸려면 inplace=True를 넣어야힘. False로 하면 copy한버전에서 삭제해서 보여주는거.


# 3. Dataframe Operations

# Serirs Operation
# index를 기준으로 연산수행
# 겹치는 index가 없을 경우 Nan값으로 변환
# numpy의 broadcasting 과는 좀 다름.
# index의 이름이 같은것끼리 더해줌.아래의 e의경우 2개인데 각각에 동일 e를 더해줌
import pandas as pd
import numpy as np
from pandas import Series, DataFrame
s1 = Series(range(1,6), index=list("abcde"))
s1
s2=Series(range(5,11), index=list("bcedef"))
s1.add(s2)
s1+s2


# Dataframe operation
# df는 column과 index를 모두 고려
# add operation을 쓰면 Nan값 0으로 변환
# operation type -> add, sub, div, mul
df1=DataFrame(np.arange(9).reshape(3,3),columns=list("abc"))
df1
df2=DataFrame(np.arange(16).reshape(4,4),columns=list("abcd"))
df2
df1+df2
df1.add(df2,fill_value=0)



# Series + Dataframe
df = DataFrame(np.arange(16).reshape(4,4),columns=list('abcd'))
df
s = Series(np.arange(10,14),index=list('abcd'))
s
df+s # Column을 기준으로 broadcasting이 일어남.
df.add(s2, axis=0) # 축을 지정해주면 그거 기준으로 broadcasting일어남.



# 3. Lambda, map, apply -> 요기가 좀 재밌고 실용적!!!
# 굉장히 편하게 pandas에서 적용가능
import pandas as pd
import numpy as np
from pandas import Series

s1 = Series(np.arange(10))
s1.head(5)
s1.map(lambda x: x**2).head(5)

# 값들 대체할때 꽤쉽게 가능.
z = {1: 'A', 2: 'B', 3: 'C'}
s1.map(z).head(5)
s1 = Series(np.arange(10))
s1
s2=Series(np.arange(10,20))
s2
s1.map(s2) #요렇게 시리즈끼리 맵핑가능


# 아래예제와 같이 sex에 따라서 one-hot 매길수 있다. dict타입으로 mapping
df = pd.read_csv("C:/djangocym/study_2018/lab_bla/data/wages.csv")
df.head()
df.sex.unique()

df["sex_code"] =  df.sex.map({"male":0, "female":1}) # 제일많이쓰는 테크닉. 데이터 컨버젼. 아래와 같이 replace도가능
df.head(5)
df.sex.replace({"male":0, "female":1}).head()
# -> inpla  ce=True쓰고 sex_code지우면 완전히 대체도 가능. 예제로 쓰장.!! del index["sex_code"]



# Apply for DataFrame
 # -> map과 달리 series전체 column에 해당함수를 적용
 # - 입력값이 series데이터로 입력받아 handling 가능.
 #연산할때 보통 쓰는듯.
 # 컬럼의 통계치 사용시 많이씀.
 # 재밌는거 좀더 할 수 있긴함. ㅋ

df_info = df[["earn", "height","age"]]
df_info.head()
f = lambda x : x.max() - x.min()
df_info.apply(f)
df_info.apply(sum)
df_info.sum()

# 이렇게 함수로 지정해서 Series를 반환하게 할 수 있도있게 잼나게 사용가능.
def f(x):
    return Series([x.min(), x.max(), x.mean()],
                    index=["min", "max", "mean"])
df_info.apply(f)

# Applymap for dataframe -> series단위가 아닌 element 단위함수를 적용함.
 # - series단위에 apply를 적용시킬때와 같은효과
 #-> apply는 통계데이터를 뽑을때, applymap은 전체 데이터를 바꿀 유용하다는 특징 기억.


# describe
 - built-in func
 - Numeric type 데이터의 요약정보를 보여줌.
df.describe()

# unique
 # - series data의 유일한 값을 list를 반환.
 # 카테고리형데이터가 몇개인지 모를때 같은경우
 # (성별말고 단과대학같은 많은애들.
 # 이놈들을 enumerate해서 dict로 하면 각카테고리별 딕트만들수있다.
 #  이걸 map혹은 replace로 숫자로 치환가능)
df.race.unique() # 유일한 인종의 값 List
np.array(dict(enumerate(df["race"].unique()))) # dict type으로 index
np.array({0:'white', 1:'other', 2:'hispanic', 3:'black'}, dtype=object)
value=list(map(int, np.array(list(enumerate(df["race"].unique())))[:, 0].tolist()))
key = np.array(list(enumerate(df["race"].unique())), dtype=str)[:,1].tolist()
value, key
# -> label index값과 label 값 각각추출
 # -> 위와 같이 여러 방법이 있당.
