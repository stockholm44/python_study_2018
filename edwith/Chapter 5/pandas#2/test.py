import pandas as pd
import numpy as np

a = {'a':[1,2,3,4], 'b':[11,22,33,44], 'c':[2,3,6,22]}
df = pd.DataFrame(a)
df
df.columns
df.columns[0]
df.columns.delete
df.columns.delete(0)
df.columns
df.columns[:-1]


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



a = "a"
b = 'b'
a + b
