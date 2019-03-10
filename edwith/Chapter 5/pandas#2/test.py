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
