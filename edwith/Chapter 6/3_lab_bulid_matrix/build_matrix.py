import numpy as np
import pandas as pd


def get_rating_matrix(filename, dtype=np.float32):
    df_data = pd.read_csv(filename, sep=',', header=0)
    df = pd.DataFrame(df_data)
    df_groupby = df.groupby(["source","target"])["rating"].sum()
    df_matrix = df_groupby.unstack(fill_value=0)
    return np.array(df_matrix, dtype=np.float32)
filename = 'C:/study_2018/python_study_2018/edwith/Chapter 6/3_lab_bulid_matrix/movie_rating.csv'
get_rating_matrix(filename)

def get_frequent_matrix(filename, dtype=np.float32):
    df_data = pd.read_csv(filename, sep=',', header=0)
    df = pd.DataFrame(df_data)
    df_groupby = df.groupby(["source","target"])["target"].count()
    df_matrix = df_groupby.unstack()
    return np.array(df_matrix)
filename = 'C:/study_2018/python_study_2018/edwith/Chapter 6/3_lab_bulid_matrix/1000i.csv'
get_frequent_matrix(filename)
