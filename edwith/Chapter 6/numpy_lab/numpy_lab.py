import numpy as np


def n_size_ndarray_creation(n, dtype=np.int):
    X=np.arange(n**2).reshape(n,n)
    return X
print(n_size_ndarray_creation(3))

def zero_or_one_or_empty_ndarray(shape, type=0, dtype=np.int):
    type = float('NaN') if type == 99 else type
    X=np.array(([type]*shape[0]*shape[1])).reshape(shape)
    return X

# def zero_or_one_or_empty_ndarray(shape, type=0, dtype=np.int):
#     if type == 0:
#         X = np.zeros(shape=shape, dtype=dtype)
#     elif type == 1:
#         X = np.ones(shape=shape, dtype=dtype)
#     elif type == 99:
#         X = np.empty(shape=shape, dtype=dtype)     # 요렇게 하면 empty가 아닌 Random #같은데??
#     return X
# print(zero_or_one_or_empty_ndarray((2,2),type=99))
# print(np.empty((3,3), np.int))
def change_shape_of_ndarray(X, n_row):
    return X.reshape(n_row, -1)
# def change_shape_of_ndarray(X, n_row):
#     return X.flatten if n_row==1 else X.reshape(n_row, -1)   # 아. n_row가 1이면 vector니까 껍데기가 1개여야하는구나.

X = np.ones((32,32), dtype=np.int)
change_shape_of_ndarray(X, 1)
print(change_shape_of_ndarray(a, 512))

def concat_ndarray(X_1, X_2, axis):     # 잘못푼건지 맞는건지 확인좀 해주셈.-> 행렬+벡터의 경우 아래것은 안됨.
    if X_1.ndim == 1 or X_2.ndim == 1:
        X_1, X_2 = X_1.reshape(1,-1), X_2.reshape(1,-1)
        # return np.concatenate((X_1, X_2), axis=axis)
    # else:
    axis_temp = 1 if axis == 0 else 0
    if X_1.shape[axis_temp]!= X_2.shape[axis_temp]:
        return False
    return np.concatenate((X_1, X_2), axis=axis)

#결과
a = np.array([[1, 2], [3, 4]])
a.ndim
a.shape
b = np.array([[5, 6]])
b.ndim
b.shape
concat_ndarray(a, b,0)
concat_ndarray(a, b,1)
a = np.array([1, 2])
a.ndim
a.shape
b = np.array([5, 6, 7])
b.ndim

# np.concatenate((a, b), axis = 1)
concat_ndarray(a,b,1)
concat_ndarray(a,b,0)


# def concat_ndarray(X_1, X_2, axis):     # 교수 정답
#     try:
#         if X_1.ndim == 1:
#             X_1 = X_1.reshape(1,-1)
#         if X_2.ndim == 1:
#             X_2 = X_2.reshape(1,-1)
#         return np.concatenate((X_1,X_2), axis=axis)
#     except ValueError as e:
#         return False
#



def normalize_ndarray(X, axis=99, dtype=np.float32):
    if axis==99:
        X_std = np.std(X).reshape(-1,1) # reshape없어도됨.
        X_avg = (np.sum(X)/X.size).reshape(-1,1) #자꾸 쉐입이 안맞데서 reshape해줌.
        return (X -X_avg)/X_std
    elif axis==1:
        X_std = np.std(X, axis=axis).reshape(-1,1)
        X_avg = (np.sum(X, axis=axis)/X.shape[axis]).reshape(-1,1)
        return (X -X_avg)/X_std
    elif axis==0:
        X_std = np.std(X, axis=axis).reshape(1,-1)
        X_avg = (np.sum(X, axis=axis)/X.shape[axis]).reshape(1,-1)
        return (X -X_avg)/X_std
X = np.arange(12, dtype=np.float32).reshape(6,2)
normalize_ndarray(X)
normalize_ndarray(X, 1)
normalize_ndarray(X, 0)


# 교수 정답. np.mean이 있는줄 몰랐다. 그리고 return 이 다 같으면 끝에 한번만 배치하면 되는데 아쉽.
# def normalize_ndarray(X, axis=99, dtype=np.float32):
#     X = X.astype(np.float32)
#     n_row, n_column = X.shape
#     if axis == 99:
#         X_mean = np.mean(X)
#         X_std = np.std(X)
#         Z = (X - X_mean) / X_std
#     if axis == 0:
#         X_mean = np.mean(X,0).reshape(1,-1)
#         X_std = np.std(X,0).reshape(1,-1)
#         Z = (X - X_mean) / X_std
#     if axis == 1:
#         X_mean = np.mean(X,1).reshape(n_row,-1)
#         X_std = np.std(X,1).reshape(n_row,-1)
#         Z = (X - X_mean) / X_std
#     return Z






def save_ndarray(X, filename="test.npy"):
    np.save(filename, arr=X)

X=np.arange(32, dtype=np.float32).reshape(4, -1)
save_ndarray(X, filename="test.npy")


# def boolean_index(X, condition):
    # return X[eval(str("X") + condition)]
# 교수 정답
def boolean_index(X, condition):   # np.where를 사용했는데... 그냥 되겠구나 ㅋ.
    condition = eval(str("X") + condition)
    return np.where(condition)
X=np.arange(40, 60, dtype=np.float32)
condition=">6"
boolean_index(X,condition)


def find_nearest_value(X, target_value): # 교수랑 쌤쌤.
    X_flat_abs=abs(X-target_value)
    return X[np.argmin(X_flat_abs)]

X=np.arange(10)
target_value = 5.3
find_nearest_value(X, target_value)

def get_n_largest_values(X, n):
    X=np.sort(X)
    return X[X.size-n:] # X.size 대신에 len(X) 써도되려나.
# 교수 정답
# arg자 붙으면 index값을 나타내는 함수중에 하나다.
# [::-1]의 의미? 역순으로 정렬해주는거.
# def get_n_largest_values(X, n):
#     return X[np.argsort(X[::-1])[:n]]
X=np.random.uniform(0,1,10)
X.sort()
n=3
print(get_n_largest_values(X, n))

x = np.arange(10)
x
x[::-1]
