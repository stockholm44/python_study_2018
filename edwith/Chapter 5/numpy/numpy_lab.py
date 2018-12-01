import numpy as np


def n_size_ndarray_creation(n, dtype=np.int):
    X=np.arange(n**2).reshape(n,n)
    return X
# print(n_size_ndarray_creation(3))

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
#         X = np.empty(shape=shape, dtype=dtype)
#     return X
# print(zero_or_one_or_empty_ndarray((2,2),type=99))
# print(np.empty((3,3), np.))
def change_shape_of_ndarray(X, n_row):
    return X.reshape(n_row, -1)

X = np.ones((32,32), dtype=np.int)
print(change_shape_of_ndarray(X, 1))
a = change_shape_of_ndarray(X, 1)

print(change_shape_of_ndarray(a, 512))

def concat_ndarray(X_1, X_2, axis):
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
b = np.array([[5, 6]])
concat_ndarray(a, b,0)
a = np.array([1, 2])
b = np.array([5, 6, 7])
# np.concatenate((a, b), axis = 1)
concat_ndarray(a,b,0)




def normalize_ndarray(X, axis=99, dtype=np.float32):
    if axis==99:
        X_std = np.std(X).reshape(-1,1)
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




def save_ndarray(X, filename="test.npy"):
    np.save(filename, arr=X)

X=np.arange(32, dtype=np.float32).reshape(4, -1)
save_ndarray(X, filename="test.npy")


def boolean_index(X, condition):
    return X[eval(str("X") + condition)]

X=np.arange(20)
condition=">4"
print(boolean_index(Y,condition))


def find_nearest_value(X, target_value):
    X_flat_abs=abs(X-target_value)
    return X[np.argmin(X_flat_abs)]

X=np.arange(10)
target_value = 5.3
find_nearest_value(X, target_value)

def get_n_largest_values(X, n):
    X=np.sort(X)
    return X[X.size-n:]

X=np.random.uniform(0,1,10)
n=3
print(get_n_largest_values(X, n))
