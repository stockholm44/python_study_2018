
import numpy as np

# 1
np.array(["1", "4", 5, 8], float)
# 2
np.array(["1", "4", 5, 8], float).shape
# 3
np.array(["1", "4", 5, 8], float).dtype
#4
tensor  = [[[1,2,5,8],[1,2,5,8],[1,2,5,8]],
           [[1,2,5,8],[1,2,5,8],[1,2,5,8]],
           [[1,2,5,8],[1,2,5,8],[1,2,5,8]],
           [[1,2,5,8],[1,2,5,8],[1,2,5,8]]]
ten_array = np.array(tensor, float)
ten_array.shape
ten_array.ndim
ten_array.size

#5
test_matrix = [[1,2,3,4], [1,2,5,8]]
np.array(test_matrix)

#6
np.array(test_matrix).reshape(2,2,2)
#7
np.array(test_matrix).reshape(2,-1)

#8
test_matrix = [[[1,2,3,4], [1,2,5,8]], [[1,2,3,4], [1,2,5,8]]]
np.array(test_matrix).flatten()
np.array(test_matrix).reshape(1,-1)

#9
test_example = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]], int)
test_example[:,2:]
#10
test_example[1,1:3]

#11
test_example[1:3]

#12
test_ex = np.arange(100).reshape(10,10)
test_ex[:,-1].reshape(-1,1)
#13
np.arange(0,5,0.5)
#14
np.arange(0,5,0.5).tolist()
#15
np.zeros(shape = (2,5))
np.zeros((2,5))
#16
np.identity(3)
# np.identity((3,3)) ->요건 에러

#17
test_array = np.arange(1,13).reshape(3,4)
test_array.sum(axis = 1)
#18
test_array.sum(axis = 0)

#19
a = np.array([[1, 2, 3]])
b = np.array([[2, 3, 4]])
np.concatenate((a,b), axis = 0)
#20
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6]])
np.concatenate((a, b.T), axis = 1)
#21
test_a = np.arange(1,7).reshape(2,3)
test_b = np.arange(7,13).reshape(3,2)
test_a + test_b.T
#22
test_a - test_b.T
#23
test_a.dot(test_b)
#24
test_matrix = np.arange(1,13).reshape(4,3)
test_vector = np.arange(10,40,10,20)
test_matrix + test_vector.T
#25
a = np.arange(10)
a
#26
a>0
#27
any(a>5), any(a<0)
#28
all(a>5), all(a<10)
#29
a = np.array([1, 3, 0], float)

np.where(a>0, 3, 2)
#30
np.where(a>0)
#31
a = np.array([1,2,4,5,8,78,23,3])
np.argmax(a), np.argmin(a)

#32
a=np.array([[1,2,4,7],[9,88,6,45],[9,76,3,4]])
b = np.sum(a, axis=1)
b
np.argmax(b)
#33
np.max(a, axis=0)

#34

A = np.array([
[12, 13, 14, 12, 16, 14, 11, 10,  9],
[11, 14, 12, 15, 15, 16, 10, 12, 11],
[10, 12, 12, 15, 14, 16, 10, 12, 12],
[ 9, 11, 16, 15, 14, 16, 15, 12, 10],
[12, 11, 16, 14, 10, 12, 16, 12, 13],
[10, 15, 16, 14, 14, 14, 16, 15, 12],
[13, 17, 14, 10, 14, 11, 14, 15, 10],
[10, 16, 12, 14, 11, 12, 14, 18, 11],
[10, 19, 12, 14, 11, 12, 14, 18, 10],
[14, 22, 17, 19, 16, 17, 18, 17, 13],
[10, 16, 12, 14, 11, 12, 14, 18, 11],
[10, 16, 12, 14, 11, 12, 14, 18, 11],
[10, 19, 12, 14, 11, 12, 14, 18, 10],
[14, 22, 12, 14, 11, 12, 14, 17, 13],
[10, 16, 12, 14, 11, 12, 14, 18, 11]])
B = A>15
B.astype(np.int)

#35
b = np.array([0, 0, 1, 3, 2, 1], int) # 반드시 integer로 선언
a = np.array([2,4,6,8,], float)
a.take(b)
a[b]
