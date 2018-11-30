import numpy as np

1
a = np.array(["1", "4", 5, 8], float)

2
a.shape
3
a.dtype

tensor  = [[[1,2,5,8],[1,2,5,8],[1,2,5,8]],
           [[1,2,5,8],[1,2,5,8],[1,2,5,8]],
           [[1,2,5,8],[1,2,5,8],[1,2,5,8]],
           [[1,2,5,8],[1,2,5,8],[1,2,5,8]]]
4
tensor_arr = np.array(tensor)
tensor_arr.shape
tensor_arr.ndim
tensor_arr.size

test_matrix = [[1,2,3,4], [1,2,5,8]]
np.array(test_matrix)
