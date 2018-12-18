# 비교가 되는 n개의 matrix가 서로 동치인지 확인하여 True 또는 False를 반환함 -------->>>>>????????????????
def is_matrix_equal(*matrix_variables):
    print(*matrix_variables)
    matrix = [[len(set(row)) for row in zip(*t)] for t in zip(*matrix_variables)]
    flatten_list=[element for row in matrix for element in row]
    return len(flatten_list) == flatten_list.count(1)
matrix_x = [[2, 2], [2, 2]]
matrix_y = [[2, 5], [2, 1]]

print (is_matrix_equal(matrix_x, matrix_y, matrix_y, matrix_y)) # Expected value: False
# print (is_matrix_equal(matrix_x, matrix_x)) # Expected value: True


def name_filter(x):
    list = ['Mr','Mrs','Mister']
    return x in list

name = ['xx','Mr','ddd']

print(list(filter(name_filter, name)))
