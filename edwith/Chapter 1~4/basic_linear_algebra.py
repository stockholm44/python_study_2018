# vector 간 덧셈 또는 뺄셈 연산을 할 때, 연산이 가능한 사이즈인지를 확인하여 가능 여부를 True 또는 False로 반환함
def vector_size_check(*vector_variables):
    result = True if len(set([len(size) for size in vector_variables])) == 1 else False
    return result
# print(vector_size_check([1,2,3], [2,3,4], [5,6,7])) # Expected value: True
# print(vector_size_check([1, 3], [2,4], [6,7])) # Expected value: True
# print(vector_size_check([1, 3, 4], [4], [6,7])) # Expected value: False

# vector간 덧셈을 실행하여 결과를 반환함, 단 입력되는 vector의 갯수와 크기는 일정하지 않음
def vector_addition(*vector_variables):
    result = [sum(row) for row in zip(vector_variable)]
    return result
# print(vector_addition([1, 3], [2, 4], [6, 7])) # Expected value: [9, 14]
# print(vector_addition([1, 5], [10, 4], [4, 7])) # Expected value: [15, 16]
# print(vector_addition([1, 3, 4], [4], [6,7])) # Expected value: ArithmeticError

# vector간 뺄셈을 실행하여 결과를 반환함, 단 입력되는 vector의 갯수와 크기는 일정하지 않음
def vector_subtraction(*vector_variables):
    # print(*vector_variables)
    # print(vector_variables)
    # print(vector_variables[0])
    if vector_size_check(*vector_variables) == False:
        raise ArithmeticError
    result = [(row[0] - sum(row[1:])) for row in zip(*vector_variables)]
    # result = map(lambda a: a*(-1), result)

    return result
# print(vector_subtraction([1, 3], [2, 4])) # Expected value: [-1, -1]
# print(vector_subtraction([1, 5], [10, 4], [4, 7])) # Expected value: [-13, -6]

# 하나의 scalar 값을 vector에 곱함, 단 입력되는 vector의 크기는 일정하지 않음
def scalar_vector_product(alpha, vector_variable):
    result = [alpha * element for element in vector_variable]
    return result
# print (scalar_vector_product(5,[1,2,3])) # Expected value: [5, 10, 15]
# print (scalar_vector_product(3,[2,2])) # Expected value: [6, 6]
# print (scalar_vector_product(4,[1])) # Expected value: [4]

# matrix 간 덧셈 또는 뺄셈 연산을 할 때, 연산이 가능한 사이즈인지를 확인하여 가능 여부를 True 또는 False로 반환함
def matrix_size_check(*matrix_variables):
    len_list = [[len(list) for list in matrix] for matrix in matrix_variables]
    print(len_list)
    result = all([False for list in len_list if not list == len_list[0]])
    return result

matrix_x = [[2, 2], [2, 2], [2, 2]]
matrix_y = [[2, 5], [2, 1]]
matrix_z = [[2, 4], [5, 3]]
matrix_w = [[2, 5], [1, 1], [2, 2]]

print (matrix_size_check(matrix_x, matrix_y, matrix_z)) # Expected value: False
print (matrix_size_check(matrix_y, matrix_z)) # Expected value: True
print (matrix_size_check(matrix_x, matrix_w)) # Expected value: True

# 비교가 되는 n개의 matrix가 서로 동치인지 확인하여 True 또는 False를 반환함 -------->>>>>????????????????
def is_matrix_equal(*matrix_variables):
    matrix = [[len(set(row)) for row in zip(*t)] for t in zip(*matrix_variables)]
    flatten_list=[element for row in matrix for element in row]
    return len(flatten_list) == flatten_list.count(1)
matrix_x = [[2, 2], [2, 2]]
matrix_y = [[2, 5], [2, 1]]

print (is_matrix_equal(matrix_x, matrix_y, matrix_y, matrix_y)) # Expected value: False
print (is_matrix_equal(matrix_x, matrix_x)) # Expected value: True


# matrix간 덧셈을 실행하여 결과를 반환함, 단 입력되는 matrix의 갯수와 크기는 일정하지 않음
def matrix_addition(*matrix_variables):
    if matrix_size_check(*matrix_variables) == False:
        raise ArithmeticError
    result = [[sum(element) for element in zip(*t)] for t in zip(*matrix_variables)]
    return result
# 실행결과
matrix_x = [[2, 2], [2, 2]]
matrix_y = [[2, 5], [2, 1]]
matrix_z = [[2, 4], [5, 3]]

# print (matrix_addition(matrix_x, matrix_y)) # Expected value: [[4, 7], [4, 3]]
# print (matrix_addition(matrix_x, matrix_y, matrix_z)) # Expected value: [[6, 11], [9, 6]]

# matrix간 뺄셈을 실행하여 결과를 반환함, 단 입력되는 matrix의 갯수와 크기는 일정하지 않음
def matrix_subtraction(*matrix_variables):
    if matrix_size_check(*matrix_variables) == False:
        raise ArithmeticError
    result = [[element[0]-sum(element[1:]) for element in zip(*t)] for t in zip(*matrix_variables)]
    return result
# 실행결과
# matrix_x = [[2, 2], [2, 2]]
# matrix_y = [[2, 5], [2, 1]]
# matrix_z = [[2, 4], [5, 3]]

# print (matrix_subtraction(matrix_x, matrix_y)) # Expected value: [[0, -3], [0, 1]]
# print (matrix_subtraction(matrix_x, matrix_y, matrix_z)) # Expected value: [[-2, -7], [-5, -2]]

# matrix의 역행렬을 구하여 결과를 반환함, 단 입력되는 matrix의 크기는 일정하지 않음
def matrix_transpose(matrix_variable):
    return None
# 실행결과
# matrix_w = [[2, 5], [1, 1], [2, 2]]
# matrix_transpose(matrix_w)

# 하나의 scalar 값을 matrix에 곱함, 단 입력되는 matrix의 크기는 일정하지 않음
def scalar_matrix_product(alpha, matrix_variable):
    result = [[alpha * element for element in row] for row in matrix_variable]
    return result
# # 실행결과
# matrix_x = [[2, 2], [2, 2], [2, 2]]
# matrix_y = [[2, 5], [2, 1]]
# matrix_z = [[2, 4], [5, 3]]
# matrix_w = [[2, 5], [1, 1], [2, 2]]
#
# print(scalar_matrix_product(3, matrix_x)) #Expected value: [[6, 6], [6, 6], [6, 6]]
# print(scalar_matrix_product(2, matrix_y)) #Expected value: [[4, 10], [4, 2]]
# print(scalar_matrix_product(4, matrix_z)) #Expected value: [[8, 16], [20, 12]]
# print(scalar_matrix_product(3, matrix_w)) #Expected value: [[6, 15], [3, 3], [6, 6]]

# 두 개의 matrix가 입력 되었을 경우, 두 matrix의 곱셈 연산의 가능 여부를 True 또는 False로 반환함
def is_product_availability_matrix(matrix_a, matrix_b):
    if matrix_a == False or matrix_b == False: print("matrix_w가 없습니다.")
    print(matrix_a, matrix_b)
    len_a = [len(matrix_a), len(list(zip(*matrix_a)))]
    len_b  = [len(matrix_b), len(list(zip(*matrix_b)))]
    return len_a[1] == len_b[0]
# 실행결과
matrix_x= [[2, 5], [1, 1]] # 2*2
matrix_y = [[1, 1, 2], [2, 1, 1]] # 2*3
matrix_z = [[2, 4], [5, 3], [1, 3]] # 3*2

print(is_product_availability_matrix(matrix_y, matrix_z)) # Expected value: True
print(is_product_availability_matrix(matrix_z, matrix_x)) # Expected value: True
print(is_product_availability_matrix(matrix_z, matrix_w)) # Expected value: False //matrix_w가없습니다
print(is_product_availability_matrix(matrix_x, matrix_x)) # Expected value: True

# 곱셈 연산이 가능한 두 개의 matrix의 곱셈을 실행하여 반환함
def matrix_product(matrix_a, matrix_b):
    if is_product_availability_matrix(matrix_a, matrix_b) == False:
        # raise ArithmeticError
        print(ㅈ)
    result = [[sum(list(map(lambda a, b: a * b , row_a, column_b))) for column_b in zip(*matrix_b)] for row_a in matrix_a]
    # result = [[list(map(lambda a, b: a * b , row_a, column_b)) for column_b in zip(*matrix_b)] for row_a in matrix_a]
    return result
# 실행결과
matrix_x= [[2, 5], [1, 1]]
matrix_y = [[1, 1, 2], [2, 1, 1]]
matrix_z = [[2, 4], [5, 3], [1, 3]]

print(matrix_product(matrix_y, matrix_z)) # Expected value: [[9, 13], [10, 14]]
print(matrix_product(matrix_z, matrix_x)) # Expected value: [[8, 14], [13, 28], [5, 8]]
print(matrix_product(matrix_x, matrix_x)) # Expected value: [[9, 15], [3, 6]]
print(matrix_product(matrix_z, matrix_w)) # Expected value: False
