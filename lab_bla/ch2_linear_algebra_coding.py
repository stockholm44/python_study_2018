# # zip 을 사용하여 vector 계산하기
#
# u = [2, 2]
# v = [2, 3]
# z = [3, 5]
# # a = []
# # for i in range(len(u)) :
# #     a.append(u[i] + v[i]+ z[i])
# # print(a)
# sum = [sum(i) for i in zip(u,v,z)]
# print(sum)
#
#
# # Scalar-Vector product
# u = [1, 2, 3]
# v = [4, 5, 6]
# alpha = 2
#
# result = [alpha * sum(z) for z in zip(u, v)]
# print(result)
#
# # Matrix addition
# a  = [[3, 6], [4, 5]]
# b  = [[5, 8], [3, 7]]
# sum = [[sum(row) for row in zip(*t)] for t in zip(a, b)]
# print(sum)
#
# # 왜안되지?? -> 따로 짜면 되네...
# matrix_a  = [[3, 6], [4, 5]]
# matrix_b  = [[5, 8], [6, 7]]
#
# result = [[sum(row) for row in zip(*t)] for t in zip(matrix_a, matrix_b)]
# print(result)
#
# # Scalar-Matrix Product
# matrix_a = [[3,6],[4,5]]
# a = 4
# result = [[a * i for i in j] for j in matrix_a]
# print(result)
#
# # Matrix Transpose
# matrix_a = [[1,2,3],[4,5,6]]
# # result = [[element for element in t] for t in zip(*matrix_a)]
# result = [list(t) for t in zip(*matrix_a)]
# print(result)
#
# # Matrix Product
'''
matrix_a     matrix_b
1 1 2        1 1
2 1 1    *   2 1
             1 3
'''
matrix_a = [[1, 1, 2], [2, 1, 1]]
matrix_b = [[1, 1], [2, 1], [1, 3]]
result1 = [sum(a * b for a, b, in zip(row_a, column_b)) for column_b in zip(*matrix_b) for row_a in matrix_a]
result2 = [[sum(a * b for a, b, in zip(row_a, column_b)) for column_b in zip(*matrix_b)] for row_a in matrix_a] # matrix_a를 뒤로-> 원래 정답
result3 = [[sum(a * b for a, b, in zip(row_a, column_b)) for row_a in matrix_a] for column_b in zip(*matrix_b)] # matrix_b를 뒤로-> 걍 해봄

print(result1)
print(result2)
print(result3)
