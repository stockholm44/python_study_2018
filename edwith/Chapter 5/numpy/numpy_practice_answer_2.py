#1 Array creation
# 1. ["1", "4", 5, 8] 를 numpy로 Float로 선언
import numpy as np
a = ["1", "4", 5, 8]
aa = np.array(a, dtype=np.float32)


# 2. 위의 것 shape확인.
aa.shape
# 3. 전체 type확인.
aa.dtype

tensor  = [[[1,2,5,8],[1,2,5,8],[1,2,5,8]],
           [[1,2,5,8],[1,2,5,8],[1,2,5,8]],
           [[1,2,5,8],[1,2,5,8],[1,2,5,8]],
           [[1,2,5,8],[1,2,5,8],[1,2,5,8]]]
4. 위의 tensor를 array로 선언하고 쉐입, 차원, 사이즈 확인
tt=np.array(tensor)
tt
#3 Handling shape
test_matrix = [[1,2,3,4], [1,2,5,8]]
# 5. 위 행렬을 array로 선언하기
nn = np.array(test_matrix)
# 6. 2*2*2 행렬로 변환
nn.reshape(2,2,2)
7. 2행짜리로 변환하고 열은 element갯수에 맞춰서 만들기
nn.reshape(2,-1)
test_matrix = [[[1,2,3,4], [1,2,5,8]], [[1,2,3,4], [1,2,5,8]]]
test_array = np.array(test_matrix)
# 8. 위행렬을 flatten 시키기. 2가지방법으로
test_array.flatten()
test_array.reshape(1,-1)

# 5. Indexing And Slicing
test_exmaple = np.array([[1, 2, 3, 4, 5],
                         [6, 7, 8, 9, 10]], int)

# 9. 위행렬을 행:전체, 열:2열이상 선택하기
test_exmaple[:, 2:]
# 10. 행:1행, 열: 1~2열 선택
test_exmaple[1, 1:3]
# 11. 행:1~2행 열:전체 선택
test_exmaple[1:3, :]

12. 0~99까지를 10*10짜리 array만들고 이중 9, 19, 29,와 같이 9로 끝나는애들을 1열짜리로 만들어라.
np.arange(0,100).reshape(-1,10)[:,-1].reshape(-1,1)


#6. Creation array
13. 0에서 5까지 0.5 step으로 만들어라
a = np.arange(0, 5.5, 0.5)
14. 위 array를 list로 변환해라.
list(a)
15. 0으로 가득한 2*5 matrix만들어라
np.zeros((2,5))
16. 3*3짜리 identity 행렬만들어라
np.identity(3)
# 7. Operation functions
test_array = np.arange(1,13).reshape(3,4)
test_array
# 17. 위 어래이를 열방향으로 sum한 array를 반환.
test_array.sum(axis=1)
18. 위 어래이를 행방향으로 sum한 것을 반환
test_array.sum(axis=0)
a = np.array([[1, 2, 3]])
b = np.array([[2, 3, 4]])
19. 위 두개를 a 아래행에 b행이오게 붙여라
np.vstack((a,b))


a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6]])
20. 위 a의 2*2에 b를 새열로 들어가게 해서 2*3으로 만들어라.(힌트 Tr)
np.concatenate((a,b.T), axis=1)
# 8. Array operation
test_a = np.arange(1,7).reshape(2,3)
test_b = np.arange(7,13).reshape(3,2)
21. a기준으로 2*3행렬이 되게 a와 b를 element 더하기해라
test_a
test_b
test_a + test_b.T

22. a기준으로 2*3행렬이 되게 a와 b를 element 빼라해라
test_a - test_b.T

23. a, b array를 dot product해라.
test_a.dot(test_b)

test_matrix = np.arange(1,13).reshape(4,3)
test_vector = np.arange(10,40,10)
24. 위 matrix를 행단위로 vector와 더하게 하라.
test_matrix
test_vector
test_matrix+test_vector
# 9. comparisons
25. a라는 벡터를 0~9 element로 선언하라.
a = np.arange(0,10)
a
26. 0보다 큰애들은 true, 아닌애들 false로 array를 행렬만들어라
a>0
27. (5보다큰것이 있는지, 0보다 작은애가 있는지) 로 결과값받아라
any(a>5), any(a<0)
28. (전부 5보다 큰지, 전부 10보다 작은지)로 결과값받아라.
all(a>5), all(a<=10)
a = np.array([1, 3, 0], float)
29. 0보다 크면 3을, 아니면 2를 element로 받게 하라
np.where(a>0, 3, 2)
30. 0보다 큰애들의 인덱스만 array로 받아라
np.where(a>0)
a = np.array([1,2,4,5,8,78,23,3])
31. 최대값/최소값의 index를 있게 array반환
np.argmax(a), np.argmin(a)


a=np.array([[1,2,4,7],[9,88,6,45],[9,76,3,4]])
32. 열방향으로 sum했을때 최대값이 있는 index를 반환하라.
np.argmax(a.sum(axis=1))
a.sum(axis=1)
33. 행방향끼리비교해서 최대값들을 array로 받아라.
a.max(axis=0)


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
34. 위 array중 15보다 큰애들만 1, 작은애들은 0으로 되게 array를 받아라. (astype이용)
np.where(A>15, 1,0)
b = np.array([0, 0, 1, 3, 2, 1], int) # 반드시 integer로 선언
35. 0->2, 1->4, 2->6, 3->8이 되게 위 b array를 변환해서 반환하라. 기준이되는 a를 만들고 2가지 방법으로 되게하라.
dict_a = {0:2, 1:4, 2:6, 3:8}
np.where(b, dict_a)
