# numpy 강의자료

import numpy as np

#1 Array creation
# shape: numpy array의 object의 dimension 구성을 반환
# dtype: numpy array의 데이터 type을 반환
test_array = np.array(["1", "4", 5, 8], float)
print(test_array)

test_array = np.array([1, 4, 5, "8"], np.float32) # String Type의 데이터를 입력해도
print(test_array)

test_array.dtype   # Array(배열) 전체의 데이터 Type을 반환함
np.array([[1, 4, 5, "8"]], np.float32).shape
test_array.shape   # Array(배열) 의 shape을 반환함

#2 Array shape (vector)
# ndim - number of dimension
# size - data의 갯수
vector  = [1,2,3,4]
np.array(vector, int).shape

matrix  = [[1,2,5,8],[1,2,5,8],[1,2,5,8]]
np.array(matrix, int).shape

tensor  = [[[1,2,5,8],[1,2,5,8],[1,2,5,8]],
           [[1,2,5,8],[1,2,5,8],[1,2,5,8]],
           [[1,2,5,8],[1,2,5,8],[1,2,5,8]],
           [[1,2,5,8],[1,2,5,8],[1,2,5,8]]]
np.array(tensor, int).shape
np.array(tensor, int).ndim
np.array(tensor, int).size

#3 Array dtype
C와 상호호환됨. 근데 거의 float32나 float64정도 씀. 음수안쓸때는 또 맞춰서 쓰긴함.
# nbytes – ndarray object의 메모리 크기를 반환함. 약간 신경쓰면서 만들지만 지금은 필요x.
 # (메모리가 클때나 관리)
np.array([[1, 2, 3], [4.5, "5", "6"]], dtype=np.float32).nbytes
np.array([[1, 2, 3], [4.5, "5", "6"]], dtype=np.float32)
np.array([1], int).nbytes
np.array([1], float).nbytes

#3 Handling shape
# 많이 쓰는 기능.
# reshape - array의 shape의 크기를 변경함.(element의 갯수는 동일)
# 데이터 호출시 y를 가져오는데 vector로 많이 가져옴. sklearn은 martix형태로 해야해서 reshape 필요.

test_matrix = [[1,2,3,4], [1,2,5,8]]
np.array(test_matrix).shape

np.array(test_matrix).reshape(2,2,2)

#이런식으로 -1이되면 -1이 아닌 행 또는 열 기준으로 element를 나눠서 표현
# -1 : size를 기반으로 row 개수 선정.
np.array(test_matrix).reshape(2,-1)

np.array(test_matrix).reshape(2,-1).shape
np.array(test_matrix).reshape(-1,2)
np.array(test_matrix).reshape(-1,2).shape

# flatten - 다차원 array를 vector로 만듬. 유용
# reshape으로 해도 상관은 없다
# 딥러닝 초기 학습시mnist(문자를 vector형태)가 28*28 array인데 이걸 펴줄때사용.
test_matrix = [[[1,2,3,4], [1,2,5,8]], [[1,2,3,4], [1,2,5,8]]]
np.array(test_matrix).flatten()

# 5. Indexing And Slicing
# 일반 list와 약간 표기법이다름.
test_example = np.array([[1, 2, 3], [4.5, 5, 6]], int)
test_example[0][0]
test_example[0,0]

#slicing - 좀 재밋다. 일부분가져올때 굉장히 유용.
# for문으로 data를 가져올때가 있는데 그럴필요없음.
test_exmaple = np.array([[1, 2, 3, 4, 5],
                         [6, 7, 8, 9, 10]], int)
test_exmaple[:,2:] # 전체 Row의 2열 이상
test_exmaple[1,1:3] # 1 Row의 1열 ~ 2열
test_exmaple[1:3] # 1 Row ~ 2Row의 전체

a = np.arange(100).reshape(10,10)
a
a[:, -1].reshape(-1,1)
# arr[::2] 하면 step2로 만 가져와라..

#6. Creation array
# arange - range(x)와 같은 기법.
np.arange(30) # 0~29까지.
np.arange(30).reshape(-1,5) # 5열단위로
np.arange(0,5,0.5) # list와 달리 소수점 단위 step 가능
np.arange(0,5,0.5).tolist() #이렇게 list로 보낼수 있다.

# zeros - 0으로 가득찬 ndarray 생성.
np.zeros(shape=(10,), dtype = np.int8) #10개의 zero vector 생성
np.zeros((2,5)) # 2 by 5 zero matrix생성

#one - 1로 가득찬.
#empty -shape만 주어지고 비어있는 ndarray 생성.(memory initialize가 되어있지않음.)
 # -> 한번도 안써봤다함 ㅎㅎ
#something_like 이것도 많이 안쓰지만 생각보다 딥러닝때 씀. tens
test_matrix = np.arange(30).reshape(5,6)
test_matrix
np.ones_like(test_matrix)

#identity 단위행렬 생성
np.identity(n=3, dtype=np.int8)
np.identity(3)
#eye 대각선만 1인행렬, k값의 시작 index변경이 가능
# 나머지 0인. 선형대수의이 가우스함수 어쩌구할떄 썼다함.
#거의 쓸일은 없당.

#diag 대각 행렬의 값을 추출함.
matrix = np.arange(9).reshape(3,3)
np.diag(matrix)
np.diag(matrix, k = 1) #k는 시작위치

# random smapling
# 분포에 따른 어레ㅣㅇ를 만들수 있다 정도만 알아두자.
np.random.uniform(0,1,10).reshape(2,5)
np.random.normal(0,1,10).reshape(2,5)

# 7. Operation functions
# numpy에서도 많은 계산식을 지원. 리스트의 sum등.
# sum average 등보다는 axis개념이 중요. axis단위로 해당 계산식을 적용할 수 있다는점.

#axis 모든 operation function을 실행할때 기준이 되는 dimension 축
test_array = np.arange(1,13).reshape(3,4)
test_array
test_array.sum(axis=1) # axis 1 즉 열증가방향으로 하므로 열방향으로 sum
test_array.sum(axis=0) # 행증가방향이므로 행방향으로 sum 해줌

#mean std 요런것도 있다..
# 이 외에도 다양한 수학ㄷ연산자를 제공함.


#concat Numpy array를 합치는 함수.
#굉장히 많이사용. 딥러닝에서도 많이 사용.
a = np.array([1, 2, 3])
b = np.array([2, 3, 4])
np.vstack((a,b))
a = np.array([ [1], [2], [3]])
b = np.array([ [2], [3], [4]])
np.hstack((a,b))

#아마 위의 vstack, hstack 보단 concatenate가 더 쉬울듯.
a = np.array([[1, 2, 3]])
b = np.array([[2, 3, 4]])
np.concatenate( (a,b) ,axis=0) # axis 0 기준(새행으로 붙이기)

a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6]])
np.concatenate( (a,b.T) ,axis=1) # b.T이므로 transpose한 [5],[6]을 새열로 붙이는거.


# 8. Array operation
# operations b/t arrays
# Numpy는 array간의 기본적인 사칙 연산을 지원.
test_a = np.array([[1,2,3],[4,5,6]], float)
test_a + test_a # Matrix + Matrix 연산
test_a - test_a # Matrix - Matrix 연산
test_a * test_a # Matrix내 element들 간 같은 위치에 있는 값들끼리 연산
#Array간 shape이 같을 때 일어나는 연산
#Dot product - matrix 기본연산, dot 함수 사용
test_a = np.arange(1,7).reshape(2,3)
test_b = np.arange(7,13).reshape(3,2)
test_a.dot(test_b)

#transpose 전치행렬.
test_a.transpose()
#broadcasting - numpy의 제일 중요한 개념.
 # -> shape이 다른 배열 간 연산을 지원하는 기능.
 # matrix와 scalar와의 연산이 됨.
 # 제일 헷갈림. 지멋대로 일어나는것 처럼 보이니 이해할 것.
test_matrix = np.array([[1,2,3],[4,5,6]], float)
scalar = 3

test_matrix + scalar # Matrix - Scalar 덧셈
test_matrix - scalar # Matrix - Scalar 뺄셈
test_matrix * 5 # Matrix - Scalar 곱셈
test_matrix / 5 # Matrix - Scalar 나눗셈
test_matrix // 0.2 # Matrix - Scalar 몫
test_matrix ** 2 # Matrix - Scalar 제곱

test_matrix = np.arange(1,13).reshape(4,3)
test_vector = np.arange(10,40,10)
test_matrix
test_vector
test_matrix+ test_vector # 각 행별로 더해줌. #Tensorflow에선 많이 쓰임.
#기억하라 브로드캐스팅.


#Numpy performance #1
#timeit - jupyter환경에서 코드의 퍼포먼스를 체크
# 속도순위 1. numpy 2. list comprehension 3. for
# 하지만 concat은 numpy가 느림. 파이썬은 확보후 가르키기만 하면 되는데 numpy는 서치하고 memory 다시할당.
# 큰데이터면 구지 numpy갔다가 list로 널멍올필요없음
def sclar_vector_product(scalar, vector):
    result = []
    for value in vector:
        result.append(scalar * value)
    return result

iternation_max = 100000000
vector = list(range(iternation_max))
scalar = 2
%timeit sclar_vector_product(scalar, vector) # for loop을 이용한 성능
%timeit [scalar * value for value in range(iternation_max)] # list comprehension을 이용한 성능
%timeit np.arange(iternation_max) * scalar # numpy를 이용한 성능

# 9. comparisons
# all & any
# 굉장히 유용한 기능이므로 반드시 기억할 것.
a = np.arange(10)
a
a>0 #일종의 브로드캐스팅이 일어났다고 보면 될듯. 각각 0보다 큰지 여부에 대한 결과값을 반환..
np.any(a>5), np.any(a<0)
np.all(a>5) , np.all(a < 10)

#logical_and - 잘안쓰임.
#요넘들보다는 a>5 이렇게 하면 같은 어레이크기에 true/false반환하는거.
a = np.array([1, 3, 0], float)
np.logical_and(a > 0, a < 3) # and 조건의 condition
b = np.array([True, False, True], bool)
np.logical_not(b) # NOT 조건의 condition

c = np.array([False, True, False], bool)
np.logical_or(b, c) # OR 조건의 condition


# np.where - 조건에 맞는거 찾기. 주로 index값 반환할때 많이씀. numpy에서 정렬하는 기법에서 많이 씀.
a
np.where(a>0, 3, 2) # Ture일때는 3, False면 2반환
np.where(a>0) #위에보다는 요것을 많이씀. 인덱스값을 반환.0,1인덱스 애들이 0보다 크다는것.

np.isnan(a) # Not a Number값을 찾음
np.isfinite(a) # finit number인지 찾음

# argmax & argmin 많이씁니다.
a = np.array([1,2,4,5,8,78,23,3])
np.argmax(a) , np.argmin(a) #최대 최소의 index값 반환

a=np.array([[1,2,4,7],[9,88,6,45],[9,76,3,4]])
np.argmax(a, axis=1) , np.argmin(a, axis=0) # 기준 axis를 설정가능

# numpy에 for문?? 최악이다. 각 이런것들이 있다는 것만 알아두고 검색해서 써라.
# boolean index - numpy이 배열에서 특정조건에 따른 값을 배열형태로 추출가능.
  # -> Comparison operation 함수들도 모두 사용가능
test_array = np.array([1, 4, 0, 2, 3, 8, 9, 7], float)
test_array > 3
test_array[test_array > 3] # 요런식으로 조건에 맞는 애들을 배열로 뽑는 기술
  # 아까는 조건에 맞는 index를 뽑는거고 요건 값을 뽑는거.

#아래 예제는 조건에 맞는애들을 1, 0으로 배열로 type바꿔주는거. one hot 이런데 쓰기좋을듯.
# 굉장히 유용
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
B = A < 15
B
B.astype(np.int)


# fancy index - numpy는 array를 index value로 사용해서 값을 추출하는 방법
a = np.array([2, 4, 6, 8], float)
b = np.array([0, 0, 1, 3, 2, 1], int) # 반드시 integer로 선언
a[b] # bracket index, ,b 배열의 값을 index로 하여 a의 값들을 추출함.
a.take(b) # take함수 : bracket index와 같은효과 -> 요걸더 권장. a[b]는 좀 헷갈림.

# fancy index를 matrix형태로도 가능.
a = np.array[[1, 4], [9, 16]], float)
b = np.array([0, 0, 1, 1, 0], int)
c = np.array([0, 1, 1, 1, 1], int)
a[b,c] # b를 row index, c를 column index로 변환하여 표시함
