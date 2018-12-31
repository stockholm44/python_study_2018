# matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. 그래프그리기.

1. X는 0~99, Y는 0~99로 변수정하고 이를 plot해라
X = np.arange(100)
Y = np.arange(100)
plt.plot(X, Y)

2. X_1은 100까지, Y_1은 y=cos(x), X_2도 100까지, Y_2는 y=sin(x),그리고 추가로 y=x도 plot을 한번에하셈.
X_1 = np.arange(100)
Y_1 = [np.cos(x) for x in X_1 ]
Y_2 = [np.sin(x) for x in X_1]
plt.plot(X_1, Y_1)
plt.plot(X_1, Y_2)
plt.plot(X_1, X_1)
plt.show()
3. 10*10 inch의 figure set을 만들고 각각 1,2,1/1,2,2의 2개의 판을 넣는다.
   첫번재판엔 위의 cos, 두째판엔 sin 그래프넣어라.
fig = plt.figure()
fig.set_size_inches(10,10)
ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)
ax1.plot(X_1, Y_1, c='b')
ax2.plot(X_1, Y_2, c='r')
plt.show()
#color
4. x= 0~100, y=x, y=x+100을 plot해라.단 전자는 색깔 '#000000', 후자는 'c'
X = range(100)
Y_1 = [x for x in X]
Y_2 = [x + 100 for x in X]
fig = plt.figure()
fig.set_size_inches(10, 10)
ax_3 = fig.add_subplot(1,2,1)
ax_4 = fig.add_subplot(1,2,2)
ax_3.plot(X, Y_1, c = "#000000")
ax_4.plot(X, Y_2, c = "c")
fig.show()


#linestyle
5. 위의 것을 활용하여 라인스타일을 다르게해라. 전자는 dashed, 후자는 dotted

plt.plot(X, Y_1, c = "#000000", ls='dashed', label="line_1")
plt.plot(X, Y_2, c = "c", ls='dotted', label="line_2")
plt.legend(shadow=True, fancybox=False, loc="upper right")
plt.show()
6. 범례추가해라 위치는 upper right



# 3. 그래프 스타일.
7. data1과 data2의 변수를 지정해라. 512 * 2 shape으로 선언.
data1 = np.random.randn(512,2)
data2 = np.random.randn(512,2)
data1
8. 각각 plot 한다. 옵션은 scatter그래프, 512 * 1을 x로 512 * 2 를 y로, 색은 한개는 블루 한개는 레드, 마커는 각각x, o로 해라.
plt.scatter(data1[:,0],data1[:,1], c='b', marker='o')
plt.scatter(data2[:,0],data2[:,1], c='r', marker='x')
plt.show()

# 4. Histogram (바차트나 이런것들 불필요해보여서 걍 뻄.)
9. X에 1000개의 랜덤수를 넣어라,
X = np.random.randn(1000)
plt.hist(X, bins=100)
10. 구간을 100개로 해서 히스토그램으로 그려라.
# 4. Box Plot
11. data에 100*5의 랜덤변수를 넣어라
data = np.random.randn(100, 5)
data
12. 박스플랏으로 플랏해라.
plt.boxplot(data)
