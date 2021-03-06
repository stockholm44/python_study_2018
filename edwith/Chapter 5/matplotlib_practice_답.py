# matplotlib

# 1. 그래프그리기.

1. X는 0~99, Y는 0~99로 변수정하고 이를 plot해라
X = range(100)
Y = range(100)
plt.plot(X, Y)

2. X_1은 100까지, Y_1은 y=cos(x), X_2도 100까지, Y_2는 y=sin(x),그리고 추가로 y=x도 plot을 한번에하셈.
X_1 = range(100)
Y_1 = [np.cos(value) for value in X]

X_2 = range(100)
Y_2 = [np.sin(value) for value in X]

plt.plot(X_1, Y_1)
plt.plot(X_2, Y_2)
plt.plot(range(100), range(100))
plt.show()

3. 10*10 inch의 figure set을 만들고 각각 1,2,1/1,2,2의 2개의 판을 넣는다.
   첫번재판엔 위의 cos, 두째판엔 sin 그래프넣어라.
X_1 = range(100)
Y_1 = [np.cos(value) for value in X]

X_2 = range(100)
Y_2 = [np.sin(value) for value in X]

fig = plt.figure() # figure 반환
fig.set_size_inches(10,10) # 크기지정
ax_1 = fig.add_subplot(1,2,1) # 두개의 plot 생성
ax_2 = fig.add_subplot(1,2,2)  # 두개의 plot 생성

ax_1.plot(X_1, Y_1, c="b")  # 첫번째 plot
ax_2.plot(X_2, Y_2, c="g")  # 두번째 plot
plt.show() # show & flush

#color
4. x= 0~100, y=x, y=x+100을 plot해라.단 wjswksms 색깔 '#000000', 후자는 'c'
X_1 = range(100)
Y_1 = [value for value in X]

X_2 = range(100)
Y_2 = [value + 100 for value in X]

plt.plot(X_1, Y_1, color="#000000")
plt.plot(X_2, Y_2, c="c")

plt.show()

#linestyle
5. 위의 것을 활용하여 라인스타일을 다르게해라. 전자는 dashed, 후자는 dotted
plt.plot(X_1, Y_1, c="b", linestyle="dashed")
plt.plot(X_2, Y_2, c="r", ls="dotted")

6. 범례추가해라 위치는 upper right

plt.show()

# 3. 그래프 스타일.
7. data1과 data2의 변수를 지정해라. 512 * 2 shape으로 선언.
data_1 = np.random.rand(512, 2)
data_2 = np.random.rand(512, 2)

8. 각각 plot 한다. 옵션은 scatter그래프, 512 * 1을 x로 512 * 2 를 y로, 색은 한개는 블루 한개는 레드, 마커는 각각x, o로 해라.
plt.scatter(data_1[:,0], data_1[:,1], c="b", marker="x")
plt.scatter(data_2[:,0], data_2[:,1], c="r", marker="o")

plt.show()

# 4. Histogram (바차트나 이런것들 불필요해보여서 걍 뻄.)
9. X에 1000개의 랜덤수를 넣어라,
X = np.random.randn(1000)

10. 구간을 100개로 해서 히스토그램으로 그려라.
plt.hist(X,bins=100)
plt.show()

# 4. Box Plot
11. data에 100*5의 랜덤변수를 넣어라
data = np.random.randn(100,5)

12. 박스플랏으로 플랏해라.
plt.boxplot(data)
plt.show()
