# matplotlib 빠르게 넘어가겠다.
# 여러가지 그래프를 그리기엔 좀 빡셈
# 필요할때마다 찾아쓸수 있게는 해야지.
#
# pyplot 객체를 사용하여 데이터를 표시
# pyplot 객체에 그래프를 쌓은다음 show로 flush


# 1. 그래프그리기.
import matplotlib.pyplot as plt

X = range(100)
Y = range(100)
plt.plot(X, Y)

# 최대단점은 argument를 kwargs를 받음.
# 고정된 argument가 없어서 alt+tab으로 확인이 어려움... 즉 인수가 겁나많음.

#여기서는 기본적으로 쓰는것만.

# Graph는 원래 figure 객체에 생성됨.
# pyplot 객체 사용시, 기본 figure에 그래프가 그려짐.


import numpy as np

X_1 = range(100)
Y_1 = [np.cos(value) for value in X]

X_2 = range(100)
Y_2 = [np.sin(value) for value in X]

plt.plot(X_1, Y_1)
plt.plot(X_2, Y_2)
plt.plot(range(100), range(100))
plt.show()

# 두개의 그림판을 그려줄수 있다. subplot 코드는 나중에 함봐.
fig = plt.figure() # figure 반환
fig.set_size_inches(10,10) # 크기지정
ax_1 = fig.add_subplot(1,2,1) # 두개의 plot 생성
ax_2 = fig.add_subplot(1,2,2)  # 두개의 plot 생성

ax_1.plot(X_1, Y_1, c="b")  # 첫번째 plot
ax_2.plot(X_2, Y_2, c="g")  # 두번째 plot
plt.show() # show & flush


# 2. 기본 옵션항목
#기본 arg는 알아야지. color, linestyle
#하지만 이런것들은 검색해서 필요할때마다 사용하는 것. arg이름정도만 기억.
#color
X_1 = range(100)
Y_1 = [value for value in X]

X_2 = range(100)
Y_2 = [value + 100 for value in X]

plt.plot(X_1, Y_1, color="#000000")
plt.plot(X_2, Y_2, c="c")

plt.show()
#linestyle
plt.plot(X_1, Y_1, c="b", linestyle="dashed")
plt.plot(X_2, Y_2, c="r", ls="dotted")

plt.show()

#범례도 넣을수 있다.
# 옵션들은 필요할때마다 지정해줘야하므로 이름정도만 기억하자는거.

plt.plot(X_1, Y_1, color="b", linestyle="dashed", label='line_1')
plt.plot(X_2, Y_2, color="r", linestyle="dotted", label='line_2')
plt.legend(shadow=True, fancybox=False, loc="upper right")


# 3. 그래프 스타일도 지정가능
# Scatter
data_1 = np.random.rand(512, 2)
data_1
data_2 = np.random.rand(512, 2)

plt.scatter(data_1[:,0], data_1[:,1], c="b", marker="x")
plt.scatter(data_2[:,0], data_2[:,1], c="r", marker="o")

plt.show()

# scatter는 s라는 영역넓이 크기값 변경이 가능(개인적으로는 괜찮은 기능)
N = 50
x = np.random.rand(N)
y = np.random.rand(N)
colors = np.random.rand(N)
area = np.pi * (15 * np.random.rand(N))**2
plt.scatter(x, y, s=area, c=colors, alpha=0.5)
plt.show()

# Bar Chart 이건 좀 어렵고 귀찮다.

data = [[5., 25., 50., 20.],
        [4., 23., 51., 17],
        [6., 22., 52., 19]]

X = np.arange(0,8,2)

#width가 0.5이므로 막대를 나란히 할려면 0.5의 배수씩 X축대칭이동해야함. 귀찮.
plt.bar(X + 0.00, data[0], color = 'b', width = 0.50)
plt.bar(X + 0.50, data[1], color = 'g', width = 0.50)
plt.bar(X + 1.0, data[2], color = 'r', width = 0.50)
plt.xticks(X+0.50, ("A","B","C", "D"))
plt.show()

# 누적그래프는 더 귀찮.. 필요할때 찾아봐라.
data = np.array([[5., 25., 50., 20.],
        [4., 23., 51., 17],
        [6., 22., 52., 19]])

color_list = ['b', 'g', 'r']
data_label = ["A","B","C"]
X = np.arange(data.shape[1])
X

data = np.array([[5., 5., 5., 5.],
        [4., 23., 51., 17],
        [6., 22., 52., 19]])

for i in range(3):
    plt.bar(X, data[i], bottom = np.sum(data[:i], axis=0),
            color = color_list[i], label=data_label[i])
plt.legend()
plt.show()


# Histogram

X = np.random.randn(1000)
plt.hist(X,bins=100)
plt.show()

# Box plot

data = np.random.randn(100,5)
plt.boxplot(data)
plt.show()


#중요한것은 이것들을 pandas와 함께 이용한다는 거.
# 아래들은 그냥 과제로 넘겨주자. 즉 강의내용 재현해보자는 거.
# http://lib.stat.cmu.edu/datasets/boston  여기서 14가지column에 해당하는 Data들의 상관관계를 그래프로 그나마 분석.
# MEDV가 중간집값이고 나머지들은 관련된 항목들.. 각각을 MEDV와 비교하면 선형관계를 알 수 있을 듯.
# 코드는 아래와 같으므로 데이터 불러오는거부터해서 구현하게 하자.
# 데이터에 대한 약간의 한글설명은 요사이트 http://www.dator.co.kr/ctg258/textyle/1721307

# 데이터의 상관관계를 볼 때 scatter graph 사용가능
fig = plt.figure()
ax=[]
for i in range(1,5):
    ax.append(fig.add_subplot(2,2,i))
ax[0].scatter(df_data["CRIM"], df_data["MEDV"])
ax[1].scatter(df_data["PTRATIO"], df_data["MEDV"])
ax[2].scatter(df_data["AGE"], df_data["MEDV"])
ax[3].scatter(df_data["NOX"], df_data["MEDV"])

# 위 그래프 꾸미기..
ax[0].scatter(df_data["CRIM"], df_data["MEDV"], color='b', lable="CRIM")
ax[1].scatter(df_data["PTRATIO"], df_data["MEDV"], color='g')
ax[2].scatter(df_data["AGE"], df_data["MEDV"])
ax[3].scatter(df_data["NOX"], df_data["MEDV"])
plt.subplots_adjust(wspace=0, hspace=0)
ax[0].legend()
ax[0].set_title("CRIM")

#Histogam으로.
fig = plt.figure()
fit.set_size_inches(10,5)
ax_1=fig.add_subplot(1,2,1)
ax_2=fig.add_subplot(1,2,2)
ax_1.plot(df_data["MEDV"])
ax_2.hist(df_data["MEDV"], bins=50)
ax_1.set_title("House price MEDV")
ax_2.set_title("House price MEDV") #히스토그램으로 보니까 좀 보인다.

# Scaled boxplot
# Scatter mattrix 등등.여러개 있는데 할지말지 상황봐서.교수가 코드를 제공안해서 일일이 치기 귀찮.
# Scatter matrix 색깔로 표현한거... 온도 그래프라고 하는데 나중에 해보는 거 정도.
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(corr_data, vmin=-1, vmax=1, interpolation='nearest')
fig.colorbar(cax)
fig.set_size_inches(10,10)
ticks=np.arange(0,14,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)

ax.set_xticklabels(df_data.columns)
ax.set_yticklabels(df_data.columns)
