import numpy as np
import pandas as pd

# 함수 구현을 위해 Test하는 영역
if __name__ == "__main__":
    a = np.array([[1,2],[3,4]], dtype=np.float32)
    a
    a_inv = np.linalg.inv(a)
    a_inv
    a.dot(a_inv)
    filename = 'C:/study_2018/python_study_2018/edwith/Chapter 7/assignment_normal_equation/test.csv'
    raw_data = pd.read_csv(filename)
    raw_data.head()
    raw_data["column_vector"] = 1
    raw_data.head()
    X = np.array(raw_data[['column_vector','x']], dtype=np.int)
    y = np.array(raw_data['y'], dtype=np.float32)
    y

    W = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
    W
    W[0]
    X = raw_data['x'].values.reshape(-1,1)
    y = raw_data['y'].values
    type(X)
    y
    XX=X[:5]
    XX.size
    np.ones(XX.size).reshape(-1,1)
    np.concatenate((np.ones(X.size).reshape(-1,1),X), axis=1)
    X
    np.mean(X)
    np.mean(X, axis=0)

class LinearRegression(object):
    def __init__(self, fit_intercept=True, copy_X=True):
        self.fit_intercept = fit_intercept
        self.copy_X = copy_X

        self._coef = None
        self._intercept = None
        self._new_X = None

    def fit(self, X, y):
        if self.fit_intercept == True:
            X = np.concatenate((np.ones(X.size).reshape(-1,1),X), axis=1)
        self.W = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
        return self.W
        # w=(X.T*X)^-1*X.T*Y
        # 교수는 y도 reshape(-1,1)했는데 할 필요있나?
        # 그리고 W 계산할때 dot product도 순차적으로 해도되는데 구지 ( )*( ) 했는데 그럴필요 x
        # 또 교슈는 W계산 마지막에 계산의 편의를 위해 flatten했다는데 그러면 구지 predict할때 또 reshape(-1,1)해줘야한다. 효율적인가?
        """
        Linear regression 모델을 적합한다.
        Matrix X와 Vector Y가 입력 값으로 들어오면 Normal equation을 활용하여, weight값을
        찾는다. 이 때, instance가 생성될 때, fit_intercept 설정에 따라 fit 실행이 달라진다.
        fit을 할 때는 입력되는 X의 값은 반드시 새로운 변수(self._new_X)에 저장
        된 후 실행되어야 한다.
        fit_intercept가 True일 경우:
            - Matrix X의 0번째 Column에 값이 1인 column vector를추가한다.
        적합이 종료된 후 각 변수의 계수(coefficient 또는 weight값을 의미)는 self._coef와
        self._intercept_coef에 저장된다. 이때 self._coef는 numpy array을 각 변수항의
        weight값을 저장한 1차원 vector이며, self._intercept_coef는 상수항의 weight를
        저장한 scalar(float) 이다.
        Parameters
        ----------
        X : numpy array, 2차원 matrix 형태로 [n_samples,n_features] 구조를 가진다
        y : numpy array, 1차원 vector 형태로 [n_targets]의 구조를 가진다.
        Returns
        -------
        self : 현재의 인스턴스가 리턴된다
        """
    def predict(self, X, normalize=False):
        self._mu_X = np.mean(X)
        self._std_X = np.std(X)
        if normalize == True:
            X = (X-self._mu_X)/self._std_X
        if self.fit_intercept == True:
            X = np.concatenate((np.ones(X.size).reshape(-1,1),X), axis=1)
        y = X.dot(self.W)
        self._coef = W[1] # Weight가 W1,W0으로만 된게 아니고 많을떄를 위해 교수는 W[1:]로 표현.
        if self.fit_intercept == True:  #intercept가 없다면 coef가 W[0]이므로 여기는 변경해주자.
            self._intercept = W[0]
        """
        적합된 Linear regression 모델을 사용하여 입력된 Matrix X의 예측값을 반환한다.
        이 때, 입력된 Matrix X는 별도의 전처리가 없는 상태로 입력되는 걸로 가정한다.
        fit_intercept가 True일 경우:
            - Matrix X의 0번째 Column에 값이 1인 column vector를추가한다.
        normalize가 True일 경우:
            - Standard normalization으로 Matrix X의 column 0(상수)를 제외한 모든 값을
              정규화을 실행함
            - 정규화를 할때는 self._mu_X와 self._std_X 에 있는 값을 사용한다.
        Parameters
        ----------
        X : numpy array, 2차원 matrix 형태로 [n_samples,n_features] 구조를 가진다
        Returns
        -------
        y : numpy array, 예측된 값을 1차원 vector 형태로 [n_predicted_targets]의
            구조를 가진다.
        """
        return y

    @property
    def coef(self):
        return self._coef

    @property
    def intercept(self):
        return self._intercept
