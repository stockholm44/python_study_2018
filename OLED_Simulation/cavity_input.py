# Cavity 시뮬레이션의 재료 물성치. 재료별 n, k값과 EML의 PL값.
import numpy as np
import pandas as pd
import os
import scipy.interpolate as ipl

# n, k값의 경로와 PL의 경로
ito_path = "../nk/ITO_nk.csv"
al_path = "../nk/Al_nk.csv"
npd_path = "../nk/NPD_nk.csv"
air_path = "../nk/Air_nk.csv"
YG_PL_path = "../nk/YG_PL.csv"
B_PL_path = "../nk/B_PL.csv"
R_PL_path = "../nk/Rss_PL.csv"


# 논문에 있는 nk, pl들
ito_new_path = "../nk/nk2/new/ITO_nk.csv"
al_new_path = "../nk/nk2/new/Al_nk.csv"
ag_new_path = "../nk/nk2/new/Ag_nk.csv"
alq3_new_path = "../nk/nk2/new/Alq3_nk.csv"
alq3_PL_path = "../nk/nk2/new/Alq3_pl.csv"



# 입력된 n, k 경로로 n, k를 ndarry 객체로 가져오는 class Get_nk
def get_nk(path):
    TEMP_nk = pd.read_csv(path, sep=',') # read DataFrame
    # nk값 내삽
    x = TEMP_nk['wavelength']
    
    y1 = TEMP_nk['n']
    y2 = TEMP_nk['k']
    x_reg = np.arange(380,784,4)
    f1 = ipl.interp1d(x, y1, bounds_error=False)
    f2 = ipl.interp1d(x, y2, bounds_error=False)
    y1_new = f1(x_reg)
    y2_new = f2(x_reg)
    TEMP_nk['n'] = y1_new
    TEMP_nk['k'] = y2_new
    nk = TEMP_nk.values[:, 1:].astype(np.float64) # (101, 2)

    n = nk[:, 0]
    k = nk[:, 1]
    nk_complex = n + k * (1j)
#    nk_complex = n + k * (1j) * (-1)
    return nk_complex.reshape(-1, 1) # (101, 2)

# (101, 1) 차원으로 nk에서 real part만 가져와서 n값을 가져오는 method
# theta 계산 및 ambient layer 투과율 계산시 사용.
def get_n(nk):
    return np.real(nk) # (101,1)


# PL data 불러들어오기.
def get_pl(path):
    ''' Get the path and make a 101 * 1 numpy array.'''
    try:
        TEMP_PL = pd.read_csv(path, sep=',', header=None)
    except IndexError:
        TEMP_PL = pd.read_csv(path, sep='\t', header=None)
    # nk값 내삽
    x = TEMP_PL[0]
    
    y = TEMP_PL[1]
    x_reg = np.arange(380,784,4)
    f = ipl.interp1d(x, y, bounds_error=False)
    y_new = f(x_reg)
    
    result = y_new.reshape(-1,1) # (101, 2)


#        PL = TEMP_PL.values
#        result = PL[:, 1].reshape(-1, 1)


    return result # (101,1)




# 임시 get_pl()
def get_pl2(path):
    TEMP_PL = pd.read_csv(path, sep=',') # read DataFrame
    PL = TEMP_PL.values
    return PL[:, 1].reshape(-1, 1) # (101,1)


# 각도를 (101, 1) shape으로 해당 발광층에서의 spectrum 별 각도를 만드는 method
def calc_theta(n, theta_ex):
    theta_radian = np.radians(theta_ex)
    return (np.arcsin(np.ones((101, 1)) / get_n(n) * np.sin(theta_radian)))

# Make 101 X 1 matrix full with 1.
def ones():
    return np.ones((101, 1))
# Make 101 X 1 matrix full with 0.
def zeros():
    return np.zeros((101, 1))
# Make 101 X 1 matrix of wavelength 380~780 with 4nm step
def wavelength():
    return np.arange(380, 784, 4).reshape(-1, 1) # matrix_s, p에 모두 활용되는 380~780짜리 
# Make 101 X 1 matrix of k for calculating phase change and layer matrix
def k(nk, theta):
    return 2 * np.pi / wavelength() * nk * np.cos(theta)

# RT1으로 하지말고 그냥 interface라고 하는 Class에 차라리 RT method를 집어 넣어버릳자.
# 2. Interface Matrix 
class Interface_Matrix:
    """Calculate 2 by 2 Interface matrix with each two layer(layer left, layer right.)
       nk_left : Refractance index of the layer which is close with reflective metal layer.(Get_nk() method, (101,1)) 
       nk_right : Refractance index of the layer which is far from  reflective metal layer.(Get_nk() method, (101,1))
       theta_ex : Angle of final observation in the air(calc_theta() method, (101,1))
    """
    def __init__(self, nk_left, nk_right, theta_ex):
        self.nk_left = nk_left
        self.nk_right = nk_right 
        
#        theta_rad = np.radians(theta_ex)        
        
        self.theta1 = calc_theta(get_n(self.nk_left), theta_ex)
        self.theta2 = calc_theta(get_n(self.nk_right), theta_ex)
        

    def r_s(self):
        return ((self.nk_left)*np.cos(self.theta1) - (self.nk_right)*np.cos(self.theta2)) /  ((self.nk_left)*np.cos(self.theta1) + (self.nk_right)*np.cos(self.theta2))
    
    def r_p(self):
#        return ((self.nk_left)*np.cos(self.theta2) - (self.nk_right)*np.cos(self.theta1)) /  ((self.nk_left)*np.cos(self.theta2) + (self.nk_right)*np.cos(self.theta1))
        result =  (np.square(self.nk_right)*self.nk_left*np.cos(self.theta1) - np.square(self.nk_left)*self.nk_right*np.cos(self.theta2))/ \
                  (np.square(self.nk_right)*self.nk_left*np.cos(self.theta1) + np.square(self.nk_left)*self.nk_right*np.cos(self.theta2))
        return result
    
    def t_s(self):
        return (2 * (self.nk_left)*np.cos(self.theta1))/((self.nk_left)*np.cos(self.theta1) + (self.nk_right)*np.cos(self.theta2))

    def t_p(self):
#        return (2 * (self.nk_left)*np.cos(self.theta1))/((self.nk_left)*np.cos(self.theta2) + (self.nk_right)*np.cos(self.theta1))
        result = (2 * self.nk_left * self.nk_right * (self.nk_left * np.cos(self.theta1))) / \
                 (np.square(self.nk_right) * (self.nk_left * np.cos(self.theta2)) + np.square(self.nk_left) * (self.nk_right * np.cos(self.theta1)))
        return result
    
    def matrix(self, sp):
        if sp == 's':
            result_b = 1 / self.t_s() * np.array([[ones(), self.r_s()], [self.r_s(), ones()]])
        elif sp == 'p':
            result_b = 1 / self.t_p() * np.array([[ones(), self.r_p()], [self.r_p(), ones()]])
        result_c = result_b.squeeze()
        result_d = np.einsum('ijk->kij',result_c)
        return result_d

class Layer_Matrix:
    def __init__(self, nk, distance, theta_ex):
        self.nk = nk 
        self.d = distance              
        
        self.theta = calc_theta(get_n(self.nk), theta_ex)
        
    def matrix(self, sp):
        result_a = np.array([[np.exp((-1j) * k(self.nk, self.theta) * self.d), zeros()], [zeros(), np.exp((1j) * k(self.nk, self.theta) * self.d)]])
        result_b = np.einsum('ijk->kij', result_a.squeeze())
        return result_b
    
    
    
if __name__ == '__main__':
    # n, k값의 경로와 PL의 경로
    ito_path = "../nk/ITO_nk.csv"
    al_path = "../nk/Al_nk.csv"
    npd_path = "../nk/NPD_nk.csv"
    air_path = "../nk/Air_nk.csv"
    Glass_path = "../nk/Glass_nk.csv"
    
    YG_PL_path = "../nk/YG_PL.csv"
    B_PL_path = "../nk/B_PL.csv"
    R_PL_path = "../nk/R_PL.csv"
    
    pl = get_pl(YG_PL_path)
    print(pl)
    print(pl.shape)
    
#    # 논문에 있는 nk, pl들
#    ito_new_path = "../nk/nk2/new/ITO_nk.csv"
#    al_new_path = "../nk/nk2/new/Al_nk.csv"
#    ag_new_path = "../nk/nk2/new/Ag_nk.csv"
#    alq3_new_path = "../nk/nk2/new/Alq3_nk.csv"
#    alq3_PL_path = "../nk/nk2/new/Alq3_pl.csv"
#
#
#    
#    ito_new = get_nk(ito_new_path)
#    al_new = get_nk(al_new_path)
#    ag_new = get_nk(ag_new_path)
#    alq3_new = get_nk(alq3_new_path)
#    alq3_pl = get_pl(alq3_PL_path)
#    
    # 주요 nk 값들은 그냥 갖고 오는게 좋을 듯 하다.
    # 그리고 nk_complex아 n을 구분하지 말고 n = np.imag(nk_complex)로 한단계 계산하는게 나을듯하다.
#    al = get_nk(al_path)
#    npd = get_nk(npd_path)

#    a = Interface_Matrix(al, npd, theta_ex=0)
#    print(a.matrix('s'))
#    print(a.matrix('s').shape)
    
#    import matplotlib.pyplot as plt
#    # calc_theta 검증
#    fig = plt.figure()
#    ax = fig.add_subplot(111)
#    x = np.arange(380, 784, 4).reshape(-1, 1)
#    for theta in [0, 15, 30, 45, 60]:
#        n_theta = calc_theta(npd, theta)
#        ax.plot(x, n_theta, label=theta)
#    ax.legend()
#    plt.show()
#    
#    # k 검증
#    fig = plt.figure()
#    ax = fig.add_subplot(111)
#    x = np.arange(380, 784, 4).reshape(-1, 1)
#    for theta in [0, 15, 30, 45, 60]:
#        k_theta = k(npd, theta)
#        ax.plot(x, n_theta, label=theta)
#    ax.legend()
#    plt.show()
#    
#    # cos(theta) 검증
#    fig = plt.figure()
#    ax = fig.add_subplot(111)
#    x = np.arange(380, 784, 4).reshape(-1, 1)
#    for theta in [0, 15, 30, 45, 60]:
#        cos_theta = np.cos(calc_theta(npd, theta))
#        ax.plot(x, cos_theta, label=theta)
#    ax.legend()
#    plt.show()
#    
    