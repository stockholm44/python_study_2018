# System Matrix, R, T 계산 후 최종 Intensity 계산.
import numpy as np
import pandas as pd
from cavity_input import *
import matplotlib.pyplot as plt

    
# 10. Total Output spectral radient intensity
# 원래는 Intensity만 계산하는 class 였는데
# 지금은 system_matrix, RT2, Power_RT, K, Phase_Change1/2 모두 여기에 method로 넣어서 1개 클래스로 만듬.
# args는 layer matrix와 interface matrix function들이 들어있는 리스드다.[[system_A 메서드들],[system_B 메서드들]]
class Intensity:
    def __init__(self, pl, nk_e, z_ex, Z, theta_ex, horizontal_dipole_ratio, *args):
        self.pl = pl
        self.nk_e = nk_e
        self.z_ex = z_ex
        self.Z = Z
        self.theta = calc_theta(get_n(self.nk_e), theta_ex)
#        print("cavity_calc : 18 ★★★★★★ horizontal_dipole_ratio  :", horizontal_dipole_ratio )
        self.horizontal_dipole_ratio = horizontal_dipole_ratio
#        print("cavity_calc : 20 ★★★★★★ args \n", args)
        self.layers_A = args[0]
        self.layers_B = args[1]

    def system_matrix(self, sp, *layers):
#        print("args \n", args)
#        print("type of args \n", type(args))
#        print("args[0] \n", args[0])
#        print("type of args[0] \n", type(args[0]))
        result = layers[0].matrix(sp)
        for i in range(1, len(layers)):
            result = result @ layers[i].matrix(sp)
        return np.einsum('kij->ijk', result) # ???????????? Interface랑 Layer가 101,2,2인데 왜 einsum 하지? 차원나중에 확인 해서 없엘수 있음 없에자.

    def rt_A(self, sp):
        system_matrix = self.system_matrix(sp, *self.layers_A)

        m11 = system_matrix[0, 0]
        m12 = system_matrix[0, 1]
        m21 = system_matrix[1, 0]
        m22 = system_matrix[1, 1]
        r_A = (-1) * m12 / m11
        t_A = (m11 * m12 - m12 * m21) / m11        
        return r_A.reshape(-1, 1), t_A.reshape(-1, 1)

    def rt_B(self, sp):
        system_matrix = self.system_matrix(sp, *self.layers_B)

        m11 = system_matrix[0, 0]
        m12 = system_matrix[0, 1]
        m21 = system_matrix[1, 0]
        m22 = system_matrix[1, 1]
        r_B = m21 / m11
        t_B = 1/ m11
        return r_B.reshape(-1, 1), t_B.reshape(-1, 1)
        
    def R_A(self, sp):
        r_A, _  = self.rt_A(sp)
        return np.square(np.absolute(r_A)).reshape(-1, 1)

    def R_B(self, sp):
        r_B, _  = self.rt_B(sp)
        return np.square(np.absolute(r_B)).reshape(-1, 1)

    def T_B(self, sp):
        _ , t_B  = self.rt_B(sp)
        return np.square(np.absolute(t_B)).reshape(-1, 1)
    
    def pi_A(self, sp):
        r_A, _  = self.rt_A(sp)
        r_A_real = np.real(r_A)        
        r_A_imag = np.imag(r_A)        
#        result = np.arctan(r_A_imag/r_A_real)
        result = np.real(np.log(r_A/np.sqrt(np.square(r_A_real)+np.square(r_A_imag)))/(1j))

        return result
    
    def pi_B(self, sp):
        r_B, _  = self.rt_B(sp)
        r_B_real = np.real(r_B)
        r_B_imag = np.imag(r_B)

        result = np.real(np.log(r_B/np.sqrt(np.square(r_B_real)+np.square(r_B_imag)))/(1j))
#        result = np.arctan(r_B_imag/r_B_real)

        return result
    
    def pi_TB_A(self, sp):
        pi_A  = self.pi_A(sp)        
# 190925 수정원본        
#        return 2 * k(self.nk_e, self.theta) * self.z_ex - self.pi_A(sp)
        return 2 * k(self.nk_e, self.theta) * self.z_ex - self.pi_A(sp)

    def pi_FP(self, sp):
        pi_A  = self.pi_A(sp)
        pi_B  = self.pi_B(sp)
# 190925 수정원본        
#        return 2 * k(self.nk_e, self.theta) * self.Z - self.pi_A(sp) - self.pi_B(sp)
        return 2 * k(self.nk_e, self.theta) * (self.Z + self.z_ex) - self.pi_A(sp) - self.pi_B(sp)


    # I_p 부분의 일부분들이 p가 아닌 그냥 s로 되어있음. 위에 시스템 메트릭스부터 쭉 수정해야하는데 일단 임시로 현버전으로 진행
    def intensity(self):
        I_s = self.pl * \
              self.T_B('s') * (1 + self.R_A('s') + 2 * np.sqrt(self.R_A('s')) * np.cos(self.pi_TB_A('s'))) / \
              (1 + self.R_A('s') * self.R_B('s') - 2 * np.sqrt(self.R_A('s')) * np.sqrt(self.R_B('s'))* np.cos(self.pi_FP('s'))) * \
              3/(16 * np.pi) * self.horizontal_dipole_ratio
        I_p = self.pl * \
              self.T_B('p') * (1 + self.R_A('p') + 2 * np.sqrt(self.R_A('p')) * np.cos(self.pi_TB_A('p'))) / \
              (1 + self.R_A('p') * self.R_B('p') - 2 * np.sqrt(self.R_A('p')) * np.sqrt(self.R_B('p'))* np.cos(self.pi_FP('p'))) * \
              (self.horizontal_dipole_ratio * 3/(16 * np.pi) * np.square(np.cos(self.theta)) + (1 - self.horizontal_dipole_ratio) * 3 /(8 * np.pi) * np.square(np.sin(self.theta)))
        return I_s + I_p

if __name__ == '__main__':    
    # n, k값의 경로와 PL의 경로
    ito_path = "../nk/ITO_nk.csv"
    al_path = "../nk/Al_nk.csv"
    npd_path = "../nk/NPD_nk.csv"
    air_path = "../nk/Air_nk.csv"
    Glass_path = "../nk/Glass_nk.csv"
    
    YG_PL_path = "../nk/YG_PL.txt"
    B_PL_path = "../nk/B_PL.txt"

    al = get_nk(al_path)
    npd = get_nk(npd_path)
    ito = get_nk(ito_path)
    glass = get_nk(Glass_path)
    air = get_nk(air_path)    
    
    YG_PL = get_pl(YG_PL_path)
    B_PL = get_pl(B_PL_path)
    
    theta_ex = 0
    m1 = Interface_Matrix(al, npd, theta_ex)    #1
    m2 = Layer_Matrix(npd, 203, theta_ex)        #2
    m3 = Interface_Matrix(npd, npd, theta_ex)  #3  
    m4 = Layer_Matrix(npd, 212, theta_ex)         #4
    m5 = Interface_Matrix(npd, ito, theta_ex)
    m6 = Layer_Matrix(ito, 120, theta_ex)
    m7 = Interface_Matrix(ito, glass, theta_ex)
    
    layer_A = [m1, m2, m3]
    layer_B = [m3, m4, m5, m6, m7]
    
    layers = [layer_A, layer_B]
    
#    intensity_1 = Intensity(YG_PL, npd, 10, 10, 0, 0.67, *layers).intensity()
    intensity_2 = Intensity(YG_PL, npd, 203, 212, 0, 0.67, *layers).intensity()
    
    
    #그래프 그려보기
    YG_EL_path = "../nk/YG_EL.csv"
    YG_EL_temp = pd.read_csv(YG_EL_path, sep=',', header=None)
    YG_EL = YG_EL_temp.values[1:, 1].astype(np.float64).reshape(-1,1)
    
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    x = np.arange(380, 784, 4).reshape(-1,1)
#    y1 = intensity_1
    y2 = intensity_2
#    ax.plot(x, y1/np.max(y1), label='intensity_1')
    ax.plot(x, y2/np.max(y2), label='intensity_2')
    
    ax.plot(x, YG_PL/np.max(YG_PL), label='YG_PL')
    ax.plot(x, YG_EL/np.max(YG_EL), label='YG_EL')
#    
    ax.set_title("Cavity Simulation(Labview vs Python)")
    ax.set_xlabel('Wavelength')
    ax.set_ylabel('Intensity')
    ax.legend()
    
    plt.show()
    
