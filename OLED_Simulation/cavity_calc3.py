'''재호C꺼 참고해서 코드 변경. 변경부분에는 ★표 씨게 박자.'''

# System Matrix, R, T 계산 후 최종 Intensity 계산.
import numpy as np
import pandas as pd
from cavity_input import *
import matplotlib.pyplot as plt

    
# 10. Total Output spectral radient intensity
'''
system_matrix, rt, R, T, pi_A, pi_B 까지는 layers만이면 충분
그러나 각 layer 별 n, d가 필요한것이 아래의 3가지이다.
1. phase_change_oled
2. pi_TB, pi_FP
3. solid angle -> 끝단의 n과 EML의 n만 필요

그러나 

'''
class Intensity:
    def __init__(self, pl, nk_e, nk_g, z_ex, Z, theta_ex, horizontal_dipole_ratio, phase_TB, phase_total, *args):
        self.pl = pl
        self.nk_e = nk_e
        self.nk_g = nk_g
        self.z_ex = z_ex
        self.Z = Z
        self.theta_ex = theta_ex
        self.theta = calc_theta(get_n(self.nk_e), theta_ex)
        self.horizontal_dipole_ratio = horizontal_dipole_ratio
        self.layers_A = args[0]
        self.layers_B = args[1]
        
        self.phase_TB = phase_TB
        self.phase_total = phase_total
        

    def system_matrix(self, sp, *layers):
        result = layers[0].matrix(sp)
        for i in range(1, len(layers)):
            result = result @ layers[i].matrix(sp)
        return np.einsum('kij->ijk', result) 

    def rt_A(self, sp):
        system_matrix = self.system_matrix(sp, *self.layers_A)

        m11 = system_matrix[0, 0]
        m12 = system_matrix[0, 1]
        m21 = system_matrix[1, 0]
        m22 = system_matrix[1, 1]
        r_A = (-1) * m12 / m11
        t_A = (m11 * m22 - m12 * m21) / m11        

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
        return np.square(abs(r_A)).reshape(-1, 1)

    def R_B(self, sp):
        r_B, _  = self.rt_B(sp)
        return np.square(abs(r_B)).reshape(-1, 1)

    def T_B(self, sp):
        _ , t_B  = self.rt_B(sp)
        a = (get_n(self.nk_g)/get_n(self.nk_e)).reshape(-1,1)
        b = (np.square(abs(t_B))).reshape(-1,1)
        return a * b
    
    def pi_A(self, sp):
        r_A, _  = self.rt_A(sp)
        r_A_real = np.real(r_A)        
        r_A_imag = np.imag(r_A)        
        result = np.real(np.log(r_A/np.sqrt(np.square(r_A_real)+np.square(r_A_imag)))/(1j))
        return result

    def pi_B(self, sp):
        r_B, _  = self.rt_B(sp)
        r_B_real = np.real(r_B)
        r_B_imag = np.imag(r_B)
        result = np.real(np.log(r_B/np.sqrt(np.square(r_B_real)+np.square(r_B_imag)))/(1j))
        return result
    
    def phase_change_oled(self, sp, d):
        phase_change_oled = 2 * np.pi / wavelength() * 2 * get_n(self.nk_e) * (self.z_ex + self.Z) * np.cos(self.theta)
        return phase_change_oled
    
    def pi_FP(self, sp):
        pi_A  = self.pi_A(sp)
        pi_B  = self.pi_B(sp)
        if type(self.phase_total) == type(np.array):
            result = -pi_A - pi_B + 2 * np.pi / wavelength() * 2 * self.phase_total # * self.theta 각도텀 적용 여부
        else:
            phase_change_oled = self.phase_change_oled(sp, self.z_ex + self.Z)
            result = -pi_A - pi_B + phase_change_oled
        return result

    def pi_TB(self, sp):
        pi_A  = self.pi_A(sp)
        if type(self.phase_TB) == type(np.array):
            result = pi_A + 2 * np.pi / wavelength() * 2 * self.phase_TB
        
        else:
            phase_change_oled = self.phase_change_oled(sp, self.z_ex)
            result = pi_A + phase_change_oled
        return result

    
    # ★★★★df 를 받았을때 
        # 1. d계산을 위해 organic/coherent, cathode등 layer이름세분화하고 
        # 2. df.unique()찍어서 이상한거 있으면 에러내게 해야겠다.
        
    def f_TB(self, sp):
        result = 1 + self.R_A(sp) + 2 * np.sqrt(self.R_A(sp)) * np.cos(self.pi_TB(sp))
        return result

    def f_FP(self, sp):
        result = self.T_B(sp)/(np.square(1 - np.sqrt(self.R_A(sp) * self.R_B(sp))) + 4 * np.sqrt(self.R_A(sp) * self.R_B(sp)) * np.square(np.sin(self.pi_FP(sp)/2)))
        return result

    def solid_angle(self):
        return (1 * np.cos(np.radians(self.theta_ex)))/(np.square(get_n(self.nk_e)) * np.cos(self.theta))
        
    def solid_angle_make_df(self, va):
        temp_df = pd.DataFrame()
        temp_df['theta'] = self.theta
        temp_df['np.cos(theta)'] = np.cos(self.theta)
        temp_df['n_e'] = get_n(self.nk_e)
        temp_df['n_e^2'] = np.square(get_n(self.nk_e))
        temp_df['solid_angle'] = (1 * np.cos(self.theta_ex))/(np.square(get_n(self.nk_e)) * np.cos(self.theta))
        temp_df.to_csv('solid_angle' + str(va) + '.csv')

        
    
    # I_p 부분의 일부분들이 p가 아닌 그냥 s로 되어있음. 위에 시스템 메트릭스부터 쭉 수정해야하는데 일단 임시로 현버전으로 진행
    def intensity(self):        
        I_s = self.pl * self.f_TB('s') * self.f_FP('s') * self.solid_angle()
        return I_s

if __name__ == '__main__':    
    # n, k값의 경로와 PL의 경로
    ito_path = "../nk/ITO_nk.csv"
    al_path = "../nk/Al_nk.csv"
    npd_path = "../nk/NPD_nk.csv"
    air_path = "../nk/Air_nk.csv"
    Glass_path = "../nk/Glass_nk.csv"
    
    YG_PL_path = "../nk/YG_PL.csv"
    B_PL_path = "../nk/B_PL.csv"
    
    al = get_nk(al_path)
    npd = get_nk(npd_path)
    ito = get_nk(ito_path)
    glass = get_nk(Glass_path)
    air = get_nk(air_path)    
    
    YG_PL = get_pl(YG_PL_path)
    B_PL = get_pl(B_PL_path)
        
    theta_ex = 0
    oled_thick = 300
    pl_position = 30
    
    fig = plt.figure(figsize=(10,10))
    ax1 = fig.add_subplot(411)
    ax2 = fig.add_subplot(412)
    ax3 = fig.add_subplot(413)
    ax4 = fig.add_subplot(414)
    x = np.arange(380, 784, 4).reshape(-1,1)
    
    for va in [0,15,30,45,60]:
        m1 = Interface_Matrix(al, npd, theta_ex)    #1
        m2 = Layer_Matrix(npd, pl_position, theta_ex)        #2
    
        m3 = Interface_Matrix(npd, npd, theta_ex)  #3  
    
        m4 = Layer_Matrix(npd, oled_thick-pl_position, theta_ex)  #4
        m5 = Interface_Matrix(npd, ito, theta_ex)
        m6 = Layer_Matrix(ito, 120, theta_ex)
        m7 = Interface_Matrix(ito, glass, theta_ex)
    
        layer_A = [m1, m2, m3]
        layer_B = [m3, m4, m5, m6, m7]
    
        layers = [layer_A, layer_B]
    
        int_temp = Intensity(YG_PL, npd, glass, pl_position, oled_thick-pl_position, va, 0.67, None, None, *layers)
        
        y1 = int_temp.f_TB('s')
        y2 = int_temp.f_FP('s')
        
        y3 = int_temp.R_A('s')
        y4 = int_temp.R_B('s')
        y5 = int_temp.T_B('s')
        
        y6 = int_temp.solid_angle()
        y7 = int_temp.intensity()
        
        ax1.plot(x, y1, label=str(va)+'f_TB')
        ax1.plot(x, y2, label=str(va)+'f_FP')
        
        ax2.plot(x, y3, label=str(va)+'R_A')
        ax2.plot(x, y4, label=str(va)+'R_B')
        ax2.plot(x, y5, label=str(va)+'T_B')
        
        ax3.plot(x, y6, label=str(va)+'SA')
        ax4.plot(x, y7, label=str(va)+'Int')
        
#        int_temp.solid_angle_make_df(va)
        
    ax1.legend()
    ax2.legend()
    ax3.legend()
    ax4.legend()
    plt.show()        
    
    
#    # 1. PL 위치
#    # n, k값의 경로와 PL의 경로
#    df = pd.DataFrame()
#    theta_ex = 0
#    
#    YG_EL_path = "../nk/YG_EL.csv"
#    YG_EL_temp = pd.read_csv(YG_EL_path, sep=',', header=None)
#    YG_EL = YG_EL_temp.values[1:, 1].astype(np.float64).reshape(-1,1)
#    count = 0
#    for oled_thick in range(200, 500, 5):
#        for pl_position in range(20, oled_thick+20, 5 ):
#            theta_ex = 0
#            m1 = Interface_Matrix(al, npd, theta_ex)    #1
#            m2 = Layer_Matrix(npd, pl_position, theta_ex)        #2
#    
#            m3 = Interface_Matrix(npd, npd, theta_ex)  #3  
#    
#            m4 = Layer_Matrix(npd, oled_thick-pl_position, theta_ex)  #4
#            m5 = Interface_Matrix(npd, ito, theta_ex)
#            m6 = Layer_Matrix(ito, 120, theta_ex)
#            m7 = Interface_Matrix(ito, glass, theta_ex)
#    
#            layer_A = [m1, m2, m3]
#            layer_B = [m3, m4, m5, m6, m7]
#    
#            layers = [layer_A, layer_B]
#    
#            int_temp = Intensity(YG_PL, npd, glass, pl_position, oled_thick-pl_position, 0, 0.67, 0, 0, *layers)
#    
#            df.loc[count, "OLED_Thick"] = oled_thick
#            df.loc[count, "PL_Position"] = pl_position
#            df.loc[count, "dSpectrum"] = np.sum(np.abs(YG_EL/max(YG_EL) - int_temp.intensity()/max(int_temp.intensity())))
#    
#            count = count + 1
#    
#    # Heat-map을 위한 pivot_table
#    df_pivot = df.pivot_table(values='dSpectrum', index='OLED_Thick', columns='PL_Position', aggfunc='mean')
#
#    import seaborn as sns
#    fig = plt.figure(figsize=(10,7))
#    ax = fig.add_subplot(111)
#    sns.heatmap(df_pivot, cmap="YlGnBu", ax=ax)
#    plt.show()
#    
#    df.loc[df['dSpectrum']==min(df['dSpectrum']), ['OLED_Thick','PL_Position','dSpectrum']]
#    
#    # 최적지점 설정
#    pl_position = 20.0
#    oled_thick = 335
#    
#    m1 = Interface_Matrix(al, npd, theta_ex)    #1
#    m2 = Layer_Matrix(npd, pl_position, theta_ex)        #2
#    
#    m3 = Interface_Matrix(npd, npd, theta_ex)  #3  
#    
#    m4 = Layer_Matrix(npd, oled_thick-pl_position, theta_ex)  #4
#    m5 = Interface_Matrix(npd, ito, theta_ex)
#    m6 = Layer_Matrix(ito, 120, theta_ex)
#    m7 = Interface_Matrix(ito, glass, theta_ex)
#    
#    layer_A = [m1, m2, m3]
#    layer_B = [m3, m4, m5, m6, m7]
#    
#    layers = [layer_A, layer_B]
#    
#    int_temp = Intensity(YG_PL, npd, glass, pl_position, oled_thick-pl_position, 0, 0.67, *layers)
#    
#    # 그래프
#    YG_EL_path = "../nk/YG_EL.csv"
#    YG_EL_temp = pd.read_csv(YG_EL_path, sep=',', header=None)
#    YG_EL = YG_EL_temp.values[1:, 1].astype(np.float64).reshape(-1,1)
#    
#    fig = plt.figure(figsize=(10,7))
#    ax = fig.add_subplot(111)
#    x = np.array(range(380, 784, 4)).reshape(-1,1)
#    y1 = int_temp.intensity()
#    
#    
#    ax.plot(x, y1/max(y1), label='Sim')
#    ax.plot(x, YG_EL/max(YG_EL), label='YG_EL', ls='--')
#    plt.show()
#    
#    print(np.sum(np.abs(YG_EL/max(YG_EL) - int_temp.intensity()/max(int_temp.intensity()))))