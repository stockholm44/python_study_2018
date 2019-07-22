# Cavity Simulator ver2
#추가로 해야할 것
#1. Layer추가 및 두께 한버의 인풋으로 자동화. -> 현재는 일일히 Interface와 Layer Matrix를 변수를 추가해줘야함
#2. 각도에 따른 Simulator 가능
# 1) 외광 theta_ex와 theta_e와는 Snell의 법칙으로 관련되어있으므로 간단하게 수정.(theta_e를 theta_ex에 종속하게 하자)
# 2) polarization 비율과 s, p관련된 함수들을 완성시키자.
#3. Phase Shift 계산 넣기.(복소평면을 통한 계산)
import numpy as np
import pandas as pd
import os
'''
Input Data
1.Layer 별 n, k
2.Layer 순서
3.Layer 두께 정보 → From the Emission Point(EML도 나뉜다. Z(EML두께)와 z_ex(반사쪽방향의 EML 두께)로)
4.PL of the Emission Layer
5.Amplitude Factor
'''

# n, k값의 경로와 PL의 경로
ito_path = "../nk/ITO_nk.csv"
al_path = "../nk/Al_nk.csv"
npd_path = "../nk/NPD_nk.csv"
air_path = "../nk/Air_nk.csv"
YG_PL_path = "../nk/YG_PL.txt"



# 입력된 n, k 경로로 n, k를 ndarry 객체로 가져오는 class Get_nk
class Get_nk:
    def __init__(self, path):
        self.path = path
        TEMP_nk = pd.read_csv(path, sep=',', header=None)
        self.nk = TEMP_nk.values[1:, 1:].astype(np.float64)
    def get_n(self):
        return self.nk[:, 0].reshape(-1, 1)
    
    def get_k(self):
        return self.nk[:, 1].reshape(-1, 1)

    def get_nk(self):
        return self.nk
    
# layer 두께정보와 순서를 정하는 dict 변수
layer_info = []

# PL data 불러들어오기.
class Get_PL:
    def __init__(self, path):
        self.path = path
        TEMP_PL = pd.read_csv(path, sep='\t', header=None)
        self.PL = TEMP_PL.values
        
    def get_PL(self):
        return self.PL[:, 1].reshape(-1, 1)



    
# 1. reflection coeff/trasmission coeff
class RT1:
    def __init__(self, w, w_plus_1, theta1=0, theta2=0):
        self.n1 = w.get_nk()[:, 0]
        self.k1 = w.get_nk()[:, 1]
        self.n2 = w_plus_1.get_nk()[:, 0]
        self.k2 = w_plus_1.get_nk()[:, 1]
        self.theta1 = theta1
        self.theta2 = theta2
        self.nk1_complex = self.n1 + self.k1 * (1j) * (-1)
        self.nk2_complex = self.n2 + self.k2 * (1j) * (-1)
        
#        self.nk1_complex = self.nk1_complex.reshape(-1, 1)
#        self.nk2_complex = self.nk2_complex.reshape(-1, 1)
        
    def r_s(self):
        return ((self.nk1_complex)*np.cos(self.theta1) - (self.nk2_complex)*np.cos(self.theta2)) /  ((self.nk1_complex)*np.cos(self.theta1) + (self.nk2_complex)*np.cos(self.theta2))
#        return ((self.nk1_complex) - (self.nk2_complex)) 

    def r_p(self):
        return ((self.nk1_complex)*np.cos(self.theta2) - (self.nk2_complex)*np.cos(self.theta1)) /  ((self.nk1_complex)*np.cos(self.theta2) + (self.nk2_complex)*np.cos(self.theta1))
    
    def t_s(self):
        return (2 * (self.nk1_complex)*np.cos(self.theta1))/((self.nk1_complex)*np.cos(self.theta1) + (self.nk2_complex)*np.cos(self.theta2))

    def t_p(self):
        return (2 * (self.nk1_complex)*np.cos(self.theta1))/((self.nk1_complex)*np.cos(self.theta2) + (self.nk2_complex)*np.cos(self.theta1))

    
# 2. Interface Matrix 
class Interface_Matrix:
    def __init__(self, layer):
        self.r_s = layer.r_s()
        self.r_p = layer.r_p()
        self.t_s = layer.t_s()
        self.t_p = layer.t_p()
        
    def matrix_s(self):
        result = np.empty((101,2,2), dtype=complex)
        for i in range(101):
            result[i] = 1 / self.t_s[i] * np.array([[1, self.r_s[i]], [self.r_s[i], 1]])
        return result
#        return 1 / self.t_s * np.array([[np.ones((101,1)), self.r_s], [self.r_s, np.ones((101,1))]])
        return result
    def matrix_p(self):
        result = np.empty((101,2,2), dtype=complex)
        for i in range(101):
            result[i] = 1 / self.t_p[i] * np.array([[1, self.r_p[i]], [self.r_p[i], 1]])
        return result
    
# 3. Layer Matrix
class Layer_Matrix:
    def __init__(self, layer_nk, d, theta = 0):
        self.n = layer_nk.get_nk()[:, 0]
        self.k = layer_nk.get_nk()[:, 1]
        self.d = d
#        self.nk_complex = (self.n + self.k * (1j) * (-1)).reshape(-1, 1)
        self.nk_complex = self.n + self.k * (1j) * (-1)
        self.theta = theta
            
    def matrix_s(self):
        result = np.empty((101,2,2), dtype=complex)
        for i in range(101):
            k = 2 * np.pi / (380 + 4 * i) * np.asscalar(self.nk_complex[i]) * np.cos(self.theta)
            result[i] = np.array([[np.exp((-1j) * k * self.d), 0], [0, np.exp((-1j) * k * self.d)]])
#            result[i] = np.array([[1, 0], [0, 1]])
        return result 

    def matrix_p(self):
        result = np.empty((101,2,2), dtype=complex)
        for i in range(101):
            k = 2 * np.pi / (380 + 4 * i) * np.asscalar(self.nk_complex[i]) * np.cos(self.theta)
            result[i] = np.array([[np.exp((-1j) * k * self.d), 0], [0, np.exp((-1j) * k * self.d)]])
#            result[i] = np.array([[1, 0], [0, 1]])
        return result 
    
# 4. System Matrix
# A,B로 나눌필요는 없고 s,p로는 나눠야함. 
# 각 레이어별 Interface M을 s, p로 System Matrix 클래스 내에서 불러오자.
class System_Matrix:
    def __init__(self, *args):
        self.layers = args
    
    def system_matrix_s(self):
        result = self.layers[0].matrix_s()  
        for i in range(1, len(self.layers)):
            result = result @ self.layers[i].matrix_s()
        return result

    def system_matrix_p(self):
            result = self.layers[0].matrix_p()
            for i in range(1, len(self.layers)):
                result = result @ self.layers[i].matrix_p()
            return result

    
# 5. Back_Reflection/Front_Reflection/Front_Transmission Coeff
class RT2:
    def __init__(self, system_matrix_A, system_matrix_B):
        self.system_matrix_A = system_matrix_A
        self.system_matrix_B = system_matrix_B
        
        
    def rt_A_s(self):
        r_A_s = np.empty((101), dtype=complex)
        t_A_s = np.empty((101), dtype=complex)
        for i in range(101):
            m11 = self.system_matrix_A.system_matrix_s()[i][0, 0]
            m12 = self.system_matrix_A.system_matrix_s()[i][0, 1]
            m21 = self.system_matrix_A.system_matrix_s()[i][1, 0]
            m22 = self.system_matrix_A.system_matrix_s()[i][1, 1]
            r_A_s[i] = (-1) * m12 / m11
            t_A_s[i] = (m11 * m12 - m12 * m21) / m11
        return r_A_s.reshape(-1, 1), t_A_s.reshape(-1, 1)

    def rt_A_p(self):
        return self.rt_A_s()

    def rt_B_s(self):
        r_B_s = np.empty((101), dtype=complex)
        t_B_s = np.empty((101), dtype=complex)
        for i in range(101):
            m11 = self.system_matrix_B.system_matrix_s()[i][0, 0]
            m12 = self.system_matrix_B.system_matrix_s()[i][0, 1]
            m21 = self.system_matrix_B.system_matrix_s()[i][1, 0]
            m22 = self.system_matrix_B.system_matrix_s()[i][1, 1]
            r_B_s[i] = m21 / m11
            t_B_s[i] = 1 / m11
        return r_B_s.reshape(-1, 1), t_B_s.reshape(-1, 1)

    def rt_B_p(self):
        return self.rt_B_s()


# 6. Power Reflectance/Transmittance 
class Power_RT:
    def __init__(self, rt2):
        self.rt2 = rt2
        self.r_A_s, self.t_A_s = self.rt2.rt_A_s()
        self.r_B_s, self.t_B_s = self.rt2.rt_B_s()
        
    def R_A_s(self):
        return np.square(np.absolute(self.r_A_s)).reshape(-1, 1)
    def R_B_s(self):
        return np.square(np.absolute(self.r_B_s)).reshape(-1, 1)
    def T_A_s(self):
        pass
    def T_B_s(self, n_e, n_ex, theta_n_e=0, theta_n_ex=0):
        self.n_e = n_e.get_n()
        self.n_ex = n_ex.get_n()
        self.t_B_s = self.t_B_s.reshape(-1, 1)
        return (self.n_ex * np.cos(theta_n_ex)) / (self.n_e * np.cos(theta_n_e)) * np.square(np.absolute(self.t_B_s))


        
# 7. Z-component of the complex propagation constant
class K:
    def __init__(self, w, theta=0):
        self.n = w.get_nk()[:, 0]
        self.k = w.get_nk()[:, 1]
        self.theta = theta
        self.nk_complex = self.n + self.k * (1j) * (-1)
        self.nk_complex = self.nk_complex.reshape(-1, 1)
        
    def k_layer(self):
#        for i in range(101):
#            k_w = 2 * np.pi / (380 + 4 * i) * np.asscalar(self.nk_complex[i]) * np.cos(self.theta)
        print(np.arange(380, 784, 4).reshape(-1,1).shape, self.nk_complex.shape)
        return 2 * np.pi / np.array(np.arange(380,784,4).reshape(-1,1)) * self.nk_complex * np.cos(self.theta)
                #  요기 해결해야한다. 
    

    
# 8. Phase Change1d
class Phase_Change1:
    def __init__(self, rt2):
        self.r_A_s, _ = rt2.rt_A_s()
        print("self.r_A_s",self.r_A_s)
        print("self.r_A_s.shape",self.r_A_s.shape)
        
        self.r_B_s, _ = rt2.rt_B_s()
        print("self.r_B_s",self.r_B_s)
        print("self.r_B_s.shape",self.r_B_s.shape)
    
    def pi_A(self):
        r_A_S_real = np.real(self.r_A_s)
        r_A_S_imag = np.imag(self.r_A_s)
        result = np.real(np.log(self.r_A_s/np.sqrt(np.square(r_A_S_real)+np.square(r_A_S_imag)))/(1j))
        print('pi_A', result)
        return result
    
    def pi_B(self):
        r_B_S_real = np.real(self.r_B_s)
        r_B_S_imag = np.imag(self.r_B_s)
        result = np.real(np.log(self.r_B_s/np.sqrt(np.square(r_B_S_real)+np.square(r_B_S_imag)))/(1j))
        print('pi_B', result)
        return result
    
# 9. Phase Change2
class Phase_Change2:
    def __init__(self, k_layer, z_ex, Z, phase_change):
        self.k = np.absolute(k_layer.k_layer())
        self.z_ex = z_ex
        self.Z = Z
        self.pi_A = phase_change.pi_A()
        self.pi_B = phase_change.pi_B()
        
    def pi_TB_A_s(self):
#        print('227 : ',2 * self.k * self.z_ex - self.pi_A)
        return 2 * self.k * self.z_ex - self.pi_A

    def pi_TB_A_p(self):
#        print('231 : ',2 * self.k * self.z_ex - self.pi_A)
        return 2 * self.k * self.z_ex - self.pi_A

    def pi_FP_s(self):
#        print('235 : ',2 * self.k * self.Z - self.pi_A - self.pi_B)
        return 2 * self.k * self.Z - self.pi_A - self.pi_B

    def pi_FP_p(self):
#        print('239 : ',2 * self.k * self.Z - self.pi_A - self.pi_B)
        return 2 * self.k * self.Z - self.pi_A - self.pi_B
    
# 10. Total Output spectral radient intensity
class Intensity:
    def __init__(self, pl, power_RT, phase_change2, n_ex, n_e, theta_ex=0, theta_e=0, horizontal_dipole_ratio=0.67):
        self.n_ex = n_ex.get_n()
        
        self.n_e = n_ex.get_n()
        self.I_R_EML = pl.get_PL()
        self.R_A_s = power_RT.R_A_s()
        self.R_B_s = power_RT.R_B_s()
#        self.T_A_s = power_RT.T_A_s()
        self.T_B_s = power_RT.T_B_s(n_e, n_ex)
        self.pi_TB_A_s = phase_change2.pi_TB_A_s()
        self.pi_TB_A_p = phase_change2.pi_TB_A_p()
        self.pi_FP_s = phase_change2.pi_FP_s()
        self.pi_FP_p = phase_change2.pi_FP_p()
        self.theta_ex = theta_ex
        self.theta_e = theta_e
        self.horizontal_dipole_ratio = horizontal_dipole_ratio
       
        
    # I_p 부분의 일부분들이 p가 아닌 그냥 s로 되어있음. 위에 시스템 메트릭스부터 쭉 수정해야하는데 일단 임시로 현버전으로 진행
    def intensity(self):
        I_s = self.I_R_EML * self.T_B_s * (1 + self.R_A_s + 2 * np.sqrt(self.R_A_s) * np.cos(self.pi_TB_A_s))/(1 + self.R_A_s * self.R_B_s - 2 * np.sqrt(self.R_A_s) * np.sqrt(self.R_B_s)* np.cos(self.pi_FP_s)) * np.square(self.n_ex) * np.cos(self.theta_ex)/(np.square(self.n_e)* np.cos(self.theta_e))* 3/(16 * np.pi) * self.horizontal_dipole_ratio
        I_p = self.I_R_EML * self.T_B_s * (1 + self.R_A_s + 2 * np.sqrt(self.R_A_s) * np.cos(self.pi_TB_A_p))/(1 + self.R_A_s * self.R_B_s - 2 * np.sqrt(self.R_A_s) * np.sqrt(self.R_B_s)* np.cos(self.pi_FP_p)) * np.square(self.n_ex) * np.cos(self.theta_ex)/(np.square(self.n_e)* np.cos(self.theta_e))* (self.horizontal_dipole_ratio * 3/(16 * np.pi) * np.square(np.cos(self.theta_e)) + (1 - self.horizontal_dipole_ratio) * 3 /(8 * np.pi) * np.square(np.sin(self.theta_e)))
        return I_s + I_p
# 11. Incoherent Multiple Reflection
class RT_to_Air:
    pass
        
# 12. Cavity Enhencement Factor(to Air)
class G:
    pass

if __name__ == '__main__':
    # 시간 기록
    from datetime import datetime
    start_time = datetime.now()
    
    # n, k값의 경로와 PL의 경로
    ito_path = "../nk/ITO_nk.csv"
    al_path = "../nk/Al_nk.csv"
    npd_path = "../nk/NPD_nk.csv"
    air_path = "../nk/Air_nk.csv"
    YG_PL_path = "../nk/YG_PL.txt"

# =============================================================================
#      System Matrix 직접 구현.\n,
#      - System_Matrix_A -> Al/NPD(218nm-50nm = 168nm)
#      - System_Matrix_B -> NPD(207nm-50nm = 157nm)/ITO(104nm)
#      - EML은 100nm
#        - Layer_Matrix_EML_L -> EML(50nm)
#        - Layer_Matrix_EML_R -> EML(50nm)
# =============================================================================
    ito = Get_nk(ito_path)
    al = Get_nk(ito_path)
    npd = Get_nk(ito_path)
    air = Get_nk(npd_path)
    
    YG_PL = Get_PL(YG_PL_path)
    
    al_npd = RT1(al, npd, 0, 0)
    npd_npd = RT1(npd, npd, 0, 0)
    npd_ito = RT1(npd, ito, 0, 0)

    # ito_air는 나중에 Cavity Enhencement Factor 할때나 활용
    #1, #2  #3은 System_A에 포함.#3, #4, #5는 System_B에 포함
    # 궁금한점은 #4 즉 EML두께에 대한 Layer텀은 어디에 포함? 위상차에 포함되서 시스템에서 필요없는건가?
    # 그리고 #6은 나중에 Ts-a에서 알아서 반영되는건가?
    al_npd_IM = Interface_Matrix(al_npd)    #1
    npd1_LM = Layer_Matrix(npd, 168)        #2
    npd_npd_IM = Interface_Matrix(npd_npd)  #3  
    npd2_LM = Layer_Matrix(npd, 157)         #4
    npd_ito_IM = Interface_Matrix(npd_ito)  #5
    ito_LM = Layer_Matrix(ito, 100)      #6
    
    sm_A, sm_B = System_Matrix(al_npd_IM, npd1_LM, npd_npd_IM), System_Matrix(npd_npd_IM, npd2_LM, npd_ito_IM)
    rt2 = RT2(sm_A, sm_B)

    power_rt = Power_RT(rt2)
    npd_k = K(npd)
    phase1 = Phase_Change1(rt2)
    phase2 = Phase_Change2(npd_k, 100, 100, phase1)
    final_intensity = Intensity(YG_PL, power_rt, phase2, npd, ito)
    print(final_intensity.intensity())
    
    
    # Graph로 확인
    import matplotlib.pyplot as plt
    
    YG_EL_path = "../nk/YG_EL.csv"
    YG_EL_temp = pd.read_csv(YG_EL_path, sep=',', header=None)
    YG_EL = YG_EL_temp.values[1:, 1].astype(np.float64).reshape(-1,1)
    print(YG_EL)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    x = np.arange(380, 784, 4).reshape(-1,1)
    y = final_intensity.intensity()

    ax.plot(x, y/np.max(y))
    ax.plot(x, YG_EL/np.max(YG_EL))
    
    plt.show()
    
    # 실행 시간 계산
    end_time = datetime.now()
    
    print("Executing Time : ", end_time-start_time)
