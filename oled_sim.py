# Cavity Simulator ver3
# 추가로 해야할 것
# 1. Layer추가 및 두께 한버의 인풋으로 자동화. -> 현재는 일일히 Interface와 Layer Matrix를 변수를 추가해줘야함
# 2. 각도에 따른 Simulator 가능 ->하는중
# 1) 외광 theta_ex와 theta_e와는 Snell의 법칙으로 관련되어있으므로 간단하게 수정.(theta_e를 theta_ex에 종속하게 하자)
# 2) polarization 비율과 s, p관련된 함수들을 완성시키자.
# 3. Phase Shift 계산 넣기.(복소평면을 통한 계산) -> 완료
import numpy as np
import pandas as pd
import os
import scipy.interpolate as ipl
import matplotlib.pyplot as plt
import collections
from datetime import datetime


np.seterr(divide='ignore', invalid='ignore')
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
        TEMP_nk = pd.read_csv(path, sep=',') # read DataFrame
#        # nk값 내삽
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
        self.nk = TEMP_nk.values[:, 1:].astype(np.float64) # (101,2)
        
    # 4번밖에 쓰이지 않는데 써야하나.. nk_complex로 대체가능하지 않을까.?
    def get_n(self):
        return self.nk[:, 0].reshape(-1, 1) # (101, 1) 
    
    def get_nk_complex(self):
        self.n = self.nk[:, 0]
        self.k = self.nk[:, 1]
        self.nk_complex = self.n + self.k * (1j) * (-1)
        return self.nk_complex.reshape(-1, 1)
        
# layer 두께정보와 순서를 정하는 dict 변수


# PL data 불러들어오기.
class Get_PL:
    def __init__(self, path):
        self.path = path
        TEMP_PL = pd.read_csv(path, sep='\t', header=None)
        self.PL = TEMP_PL.values
        
    def get_PL(self):
        return self.PL[:, 1].reshape(-1, 1) # (101,1)

    
# 1. reflection coeff/trasmission coeff
class RT1:
    def __init__(self, w, w_plus_1, theta_ex):
        self.nk1_complex = w.get_nk_complex()
        self.nk2_complex = w_plus_1.get_nk_complex()

        self.n1 = w.get_n()
        self.n2 = w_plus_1.get_n()
        self.theta_ex = np.radians(theta_ex)
        
        self.theta1 = (np.arcsin(np.ones((101, 1)) / self.n1 * np.sin(self.theta_ex)))
        self.theta2 = (np.arcsin(np.ones((101, 1)) / self.n2 * np.sin(self.theta_ex)))

    def r_s(self):
        return ((self.nk1_complex)*np.cos(self.theta1) - (self.nk2_complex)*np.cos(self.theta2)) /  ((self.nk1_complex)*np.cos(self.theta1) + (self.nk2_complex)*np.cos(self.theta2))
    
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
        ones = np.ones((101, 1))
        result_b = 1 / self.t_s * np.array([[ones, self.r_s], [self.r_s, ones]])
        result_c = result_b.squeeze()
        result_d = np.einsum('ijk->kij',result_c)
        return result_d

    def matrix_p(self):
        ones = np.ones((101, 1))
        result_b = 1 / self.t_p * np.array([[ones, self.r_p], [self.r_p, ones]])
        result_c = result_b.squeeze()
        result_d = np.einsum('ijk->kij',result_c)
        return result_d
    
# 3. Layer Matrix
class Layer_Matrix:
    def __init__(self, layer_nk, d, theta_ex):
        self.n = layer_nk.get_n()
        self.d = d
        self.nk_complex = layer_nk.get_nk_complex()
        self.theta_ex = np.radians(theta_ex)
        self.theta = np.arcsin(np.ones((101,1))/self.n * np.sin(self.theta_ex))
########################################여기 수정할 차            
    def matrix_s(self):
        wavelength = np.arange(380, 784, 4).reshape(-1, 1)
        k = 2 * (np.pi / wavelength) * self.nk_complex * np.cos(self.theta)
        result_a = np.array([[np.exp((-1j) * k * self.d), np.zeros((101, 1))], [np.zeros((101, 1)), np.exp((1j) * k * self.d)]])
        result_b = np.einsum('ijk->kij', result_a.squeeze())
#        print("Thickness : ", self.d)
#        print("Layer Matrix : ", result_b)
        
        return result_b 

    def matrix_p(self):
        wavelength = np.arange(380, 784, 4)
        k = 2 * np.pi / wavelength * self.nk_complex * np.cos(self.theta)
        result_a = np.array([[np.exp((-1j) * k * self.d), np.zeros(101)], [np.zeros(101), np.exp((-1j) * k * self.d)]])
        result_b = np.einsum('ijk->kij', result_a)
        return result_b 
    
# 4. System Matrix
# A,B로 나눌필요는 없고 s,p로는 나눠야함. 
# 각 레이어별 Interface M을 s, p로 System Matrix 클래스 내에서 불러오자.
class System_Matrix:
    def __init__(self, *args):
#        print(args)
#        print(type(args))
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
        system_matrix = np.einsum('kij->ijk', self.system_matrix_A.system_matrix_s())
        m11 = system_matrix[0, 0]
#        print("m11_System_A", m11)
        m12 = system_matrix[0, 1]
        m21 = system_matrix[1, 0]
        m22 = system_matrix[1, 1]
        r_A_s = (-1) * m12 / m11
        t_A_s = (m11 * m12 - m12 * m21) / m11
        
        return r_A_s.reshape(-1, 1), t_A_s.reshape(-1, 1)

    def rt_A_p(self):
        return self.rt_A_s()

    def rt_B_s(self):
        system_matrix = np.einsum('kij->ijk', self.system_matrix_B.system_matrix_s())
        m11 = system_matrix[0, 0]
#        print("m11_System_B", m11)
        m12 = system_matrix[0, 1]
        m21 = system_matrix[1, 0]
        m22 = system_matrix[1, 1]
        r_B_s = m21 / m11
        t_B_s = 1 / m11
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
    def __init__(self, w, theta_ex):
        self.nk_complex = w.get_nk_complex()
        self.theta_ex = theta_ex
        
    def k_layer(self):
        return 2 * np.pi / np.array(np.arange(380,784,4).reshape(-1,1)) * self.nk_complex * np.cos(self.theta_ex)
                #  요기 해결해야한다. 
    

    
# 8. Phase Change1d
class Phase_Change1:
    def __init__(self, rt2):
        self.r_A_s, _ = rt2.rt_A_s()
        self.r_B_s, _ = rt2.rt_B_s()
    
    def pi_A(self):
        r_A_S_real = np.real(self.r_A_s)        
        r_A_S_imag = np.imag(self.r_A_s)        
        result = np.real(np.log(self.r_A_s/np.sqrt(np.square(r_A_S_real)+np.square(r_A_S_imag)))/(1j))
        return result
    
    def pi_B(self):
        r_B_S_real = np.real(self.r_B_s)
        r_B_S_imag = np.imag(self.r_B_s)
        result = np.real(np.log(self.r_B_s/np.sqrt(np.square(r_B_S_real)+np.square(r_B_S_imag)))/(1j))
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
        return 2 * self.k * self.z_ex - self.pi_A

    def pi_TB_A_p(self):
        return 2 * self.k * self.z_ex - self.pi_A

    def pi_FP_s(self):
        return 2 * self.k * self.Z - self.pi_A - self.pi_B

    def pi_FP_p(self):
        return 2 * self.k * self.Z - self.pi_A - self.pi_B
    
# 10. Total Output spectral radient intensity
class Intensity:
    def __init__(self, pl, power_RT, phase_change2, n_e, n_ex, theta_ex=0, theta_e=0, horizontal_dipole_ratio=0.67):
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
class T_SUB:
    def __init__(self, n_s, n_m, n_a):
        r_s_m = RT1(n_s, n_m, 0).r_s() # 여기도 p에 대해 나중에 따로 만들어야함.
        r_s_a = RT1(n_s, n_a, 0).r_s()
        self.R_s_a = np.square(np.absolute(r_s_a))
        self.R_s_m = np.square(np.absolute(r_s_m))
        self.T_s_a = 1 - self.R_s_a
        
    def T_sub_s(self):
        return self.T_s_a / (1 - self.R_s_a * self.R_s_m)  
    def T_sub_p(self):
        return T_sub_s()
        
# 12. Cavity Enhencement Factor(to Air)
class G:
    def __init__(self, intensity_e_ex, t_sub_e_ex):
        self.intensity = intensity_e_ex.intensity()
        self.t_sub = t_sub_e_ex.T_sub_s()
    
    def g_final(self):
        return self.intensity * self.t_sub

class Stack:
    def __init__(self, layers):
        self.d = layers
        self.layer_count = len(self.d)
        self.theta_col = [0, 15, 30, 45, 60]
    
    def one_stack_intensity_calculate(self):
        
        # layer들간의 Interface, Layer Matrix 만들기.
        # 1. 첫번째 layer는 반사막이므로 layer matrix 안만듬
        # 2. layer를 검사해서 coherent이고 layer인 애들만 만들기.
        # 3. 마지막 coherent layer에 대해서는 layer matrix 안만듬.
        
        # pl, coherent layer, incoherent layer 갯수, 위치 파악
        theta = 0
#        print(d)
        pl_col = []
        coherent_col = []
        incoherent_col = []
        for k, v in d.items():
            if v[2] == 'coherent' and v[3] == 'pl':
                pl_col.append(k) 
            elif v[2] == 'coherent' and v[3] == 'layer':
                coherent_col.append(k)
            elif v[2] == 'incoherent' or k == len(d) - 1:
                incoherent_col.append(k)
        
        print("pl_col", pl_col)
        print("coherent_col", coherent_col)
        print("incoherent_col", incoherent_col)
        
        # 각 pl 별로 system_A, system_B를 구하고 그거에 해당하는 layer들만 Matrix만들기.
        # system_A @ pl
        pl_dict = {}
        rt_2_dict ={}
        power_rt_dict = {}
        phase1_dict = {}
        phase2_dict = {}
        coherent_intensity = {}
        final_intensity = np.zeros(101).reshape(-1, 1)
        for i, pl in enumerate(pl_col):
            system_A = []
            system_B = []
            system_A_name_list = []
            system_B_name_list = []
            
            for j, k in enumerate(coherent_col):
                if k < pl - 2: # System_Matrix_A 구하기_1      # ♥♥♥♥♥♥♥ coherent_col[j]로 레이어 지정안하면 나중에 PL 늘었을때에 PL껄 건들일수 있음. coherent 안에서 +1 해야함.아래 elif 3개 다 해야함.
                    system_A.append(Interface_Matrix(RT1(d[coherent_col[j]][0], d[coherent_col[j+1]][0], theta)))
                    system_A_name_list.append('interface ' + d[coherent_col[j]][5] + d[coherent_col[j+1]][5] )
                    system_A.append(Layer_Matrix(d[coherent_col[j+1]][0], d[coherent_col[j+1]][1], theta))
                    system_A_name_list.append('layer ' + str(d[coherent_col[j+1]][1]) + d[coherent_col[j+1]][5])

                elif k == pl - 2: # System_Matrix_A 구하기_2(마지막 Layer 처리)
                    system_A.append(Interface_Matrix(RT1(d[k][0], d[k + 1][0], theta)))
                    system_A_name_list.append('interface ' + d[k][5] + d[k + 1][5] )

                elif k > pl and k < coherent_col[-2]: # System_Matrix_B 구하기
                    system_B.append(Interface_Matrix(RT1(d[k][0], d[k + 1][0], theta)))
                    system_B_name_list.append('interface ' + d[k][5] + d[k + 1][5] )
                    system_B.append(Layer_Matrix(d[coherent_col[j+1]][0], d[coherent_col[j+1]][1], theta))
                    system_B_name_list.append('layer ' + str(d[coherent_col[j+1]][1]) + d[coherent_col[j+1]][5])
                elif k == coherent_col[-2]:
                    system_B.append(Interface_Matrix(RT1(d[coherent_col[j]][0], d[coherent_col[j+1]][0], theta)))
                    system_B_name_list.append('interface ' + d[coherent_col[j]][5] + d[coherent_col[j+1]][5] )

#            print('i', i)
#            print('pl', pl)
#            print('j', j)
#            print('k', k)
            
            print("system_A_name_list", system_A_name_list)
            print("system_B_name_list", system_B_name_list)
            pl_dict[i] = System_Matrix(*system_A), System_Matrix(*system_B)
        
            rt_2_dict[i] = RT2(pl_dict[i][0], pl_dict[i][1])
            power_rt_dict[i] = Power_RT(rt_2_dict[i])
            npd_k = K(npd, theta)
            phase1_dict[i] = Phase_Change1(rt_2_dict[i])
            phase2_dict[i] = Phase_Change2(npd_k, d[pl_col[i] - 1][1], d[pl_col[i] - 1][1] + d[pl_col[i] + 1][1], phase1_dict[i])
            coherent_intensity[i] = Intensity(d[pl_col[i]][0], power_rt_dict[i], phase2_dict[i], npd, glass)
#            print(final_intensity)
#            print(coherent_intensity[i].intensity())
            final_intensity = final_intensity + coherent_intensity[i].intensity()
        print("pl_dict", pl_dict)
        print("rt_2_dict", rt_2_dict)
        print("power_rt_dict",power_rt_dict)
        return final_intensity
        # item들 다 추가하고 set 함수로 중복된 layer 지우기.
        
        
        
#        rt2 = RT2(sm_A, sm_B)
#    
#        power_rt = Power_RT(rt2)
#        npd_k = K(npd, 0)
#        phase1 = Phase_Change1(rt2)
#        phase2 = Phase_Change2(npd_k, 10, 10, phase1)
#        coherent_intensity = Intensity(YG_PL, power_rt, phase2, npd, glass)
#        t_sub_ito_glass = T_SUB(glass, ito, air)
#        final_incoherent_intensity = G(coherent_intensity, t_sub_ito_glass).g_final()    
        
            
        # nk 만들기



if __name__ == '__main__':

    # 시간 기록
    start_time = datetime.now()
    
    # n, k값의 경로와 PL의 경로
    ito_path = "../nk/ITO_nk.csv"
    al_path = "../nk/Al_nk.csv"
    npd_path = "../nk/NPD_nk.csv"
    air_path = "../nk/Air_nk.csv"
    Glass_path = "../nk/Glass_nk.csv"
    
    YG_PL_path = "../nk/YG_PL.txt"

    al = Get_nk(al_path)
    npd = Get_nk(npd_path)
    ito = Get_nk(ito_path)
    glass = Get_nk(Glass_path)
    air = Get_nk(air_path)    
    
    YG_PL = Get_PL(YG_PL_path)

    # Layer 실행        
    d = collections.OrderedDict()
    d[0] = [al, 100, 'coherent', 'layer', 0, 'al']
    d[1] = [npd, 203, 'coherent', 'layer', 0, 'npd']
    d[2] = [npd, 10, 'coherent', 'layer', 0, 'npd']
    d[3] = [YG_PL, 0, 'coherent', 'pl', 2, 'YG_PL']
    d[4] = [npd, 10, 'coherent', 'layer', 0, 'npd']
    d[5] = [npd, 412, 'coherent', 'layer', 0, 'npd']
    d[6] = [ito, 120, 'coherent', 'layer', 0, 'ito']
    d[7] = [glass, 5000, 'incoherent', 'layer', 0, 'glass']
    d[8] = [air, 0, 'incoherent', 'layer', 0, 'air']

#    d[0] = [al, 100, 'coherent', 'layer', 0]
#    d[1] = [npd, 168, 'coherent', 'layer', 0]
#    d[2] = [npd, 50, 'coherent', 'layer', 0]
#    d[3] = [YG_PL, 0, 'coherent', 'pl', 2]
#    d[4] = [npd, 50, 'coherent', 'layer', 0]
#    d[5] = [npd, 157, 'coherent', 'layer', 0]
#    d[6] = [ito, 104, 'coherent', 'layer', 0]
#    d[7] = [glass, 5000, 'incoherent', 'layer', 0]
#    d[8] = [air, 0, 'incoherent', 'layer', 0]
#    
    
#    Stack1 = Stack(d).one_stack_intensity_calculate()
#    print(Stack1)
    
    # Graph 체크  
    fig = plt.figure()
    ax = fig.add_subplot(111)
    x = np.arange(380, 784, 4).reshape(-1,1)
    y = Stack1
    
    ax.plot(x, y/max(y), label='Stack')
    ax.legend()
    plt.show()
#    
    
    # 레이어 정보.(nk값가 동일한 변수이름의 레이어로 작성)
    # 중간에 npd 50에 대한 정보가 분명 존재. 그래서 만약 20이되면 그 옆에 있는 168이 198로 변경되어야함.
    # 근데 전체 그러한 pl이 실제로 존재하는 EML의 두께에 대한 정보가 아무리 봐도 식에 묻어나오지 않음.
    # 나중에 EML과 다른레이어들의 nk를 상이하게 한다면 EML에 대한 정보는 어디있나?
    # Electric filed 계산시 system matrix에 각각 layer matrix가 붙는데 여기서EML정보가 묻어나옴.
    # 각 연립방정식을 풀기위한 8개의 식에서 뭍어나오는건가?
    # 그리고 glass두꼐는 상관이 없나? 
    theta_ex = 0
# =============================================================================
#      System Matrix 직접 구현.\n,
#      - System_Matrix_A -> Al/NPD(218nm-50nm = 168nm)
#      - System_Matrix_B -> NPD(207nm-50nm = 157nm)/ITO(104nm)
#      - EML은 100nm
#        - Layer_Matrix_EML_L -> EML(50nm)
#        - Layer_Matrix_EML_R -> EML(50nm)
# =============================================================================
    
    al_npd = RT1(al, npd, theta_ex)
    npd_npd = RT1(npd, npd, theta_ex)
    npd_ito = RT1(npd, ito, theta_ex)
#    ito_glass = RT1(ito, glass, theta_ex)
    
    # Graph r_s  
    fig = plt.figure()
    ax = fig.add_subplot(111)
    x = np.arange(380, 784, 4).reshape(-1,1)
    y1 = al_npd.r_s()
    y2 = npd_npd.r_s()
    y3 = npd_ito.r_s()
    
    ax.plot(x, y1, label='al_npd')
    ax.plot(x, y2, label='npd_npd')
    ax.plot(x, y3, label='npd_ito')
    
    ax.set_title("r_s of each layers")
    ax.legend()
    plt.show()


    # Graph 체크_각 t값들..  
    fig = plt.figure()
    ax = fig.add_subplot(111)
    x = np.arange(380, 784, 4).reshape(-1,1)
    y1 = al_npd.t_s()
    y2 = npd_npd.t_s()
    y3 = npd_ito.t_s()
    
    ax.plot(x, y1, label='al_npd')
    ax.plot(x, y2, label='npd_npd')
    ax.plot(x, y3, label='npd_ito')
    
    ax.set_title("t_s of each layers")
    ax.legend()
    plt.show()

    
    
    # ito_air는 나중에 Cavity Enhencement Factor 할때나 활용
    #1, #2  #3은 System_A에 포함.#3, #4, #5는 System_B에 포함
    # 궁금한점은 #4 즉 EML두께에 대한 Layer텀은 어디에 포함? 위상차에 포함되서 시스템에서 필요없는건가?
    # 그리고 #6은 나중에 Ts-a에서 알아서 반영되는건가?
#    total = 415
#    yg_posi = 212 
    al_npd_IM = Interface_Matrix(al_npd)    #1
    npd1_LM = Layer_Matrix(npd, 203, theta_ex)        #2
    npd_npd_IM = Interface_Matrix(npd_ito)  #3  
    npd2_LM = Layer_Matrix(npd, 212, theta_ex)         #4
    npd_ito_IM = Interface_Matrix(npd_ito)  #5
#    npd_ito_IM.matrix_s()
#    ito_LM = Layer_Matrix(ito, 120, theta_ex)      #6
#    ito_glass_IM = Interface_Matrix(ito_glass)  #5

    sm_A, sm_B = System_Matrix(al_npd_IM, npd1_LM, npd_npd_IM), System_Matrix(npd_npd_IM, npd2_LM, npd_ito_IM)
    
    print("sm_A == sm_B???", sm_A.system_matrix_s() == sm_B.system_matrix_s())
    print("sm_A\n", sm_A.system_matrix_s())
    print("sm_B\n", sm_B.system_matrix_s())
    rt2 = RT2(sm_A, sm_B)

    power_rt = Power_RT(rt2)

    # Graph 체크_각 Power R, Power T값들..  
    fig = plt.figure()
    ax = fig.add_subplot(111)
    x = np.arange(380, 784, 4).reshape(-1,1)
    y1 = power_rt.R_A_s()
    y2 = power_rt.R_B_s()
    y3 = power_rt.T_B_s(npd, ito)
#    y4 = power_rt.T_B_s()
    
    ax.plot(x, y1, label='R_A_s')
    ax.plot(x, y2, label='R_B_s')
    ax.plot(x, y3, label='T_B_s')
#    ax.plot(x, y4, label='ito_glass')
    
    ax.set_title("R, T of A/B systems")
    ax.legend()
    plt.show()
#    
#    
    
    npd_k = K(npd, 0)
    phase1 = Phase_Change1(rt2)
    phase2 = Phase_Change2(npd_k, 10, 20, phase1)
    coherent_intensity = Intensity(YG_PL, power_rt, phase2, npd, glass)
#    print("coherent_intensity.intensity()", coherent_intensity.intensity())
#    print("coherent_intensity.intensity().shape", coherent_intensity.intensity().shape)
    
    
    # T_sub 계산
    t_sub_ito_glass = T_SUB(glass, ito, air)
#    print('t_sub_ito_glass', t_sub_ito_glass.T_sub_s())
#    print('t_sub_ito_glass.shape', t_sub_ito_glass.T_sub_s().shape)
    
#    print(YG_EL)
    
    
    # Graph 체크  
    fig = plt.figure()
    ax = fig.add_subplot(111)
    x = np.arange(380, 784, 4).reshape(-1,1)
    y = t_sub_ito_glass.T_sub_s()
    
    ax.plot(x, y, label='T_sub_s()')
    ax.legend()
    plt.show()
    
    
    
    # 최종 Cavity 계산
    final_incoherent_intensity = G(coherent_intensity, t_sub_ito_glass).g_final()
#    print("final_incoherent_intensity", final_incoherent_intensity)
#    print("final_incoherent_intensity.shape", final_incoherent_intensity.shape)
    sss = pd.DataFrame(final_incoherent_intensity)
    sss.to_csv("../nk/YG_Sim1.csv")
    
#     Graph로 확인

    # YG_EL 전체 418mnm, YG 218nm, ITO 104nm
    YG_EL_path = "../nk/YG_EL.csv"
    YG_EL_temp = pd.read_csv(YG_EL_path, sep=',', header=None)
    YG_EL = YG_EL_temp.values[1:, 1].astype(np.float64).reshape(-1,1)
    # YG_EL 전체 418mnm, YG 223nm(EML 중심에서 발광 가정 시), ITO 104nm
    YG_EL2_path = "../nk/YG_EL2.csv"
    YG_EL2_temp = pd.read_csv(YG_EL2_path, sep=',', header=None)
    YG_EL2 = YG_EL2_temp.values[1:, 1].astype(np.float64).reshape(-1,1)
    # YG_EL 전체 418mnm, YG 223nm(EML 중심에서 발광 가정 시), ITO 0nm(현재 이시뮬도 ITO 없으므로)
    YG_EL3_path = "../nk/YG_EL3.csv"
    YG_EL3_temp = pd.read_csv(YG_EL3_path, sep=',', header=None)
    YG_EL3 = YG_EL3_temp.values[1:, 1].astype(np.float64).reshape(-1,1)
    # YG_EL 전체 418mnm, YG 223nm(EML 중심에서 발광 가정 시), ITO 0nm(현재 이시뮬도 ITO 없으므로)
    YG_EL4_path = "../nk/YG_EL4.csv"
    YG_EL4_temp = pd.read_csv(YG_EL4_path, sep=',', header=None)
    YG_EL4 = YG_EL4_temp.values[1:, 1].astype(np.float64).reshape(-1,1)
    # YG_EL 전체 425nm, YG 212nm
    YG_EL5_path = "../nk/YG_EL5.csv"
    YG_EL5_temp = pd.read_csv(YG_EL5_path, sep=',', header=None)
    YG_EL5 = YG_EL5_temp.values[1:, 1].astype(np.float64).reshape(-1,1)



#    print(YG_EL)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    x = np.arange(380, 784, 4).reshape(-1,1)
    y1 = coherent_intensity.intensity()
    y2 = final_incoherent_intensity
    y3 = Stack1
    ax.plot(x, y1/np.max(y1), label='Coherent_Sim_result')
#    ax.plot(x, y2/np.max(y2), label='Incoherent_Sim_result', linestyle='--')
#    ax.plot(x, YG_EL/np.max(YG_EL), label='Labview_Simulation_418/218nm')
#    ax.plot(x, YG_EL2/np.max(YG_EL2), label='Labview_Simulation_418/223nm')
#    ax.plot(x, YG_EL3/np.max(YG_EL3), label='Labview_Simulation_418/223nm, ITO 0nm')
#    ax.plot(x, YG_EL4/np.max(YG_EL4), label='Labview_Simulation_418/223nm/ITO 0')
    ax.plot(x, YG_PL.get_PL()/np.max(YG_PL.get_PL()), label='YG_PL')
#    ax.plot(x, YG_EL5/np.max(YG_EL5), label='Labview_Simulation_418/223nm/ITO 0')
    ax.plot(x, y3/max(y3), label='Stack')
    
    ax.set_title("Cavity Simulation(Labview vs Python)")
    ax.set_xlabel('Wavelength')
    ax.set_ylabel('Intensity')
    ax.legend()
    
    plt.show()


    # 실행 시간 계산
    end_time = datetime.now()
     
    print("Executing Time : ", end_time-start_time)
