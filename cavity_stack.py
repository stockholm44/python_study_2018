import numpy as np
import pandas as pd
from cavity_input import *
from cavity_calc2 import *
import matplotlib.pyplot as plt
import collections
import pprint
from datetime import datetime

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



# Input 형태는 아래와 같이 OrderredDict 이며 
# 순서대로, "layer nk값/두께/coherent 여부/Layer형태/EL_Factor/Layer 이름" 6개 항으로 구성.
'''
input 된 df 예시
   layer_name  thickness    coherent  layer  el_factor
no                                                    
1          al        100    coherent  layer          0
2         npd        106    coherent  layer          0
3         npd         10    coherent  layer          0
4        B_PL          0    coherent     pl          1
5         npd         10    coherent  layer          0
6         npd        285    coherent  layer          0
7         ito        120    coherent  layer          0
8       glass       5000    coherent  layer          0
9         air          0  incoherent  layer          0
'''

def load_stack_excel(path):
    df = pd.read_excel(path, sheet_name=0, header=0, index_col=0)
    return df

# 참고 Setfos의 경우
# Layer Nme/Thickness/Refractive index/Electrical material

'''
0. stack 구조를 excel로 저장.
1. 받은 excel DataFrame으로 받아서
2. pl index 위치들 가져오기(save_pol_index 메서드)
3. pl 위치 리스트를 이용해서 df_A, df_B를 만들고 clean 하는 반복문 작성.(split_and_clean_df 메서드)
4. 그렇게 한번의 iteration으로 만들어진 df_A, df_B로 Intensity 클래스의 *args를 계산.(make_system_matrix_list 메서드)
   (※ *args는 [[A쪽의 interface/layer 메서드의 list], [A쪽의 interface/layer 메서드의 list]])로 이루어진 리스트 2개임.
   (※ df_A/B를 그냥 df라고 표현.)
4-1. df의 길이 파악.
4-2. 길이 고려하여 반복문으로 한개 list 만듬.    
5. Intensity 클래스로 한개 PL에 대한 Intensity 계산가능
6. for문과 EL_Factor(가중치)로 각 Stack의 Intensity 계산하여 전체 Intensity 계산.
'''

class Stack:
    def __init__(self, df): # parameter layers is DataFrame
        self.df = df
#        self.d = layers
#        self.layer_count = len(self.d)
        self.theta_col = [0, 15, 30, 45, 60]
    
    def save_pl_index(self):
        return self.df.index[self.df['layer'] == 'pl'].tolist()
    
    def split_and_clean_df(self, pl_index):
        df_A = self.df.iloc[0:pl_index]
        df_B = self.df.iloc[pl_index:]
#        print(df_A)
        df_A.drop(df_A.index[df_A['layer'] == 'pl'], inplace=True)
        df_A.drop(df_A.index[df_A['coherent'] == 'incoherent'], inplace=True)
        df_B.drop(df_B.index[df_B['layer'] == 'pl'], inplace=True)
        df_B.drop(df_B.index[df_B['coherent'] == 'incoherent'], inplace=True)
        
        return df_A, df_B 

    def make_system_matrix_list(self, df_system, theta_ex):
        df_system.reset_index(inplace=True,drop=True)
        matrix_list = []
        matrix_list.append(Interface_Matrix(eval(df_system['layer_name'][0]) , eval(df_system['layer_name'][1]), theta_ex))
        for i in range(1, len(df_system)-1):
            matrix_list.append(Layer_Matrix(eval(df_system['layer_name'][i]) , int(df_system['thickness'][i]), theta_ex))
            matrix_list.append(Interface_Matrix(eval(df_system['layer_name'][i]) , eval(df_system['layer_name'][i+1]), theta_ex))
        return matrix_list
        
    def calc_intensity(self, pl_index, theta_ex):
        df_A, df_B = self.split_and_clean_df(pl_index)
        matrix_list = [self.make_system_matrix_list(df_A, theta_ex), self.make_system_matrix_list(df_B, theta_ex)]
        horizontal_dipole_ratio = 0.67
        
        layer_name = self.df['layer_name']
        thickness = self.df['thickness']
        el_factor = self.df['el_factor']
        intensity_at_pl_theta = Intensity(eval(layer_name[pl_index]), eval(layer_name[pl_index+1]), thickness[pl_index-1], \
                                          thickness[pl_index+1], theta_ex, horizontal_dipole_ratio, *matrix_list).intensity()
        return intensity_at_pl_theta

    def calc_stack(self, theta_ex):
        pl_index_list = self.save_pl_index()
        final_intensity = np.zeros(101).reshape(-1, 1)
        for pl_index in pl_index_list:
            final_intensity = final_intensity + self.calc_intensity(pl_index, theta_ex)  * self.df['el_factor'][pl_index]
        return final_intensity



if __name__ == '__main__':
    # 기본정도(nk, pl) 불러오기
    
    # n, k값의 경로와 PL의 경로
    ito_path = "../nk/ITO_nk.csv"
    al_path = "../nk/Al_nk.csv"
    npd_path = "../nk/NPD_nk.csv"
    air_path = "../nk/Air_nk.csv"
    Glass_path = "../nk/Glass_nk.csv"
    
    YG_PL_path = "../nk/YG_PL.txt"
    B_PL_path = "../nk/B_PL.txt"

    # nk 미리 만들기
    al = get_nk(al_path)
    npd = get_nk(npd_path)
    ito = get_nk(ito_path)
    glass = get_nk(Glass_path)
    air = get_nk(air_path)    
    
    # pl 미리 만들기
    YG_PL = get_pl(YG_PL_path)
    B_PL = get_pl(B_PL_path)

    # 시간 기록
    start_time = datetime.now()
    

    # stack 계산 processing
    df = load_stack_excel('YG_mono.xlsx')
    yg_stack = Stack(df).calc_stack(0)
    print(df)

#    #그래프 그려보기
#    YG_EL_path = "../nk/YG_EL.csv"
#    YG_EL_temp = pd.read_csv(YG_EL_path, sep=',', header=None)
#    YG_EL = YG_EL_temp.values[1:, 1].astype(np.float64).reshape(-1,1)
#    
#    
#    fig = plt.figure()
#    ax = fig.add_subplot(111)
#    x = np.arange(380, 784, 4).reshape(-1,1)
#    y1 = yg_stack
#    ax.plot(x, y1/np.max(y1), label='Coherent_Sim_result')
#    ax.plot(x, YG_PL/np.max(YG_PL), label='YG_PL')
#    ax.plot(x, YG_EL/np.max(YG_EL), label='YG_EL')
##    
#    ax.set_title("Cavity Simulation(Labview vs Python)")
#    ax.set_xlabel('Wavelength')
#    ax.set_ylabel('Intensity')
#    ax.legend()
#    
#    plt.show()
#    
#
    # 시야각 변경시.(W Stack)
    W_EL_path = "../nk/W_EL.csv"
    W_EL_temp = pd.read_csv(W_EL_path, sep=',', header=None)
#    print(W_EL_temp)
    W_EL = W_EL_temp.values[0:101, 1].astype(np.float64).reshape(-1,1)
#    print(W_EL.shape)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    x = np.arange(380,784,4).reshape(-1,1)
    
    df = load_stack_excel('W_stack.xlsx')
    for i in np.arange(0, 75, 15):
        W_stack = Stack(df).calc_stack(i)
        ax.plot(x, W_stack, label=i)
#    ax.plot(x, W_EL/max(W_EL), label='W_EL')
    ax.set_title("Cavity Simulation(Change View Angle @ W Stack)")
    ax.legend()
    plt.show()
    
    
    # HTL1 변경시(W Stack)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    x = np.arange(380,784,4).reshape(-1,1)
    
    df = load_stack_excel('W_stack.xlsx')
    for i in [104, 109, 114, 119, 124]:
        df['thickness'][16] = i
        W_stack = Stack(df).calc_stack(0)
        ax.plot(x, W_stack, label=i)
    ax.set_title("Cavity Simulation(Change HTL1 @ W Stack)")
    ax.legend()
    plt.show()

    # HTL4 변경시(W Stack)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    x = np.arange(380,784,4).reshape(-1,1)
    
    df = load_stack_excel('W_stack.xlsx')
    for i in [154, 158, 162, 166, 170]:
        df['thickness'][5] = i
        W_stack = Stack(df).calc_stack(0)
        ax.plot(x, W_stack, label=i)
    ax.set_title("Cavity Simulation(Change HTL4 @ W Stack)")
    ax.legend()
    plt.show()
    
    df = load_stack_excel('W_stack.xlsx')
    W_stack = Stack(df)
    
    
    
    
    
    # 실행 시간 계산
    end_time = datetime.now()
     
    print("Executing Time : ", end_time-start_time)
