import numpy as np
import pandas as pd
from cavity_input import *
from cavity_calc3 import *
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

YG_PL_path = "../nk/YG_PL.csv"
B_PL_path = "../nk/B_PL.csv"
R_PL_path = "../nk/R_PL.csv"

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
        self.df = df.copy()
        self.theta_col = [0, 15, 30, 45, 60]

    def save_pl_index(self):
        return self.df.index[self.df['layer_func'] == 'organic_EML'].tolist()

    def split_and_clean_df(self, pl_index):
        EML_A_thickness = self.df.loc[pl_index, 'pl_position']
        EML_B_thickness = self.df.loc[pl_index, 'thickness'] - EML_A_thickness
        #         if EML_B_thickness < 0:
#             두께가 0 아래면 에러발생시키자.
        df_A = self.df.iloc[0:pl_index+1, :].copy()
        df_B = self.df.iloc[pl_index:, :].copy()
        
        df_A.loc[pl_index, 'thickness'] = EML_A_thickness
        df_B.loc[pl_index, 'thickness'] = EML_B_thickness
        
        return df_A, df_B

    def make_system_matrix(self, df_system, theta_ex, ab):
        df_system.reset_index(inplace=True,drop=True)
        matrix_list = []
        if ab == 'B': # System_B일경우 처음에 PL 계면에서의 Interface_M과 Layer_M을 추가해줘야함. 
            matrix_list.append(Interface_Matrix(eval(df_system['material'][0]) , eval(df_system['material'][0]), theta_ex))
            print("Interface_Matrix", df_system['material'][0], df_system['material'][0])
            matrix_list.append(Layer_Matrix(eval(df_system['material'][0]) , int(df_system['thickness'][0]), theta_ex))
            print("Layer_Matrix", df_system['material'][0] , df_system['thickness'][0])
        matrix_list.append(Interface_Matrix(eval(df_system['material'][0]) , eval(df_system['material'][1]), theta_ex))
        print("Interface_Matrix", df_system['material'][0], df_system['material'][1])
        for i in range(1, len(df_system)-1):
            matrix_list.append(Layer_Matrix(eval(df_system['material'][i]) , int(df_system['thickness'][i]), theta_ex))
            print("Layer_Matrix", df_system['material'][i] , df_system['thickness'][i])
            matrix_list.append(Interface_Matrix(eval(df_system['material'][i]) , eval(df_system['material'][i+1]), theta_ex))
            print("Interface_Matrix", df_system['material'][i], df_system['material'][i])
        if ab == 'A': # System_A일 경우 마지막에 EML의 Layer_M고 PL 계면에서의 Interface_M을 추가해줘야함.
            end = len(df_system)-1
            matrix_list.append(Layer_Matrix(eval(df_system['material'][end]) , int(df_system['thickness'][end]), theta_ex))
            print("Layer_Matrix", df_system['material'][end] , df_system['thickness'][end])
            matrix_list.append(Interface_Matrix(eval(df_system['material'][end]) , eval(df_system['material'][end]), theta_ex))
            print("Interface_Matrix", df_system['material'][end], df_system['material'][end])
        return matrix_list

    def calc_phase_change_TB(self, pl_index, theta_ex):
        df, _ = self.split_and_clean_df(pl_index)
        df.drop(index=df.loc[df['layer_func']=='cathode'].index, axis=0, inplace=True)
        
        material_list = df['material'].values
        thickness_list = df['thickness'].values
        theta = calc_theta(get_n(eval(self.df.loc[pl_index, 'material'])), theta_ex)
        
        phase_TB = np.ones((101, 1))
        
        for mat, thick in zip(material_list, thickness_list):
            phase_TB = phase_TB + (get_n(eval(mat)) * thick)
        
        return phase_TB * np.cos(theta)
        
        

    def calc_phase_change_total(self, theta_ex):
        df = self.df.copy()
        
        df.drop(index=df.loc[df['layer_func']=='cathode'].index, axis=0, inplace=True)
        df.drop(index=df.loc[df['layer_func']=='substrate'].index, axis=0, inplace=True)

        material_list = df['material'].values
        thickness_list = df['thickness'].values
        
        phase_total = np.ones((101, 1))
        
        for mat, thick in zip(material_list, thickness_list):
            theta = calc_theta(get_n(eval(mat)), theta_ex)
            phase_total = phase_total + (get_n(eval(mat)) * thick * np.cos(theta))
        
        return phase_total

    def calc_thickness(self, pl_index):
        df_A, df_B = self.split_and_clean_df(pl_index)
        df_A.drop(index=df_A.loc[df['layer_func']=='cathode'].index, axis=0, inplace=True)
        df_B.drop(index=df_B.loc[df['layer_func']=='substrate'].index, axis=0, inplace=True)

        thick_A = df_A['thickness'].sum()
        thick_B = df_B['thickness'].sum()
        return thick_A, thick_B
    
    def calc_intensity(self, pl_index, theta_ex):
        df_A, df_B = self.split_and_clean_df(pl_index)
        matrix_list = [self.make_system_matrix(df_A, theta_ex, 'A'), self.make_system_matrix(df_B, theta_ex, 'B')]
        
        horizontal_dipole_ratio = 0.67
        phase_TB = self.calc_phase_change_TB(pl_index, theta_ex)
        phase_total = self.calc_phase_change_total(theta_ex)
        thick_A, thick_B = self.calc_thickness(pl_index)
        
        layer_name = self.df['material']
        thickness = self.df['thickness']
        el_factor = self.df['el_factor']
                                #(self, pl, nk_e, nk_g, z_ex, Z, theta_ex, horizontal_dipole_ratio, phase_TB=0, phase_total=0, *args)
        self.intensity_at_pl_theta = Intensity(eval(self.df['pl'][pl_index]), eval(self.df['material'][pl_index]),glass , thick_A, \
                                         thick_B, theta_ex, horizontal_dipole_ratio, phase_TB, phase_total, *matrix_list)

        return self.intensity_at_pl_theta.intensity()

    
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
    
    YG_PL_path = "../nk/YG_PL.csv"
    B_PL_path = "../nk/B_PL.csv"
    R_PL_path = "../nk/R_PL.csv"

    # nk 미리 만들기
    al = get_nk(al_path)
    npd = get_nk(npd_path)
    ito = get_nk(ito_path)
    glass = get_nk(Glass_path)
    air = get_nk(air_path)    
    
    # pl 미리 만들기
    YG_PL = get_pl(YG_PL_path)
    B_PL = get_pl(B_PL_path)
    R_PL = get_pl(R_PL_path)

    # 시간 기록
    start_time = datetime.now()
    

    # stack 계산 processing
    df = load_stack_excel('W2_stack.xlsx')
#    yg_stack = Stack(df).calc_stack(0)
    print(df)

    # stack 계산 processing
    a = Stack(df)


    fig = plt.figure(figsize=(10,8))
    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(312)
    ax3 = fig.add_subplot(313)
    
    x = np.arange(380,784,4).reshape(-1,1)
        
    for va in [0, 15, 30,45,60]:
        y1 = a.calc_stack(va)
        y2 = a.calc_phase_change_TB(3, va)
        y3 = a.calc_phase_change_total(va)
        
        ax1.plot(x, y1, label=va)
        ax2.plot(x, y2, label=va)
        ax3.plot(x, y3, label=va)
    
    ax1.set_title("Int")
    ax2.set_title("phase_FP")
    ax3.set_title("phase_total")
    
    plt.legend()    
    plt.show()


    
    
    # 실행 시간 계산
    end_time = datetime.now()
     
    print("Executing Time : ", end_time-start_time)