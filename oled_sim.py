import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ito_nk = pd.read_csv("./nk/ITO_nk.csv", index_col=['wavelength'])
al_nk = pd.read_csv("./nk/Al_nk.csv", index_col=['wavelength'])
npd_nk = pd.read_csv("./nk/NPD_nk.csv", index_col=['wavelength'])
air_nk = pd.read_csv("./nk/Air_nk.csv", index_col=['wavelength'])

# 560nm만 Test 하기위한 560nm의 각 n, k값 선언
# 500, 524, 552, 600, 652

ito_560nm_n = ito_nk.loc[652,'n']
ito_560nm_k = ito_nk.loc[652,'k']

al_560nm_n = al_nk.loc[652,'n']
al_560nm_k = al_nk.loc[652,'k']

npd_560nm_n = npd_nk.loc[652,'n']
npd_560nm_k = npd_nk.loc[652,'k']

air_560nm_n = air_nk.loc[652,'n']
air_560nm_k = air_nk.loc[652,'k']

YG_PL = pd.read_csv("./nk/YG_PL.txt", sep='\t', header=None, names=['Wavelength','PL'], index_col=['Wavelength'])

YG_PL_560nm = YG_PL.loc[652, 'PL']
YG_PL_560nm

horizontal_dipole_ratio = 0.67
theta = 0
theta1 = theta2 = theta 

def r_s(n1, n2, k1, k2, theta1, theta2):
    return ((n1-1j*k1)*np.cos(theta1) - (n2-1j*k2)*np.cos(theta2)) /  ((n1-1j*k1)*np.cos(theta1) + (n2-1j*k2)*np.cos(theta2))

def r_p(n1, n2, k1, k2, theta1, theta2):
    return ((n1-1j*k1)*np.cos(theta2) - (n2-1j*k2)*np.cos(theta1)) /  ((n1-1j*k1)*np.cos(theta2) + (n2-1j*k2)*np.cos(theta1))

def t_s(n1, n2, k1, k2, theta1, theta2):
    return (2 * (n1-1j*k1)*np.cos(theta1))/((n1-1j*k1)*np.cos(theta1) + (n2-1j*k2)*np.cos(theta2))

def t_p(n1, n2, k1, k2, theta1, theta2):
    return (2 * (n1-1j*k1)*np.cos(theta1))/((n1-1j*k1)*np.cos(theta2) + (n2-1j*k2)*np.cos(theta1))
    
    
System_Matrix_A_s = np.dot(interface_matrix_s(al_560nm_n, npd_560nm_n, al_560nm_k, npd_560nm_k, theta1, theta2), layer_matrix(npd_560nm_n, npd_560nm_k ,168)).dot(interface_matrix_s(npd_560nm_n, npd_560nm_n, npd_560nm_k, npd_560nm_k, theta1, theta2))
System_Matrix_B_s = np.dot(interface_matrix_s(npd_560nm_n, npd_560nm_n, npd_560nm_k, npd_560nm_k, theta1, theta2), layer_matrix(npd_560nm_n, npd_560nm_k, 157)).dot(interface_matrix_s(npd_560nm_n, ito_560nm_n, npd_560nm_k, ito_560nm_k, theta1, theta2)).dot(layer_matrix(ito_560nm_n, ito_560nm_k, 104)).dot(interface_matrix_s(ito_560nm_n, air_560nm_n, ito_560nm_k, air_560nm_k, theta1, theta2))

System_Matrix_A_p = np.dot(interface_matrix_p(al_560nm_n, npd_560nm_n, al_560nm_k, npd_560nm_k, theta1, theta2), layer_matrix(npd_560nm_n, npd_560nm_k ,168)).dot(interface_matrix_p(npd_560nm_n, npd_560nm_n, npd_560nm_k, npd_560nm_k, theta1, theta2))
System_Matrix_B_p = np.dot(interface_matrix_p(npd_560nm_n, npd_560nm_n, npd_560nm_k, npd_560nm_k, theta1, theta2), layer_matrix(npd_560nm_n, npd_560nm_k, 157)).dot(interface_matrix_p(npd_560nm_n, ito_560nm_n, npd_560nm_k, ito_560nm_k, theta1, theta2)).dot(layer_matrix(ito_560nm_n, ito_560nm_k, 104)).dot(interface_matrix_p(ito_560nm_n, air_560nm_n, ito_560nm_k, air_560nm_k, theta1, theta2))

Layer_Matrix_EML_L = layer_matrix(npd_560nm_n, npd_560nm_k, 50)
Layer_Matrix_EML_R = layer_matrix(npd_560nm_n, npd_560nm_k, 50)

gls_560nm_n = 1.5

n_EML = npd_560nm_n
k_EML = npd_560nm_k

n_nPlus1 = 1 # n_n+1을 최종단에 incoherent layer로 사용해서 glass로 넣었는데 공기로 넣어햐하는지 모르겠다.
I_R_EML = YG_PL_560nm


R_A_s = np.square(np.absolute(Reflection_Coef_A_s)) # 식29 30사이에는 exp^-j term이 곱해지긴한데. 일단은 제곱텀으로 해다.
R_B_s = np.square(np.absolute(Reflection_Coef_B_s))

R_A_p = np.square(np.absolute(Reflection_Coef_A_p))
R_B_p = np.square(np.absolute(Reflection_Coef_B_p))


# T_A = n_nPlus1 * np.cos(theta) /(n_EML * np.cos(theta)) * np.square(Transmission_Coef_A)
T_B_s = n_nPlus1 * np.cos(theta) /(n_EML * np.cos(theta)) * np.square(np.absolute(Transmission_Coef_B_s))
T_B_p = n_nPlus1 * np.cos(theta) /(n_EML * np.cos(theta)) * np.square(np.absolute(Transmission_Coef_B_p))

pi_A = np.pi
pi_B = 0 # ITO의 위상차가 없네. 그거 고려해서 식 세워.


pi_FP = 2 * 2 * np.pi/560 * (n_EML-1j*k_EML) * np.cos(theta) * 50 - pi_A - pi_B  
# k가 2파이/람다 인줄 알았는데 더 복잡
# k = 2파이/람다 *n_w*cos(theta_w)
pi_TB_B = 2 * 2 * np.pi/560 * (n_EML-1j*k_EML) * np.cos(theta) * 50 -pi_B
pi_TB_A = 2 * 2 * np.pi/560 * (n_EML-1j*k_EML) * np.cos(theta) * 50 -pi_A


I_R_nPlus1_s = I_R_EML * T_B_s * (1 + R_A_s + 2 * np.sqrt(R_A_s) * np.cos(pi_TB_A))/(1 + R_A_s * R_B_s - 2 * np.sqrt(R_A_s) * np.sqrt(R_B_s)* np.cos(pi_FP)) * np.square(n_nPlus1) * np.cos(theta)/(np.square(n_EML)* np.cos(theta))* 3/(16 * np.pi) * horizontal_dipole_ratio
I_R_nPlus1_p = I_R_EML * T_B_p * (1 + R_A_p + 2 * np.sqrt(R_A_p) * np.cos(pi_TB_A))/(1 + R_A_p * R_B_p - 2 * np.sqrt(R_A_p) * np.sqrt(R_B_p)* np.cos(pi_FP)) * np.square(n_nPlus1) * np.cos(theta)/(np.square(n_EML)* np.cos(theta))* (horizontal_dipole_ratio * 3/(16 * np.pi) * np.square(np.cos(theta)) + (1 - horizontal_dipole_ratio) * 3 /(8 * np.pi) * np.square(np.sin(theta)))


I_R_nPlus1 = I_R_nPlus1_s + I_R_nPlus1_p
