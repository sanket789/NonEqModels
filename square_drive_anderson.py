from __future__ import print_function, division

import numpy as np
import methods as func
import time
import matplotlib.pyplot as plt
from corr_init import ground_state
from time_evolution import evolve_correlation_matrix

'''
	This script is for time evolution of Anderson insulator using correlators method. 
	The code is written in serial execution manner for single configuration.  
	https://doi.org/10.1103/PhysRevB.98.214202	
	The reults of this test code has to compared with time evolution using quspin package
	The Anderson insulator is driven such that the hopping apmlitude switches between (J+dJ) and (J-dJ) with period T.
'''
start_time = time.time()
L = 100
J = 1.0
mu0 = 5.0
dJ = 0.2*J
T = 1.
tf = 100.0
# dt = 0.1
# dt = 0.001
nT = 2*int(tf/T)
save_data = True
fname = 'WDIR/test/Serial_0_'#"WDIR/test/SQ_L_%d_T_%g_mu0_%g_tf_%g_dJ_%g_"%(L,T,mu0,tf,dJ)

'''
	#########################################  Main Code  ##############################################
'''
mu_array = np.random.uniform(-mu0,mu0,L)
np.save("WDIR/test/mu_0.npy",mu_array)
#Define two Hamiltonians
#Hamiltonian with (J+dJ)
HH_h = np.diag(mu_array) - (J+dJ)*np.diag(np.ones(L-1),1)- (J+dJ)*np.diag(np.ones(L-1),-1)
HH_h[-1,0] = -(J+dJ)
HH_h[0,-1] = -(J+dJ)
v_eps_h , DD_h = np.linalg.eigh(HH_h)
EE_h = np.diag(np.exp(-1j*0.5*T*v_eps_h))
UU_h = np.dot(np.conj(DD_h),np.dot(EE_h,DD_h.T))
#Hamiltonian with (J-dJ)
HH_l = np.diag(mu_array) - (J-dJ)*np.diag(np.ones(L-1),1)- (J-dJ)*np.diag(np.ones(L-1),-1)
HH_l[-1,0] = -(J-dJ)
HH_l[0,-1] = -(J-dJ)
v_eps_l , DD_l = np.linalg.eigh(HH_l)
EE_l = np.diag(np.exp(-1j*0.5*T*v_eps_l))
UU_l = np.dot(np.conj(DD_l),np.dot(EE_l,DD_l.T))
#initialize correlation matrix to the ground state of HH_start
HH_start = - (J+dJ)*np.diag(np.ones(L-1),1)- (J+dJ)*np.diag(np.ones(L-1),-1)
HH_start[-1,0] = -(J+dJ)
HH_start[0,-1] = -(J+dJ) 
CC_0 = ground_state(L//2,HH_start)
print(np.shape(CC_0))
CC_t = CC_0.copy()

#Initialize the storage matrices
tlist = np.linspace(0.0,tf,nT,endpoint=False)
m_energy = np.zeros(nT)
m_nbar = np.zeros(nT)
m_corr = np.zeros((nT,L))
m_spectrum = np.zeros((nT,L))
occ_indx = np.arange(L/2)-L//4
m_exact_energy = np.zeros(nT)
ham = np.zeros(nT)
#extract Hamiltonian paramter from quspin simulation

'''
	#########################################  Loop over Time  ##############################################
'''
for i in range(nT):
	#Calculate Hamiltonian		 
	if i%2 == 0:
		m_energy[i] = np.sum(np.multiply(HH_h,CC_t)).real
		m_nbar[i] = np.sum(np.diag(CC_t)[0:L//2]).real
		CC_next = np.dot(np.conj(UU_h.T),np.dot(CC_t,UU_h))
		# m_exact_energy[i] = sum([-2*(J+dJ)*np.cos(2*np.pi*m/L)+mu0 for m in occ_indx]).real
	else:
		m_energy[i] = np.sum(np.multiply(HH_l,CC_t)).real
		m_nbar[i] = np.sum(np.diag(CC_t)[0:L//2]).real
		CC_next = np.dot(np.conj(UU_l.T),np.dot(CC_t,UU_l))
		# m_exact_energy[i] = sum([-2*(J-dJ)*np.cos(2*np.pi*m/L)+mu0 for m in occ_indx]).real
	CC_t = CC_next.copy()
	
'''
	#############################	Save Data Files 	###################################
'''
if save_data == True:
	np.save(fname+"nbar.npy",m_nbar)
	np.save(fname+"energy.npy",m_energy)
	# np.save(fname+"mu.npy",mu_array)
	# np.save(fname+"corr.npy",m_corr)
	print('Data files saved successfully.')
	print('Filename : ',fname)

'''
############################	Plot the data for comparison	########################################

'''
# EE_q_load = np.load("WDIR/Quspin_SQ_L_%d_T_%g_mu0_%g_tf_%g_dJ_%g_energy.npy"%(L,T,mu0,tf,dJ))
# # EE_q = [EE_q_load[i] for i in 5*np.arange(nT)]
# plt.plot(tlist,m_energy,label = 'simulation')
# # plt.plot(tlist,m_exact_energy,'--',label="exact")
# plt.plot(tlist,EE_q_load,label='quspin')
# plt.xlabel('time')
# plt.ylabel('energy')
# plt.title('Energy for L = %d dJ = %g T = %g mu0 = %g '%(L,dJ,T,mu0))
# plt.legend()
# plt.savefig('energy_square.png')
# plt.show()
# plt.plot(tlist,np.abs(m_energy - EE_q_load))
# plt.xlabel('time')
# plt.ylabel('|E_c - E_quspin|')
# plt.title(' Error: L = %d dJ = %g T = %g mu0 = %g '%(L,dJ,T,mu0))
# plt.savefig('error_energy_square.png')
# plt.show()
# end_time = time.time()
# print("Simulation Done. Time taken : ",(end_time - start_time)," seconds")

