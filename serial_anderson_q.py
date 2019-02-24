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
'''
start_time = time.time()
L = 102
J = 1.0
mu0 = 5.0
A = 1.0
w = 0.25
dt = 0.01
tf = 15.0
nT = int(tf/dt)
save_data = False
'''
	#########################################  Main Code  ##############################################
'''
mu_array = mu0*np.ones(L)
#initialize correlation matrix to the ground state of clean Hamiltonian at t=0
ham_init = func.ham_anderson_insl_PBC(L,J,0,mu_array)
CC_0 = ground_state(L//2,ham_init)
print(np.shape(CC_0))
CC_t = CC_0.copy()

#Initialize the storage matrices
T = np.linspace(0.0,tf,nT)
fname = "WDIR/SER_NODIS_L_%d_A_%g_w_%g_mu0_%g_num_%d_tf_%g_dt_%g_"%(L,A,w,mu0,1,tf,dt)
m_energy = np.zeros(nT)
m_nbar = np.zeros(nT)
m_corr = np.zeros((nT,L))
m_spectrum = np.zeros((nT,L))
occ_indx = np.arange(L/2)-L//4
m_exact_energy = np.zeros(nT)
#extract Hamiltonian paramter from quspin simulation

'''
	#########################################  Loop over Time  ##############################################
'''
for i in range(nT):
	# print(np.trace(CC_t))
	if i%100==0:
		print('t = %g'%(i*dt))
	#Calculate Hamiltonian		 
	phi_hop = A*np.sin(w*T[i])*T[i]**2	
	HH = func.ham_anderson_insl_PBC(L,J,phi_hop,mu_array)
	# if not func.is_hermitian(HH):
	# 	print(HH)
	#Calculate exact energy
	m_exact_energy[i] = sum([(mu0-2*J*np.cos(2.0*np.pi*m/L - phi_hop)) for m in occ_indx]).real
	#Calculate energy
	m_energy[i] = np.sum(np.multiply(HH,CC_t)).real
	#Calculate occupation of subsystem
	# diagonal = np.diag(CC_t).real
	# m_corr[i,:] = diagonal.copy()
	# m_nbar[i] = np.sum(diagonal)
	#Time evolution of correlation matrix
	v_eps , DD = np.linalg.eigh(HH)	
	CC_next = evolve_correlation_matrix(dt,CC_t,v_eps,DD)
	CC_t = CC_next.copy()
	
'''
	#############################	Save Data Files 	###################################
'''
if save_data == True:
	# np.save(fname+"nbar.npy",m_nbar)
	np.save(fname+"energy.npy",m_energy)
	# np.save(fname+"mu.npy",mu_array)
	# np.save(fname+"corr.npy",m_corr)
	print('Data files saved successfully.')
	print('Filename : ',fname)

'''
############################	Plot the data for comparison	########################################

'''
plt.plot(T,m_energy,label = 'simulation')
plt.plot(T,m_exact_energy,label="exact")
# plt.plot(T,EE_q,label='quspin')
plt.legend()
plt.show()
plt.plot(T,m_energy - m_exact_energy)
plt.show()
end_time = time.time()
print("Simulation Done. Time taken : ",(end_time - start_time)," seconds")

