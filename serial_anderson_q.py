from __future__ import print_function, division

import numpy as np
import methods as func
import time
# from mpi4py import MPI
# import argparse
import matplotlib.pyplot as plt
'''
	This script is for time evolution of Anderson insulator using correlators method. 
	The code is written in serial execution manner for single configuration.  
	https://doi.org/10.1103/PhysRevB.98.214202	
	The reults of this test code has to compared with time evolution using quspin package
'''
start_time = time.time()
L = 10
J = 1.0
mu0 = 5.0
A = 0.001
w = 0.5
dt = 0.001
tf = 100.0
nT = int(tf/dt)
save_data = True
#initialize correlation matrix to the ground state of clean Hamiltonian at t=0

CC_0 = np.zeros((L,L),dtype=complex)
for i in range(L//2):
	 CC_0[i,i] = 1.0

#Initialize the storage matrices
T = np.linspace(0.0,tf,nT)
fname = "WDIR/CORR_L_%d_A_%g_w_%g_mu0_%g_num_%d_tf_%g_dt_%g_"%(L,A,w,mu0,1,tf,dt)

m_energy = np.zeros(nT)
m_nbar = np.zeros(nT)
m_corr = np.zeros(nT)
m_spectrum = np.zeros((nT,L))
# m_deln = np.zeros((num_mu0,L))
# m_Einf = np.zeros((num_mu0))
#Loop over nT

CC_t = CC_0.copy()
mu_array = mu0*np.ones(L)#np.random.uniform(-1.0*mu0,mu0,L)
# print(mu_array[p,0,0])
for i in range(nT):
	#Calculate Hamiltonian	
	m_corr[i] = CC_t[0,0].real		 
	phi_hop = A*np.cos(w*T[i])	
	
	HH = func.ham_anderson_insl_PBC(L,J,phi_hop,mu_array)
	if i==0:
		for j in range(L):
			for k in range(L):
				pass# print(repr(HH[j,k]))
	if not func.is_hermitian(HH):
		print(HH)
	#Calculate energy
	m_energy[i] = np.sum(np.multiply(HH,CC_t)).real
	#Calculate occupation of subsystem
	diagonal = np.diag(CC_t).real
	m_nbar[i] = np.sum(diagonal)
	# print(repr(m_nbar[i]))
	#Time evolution of correlation matrix. Refer readme for formulae
	v_eps , DD = np.linalg.eigh(HH)
	# m_spectrum[i,:] =np.linalg.eigvalsh(HH)
	EE = np.diag(np.exp(-1j*dt*v_eps))
	UU = np.dot(np.conj(DD),np.dot(EE,DD.T))
	# if i==0:
	# 	EE2 = np.diag(np.exp(-1j*tf*v_eps))
	# 	UU_2 = np.dot(np.conj(DD),np.dot(EE2,DD.T))
	# 	CC_2 = np.dot(np.conj(UU_2.T),np.dot(CC_t,UU_2))
	CC_next = np.dot(np.conj(UU.T),np.dot(CC_t,UU))
	CC_t = CC_next.copy()
	# print (repr(m_spectrum[i,-1]))

print('--------------------------------------')
print("Constant total number opeartor = ",np.allclose(m_nbar,0.5*L*np.ones(np.shape(m_nbar))),0.5*L)
print('--------------------------------------')
if save_data == True:
	np.save(fname+"nbar.npy",m_nbar)
	np.save(fname+"energy.npy",m_energy)
	np.save(fname+"mu.npy",mu_array)
	np.save(fname+"corr.npy",m_corr)
	# np.save(fname+"spectrum.npy",m_spectrum)
	print('Data files saved successfully.')
	print('Filename : ',fname)
end_time = time.time()

