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
A = 1.
w = 0.25
dt = 0.0001
tf = 100.0
nT = int(tf/dt)
save_data = True
tol = 1E-3
#initialize correlation matrix to the ground state of clean Hamiltonian at t=0

CC_0 = np.zeros((L,L),dtype=complex)
for i in range(L//2):
	 CC_0[i,i] = 1.0

#Initialize the storage matrices
T = np.linspace(0.0,tf,nT)
fname = "WDIR/SER_L_%d_A_%g_w_%g_mu0_%g_num_%d_tf_%g_dt_%g_"%(L,A,w,mu0,1,tf,dt)

m_energy = np.zeros(nT)
m_nbar = np.zeros(nT)
m_corr = np.zeros((nT,L))
m_spectrum = np.zeros((nT,L))
# m_deln = np.zeros((num_mu0,L))
# m_Einf = np.zeros((num_mu0))
#Loop over nT

CC_t = CC_0.copy()
mu_array = np.load("WDIR/QUSPIN_L_%d_A_%g_w_%g_mu0_%g_num_%d_tf_%g_dt_%g_mu.npy"%(L,A,w,mu0,1,tf,dt))
# print(mu_array[p,0,0])
for i in range(nT):
	# print(CC_t)
	#Calculate Hamiltonian	
	if i%100==0:
		print('t = %g'%(i*dt))	 
	phi_hop = A*np.cos(w*T[i])	
	
	HH = func.ham_anderson_insl_PBC(L,J,phi_hop,mu_array)
	
	if not func.is_hermitian(HH):
		print(HH)
	#Calculate energy
	m_energy[i] = np.sum(np.multiply(HH,CC_t)).real
	#Calculate occupation of subsystem
	diagonal = np.diag(CC_t).real
	m_corr[i,:] = diagonal.copy()
	m_nbar[i] = np.sum(diagonal)
	# print(repr(m_nbar[i]))
	#Time evolution of correlation matrix. Refer readme for formulae
	v_eps , DD = np.linalg.eigh(HH)
	# m_spectrum[i,:] =np.linalg.eigvalsh(HH)
	EE = np.diag(np.exp(-1j*dt*v_eps))
	UU = np.dot(np.conj(DD),np.dot(EE,DD.T))
	if i==nT-1:
		print('EE ',(np.where(np.absolute(EE)<tol)[1].size)*100/L**2,'%')
		print('DD ',(np.where(np.absolute(DD)<tol)[1].size)*100/L**2,'%')
		print('EE*DD.T ',(np.where(np.absolute(np.dot(EE,DD.T))<tol)[1].size)*100/L**2,'%')	
		print('UU ',(np.where(np.absolute(UU)<tol)[1].size)*100/L**2,'%')

		print('---------------------------------------------')
	
	CC_next = np.dot(np.conj(UU.T),np.dot(CC_t,UU))
	CC_t = CC_next.copy()
	# print (repr(m_spectrum[i,-1]))
# plt.plot(range(L),np.absolute(DD[:,0]))
# plt.plot(range(L),np.absolute(DD[:,-1]))
# plt.show()
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
plt.plot(T,m_energy)
plt.title('Energy dt = %g'%dt)
plt.xlabel('time')
plt.ylabel('energy')
plt.savefig('WDIR/energy_dt_%g_A_%g_w_%g_.png'%(dt,A,w))
plt.show()
###########################################################################
#
#	Plot the data for comparison
#
###########################################################################

EE_q = np.load("WDIR/QUSPIN_L_%d_A_%g_w_%g_mu0_%g_num_%d_tf_%g_dt_%g_energy.npy"%(L,A,w,mu0,1,tf,dt))
cc_q = np.load("WDIR/QUSPIN_L_%d_A_%g_w_%g_mu0_%g_num_%d_tf_%g_dt_%g_corr.npy"%(L,A,w,mu0,1,tf,dt))
print(min(abs(m_energy)))
plt.plot(T,(EE_q - m_energy)/m_energy)
plt.title('Error in energy dt = %g'%dt)
plt.xlabel('time')
plt.ylabel('E_q - E_ser')
plt.savefig('WDIR/rel_energy_error_dt_%g_.png'%(dt))
plt.show()

for k in range(L):
	plt.plot(T,(cc_q[:,k] - m_corr[:,k])/m_corr[:,k],label='%d'%k)
	plt.xlabel('time')
	#plt.ylabel('E_q - E_ser')
plt.title('Error in correlators dt = %g'%dt)
plt.legend()
plt.savefig('WDIR/rel_corr_error_dt_%g_.png'%(dt))
plt.show()

end_time = time.time()
print("Simulation Done. Time taken : ",(end_time - start_time)," seconds")

# print(repr(m_energy[-1]))
# enrgy_correlator_1 = np.zeros(nT)
# enrgy_correlator_2 = np.zeros(nT)
# for i in range(nT):
# 	enrgy_correlator_1[i] = np.sum(np.multiply(cc_q[i,:],mu_array))
# 	enrgy_correlator_2[i] = np.sum(np.multiply(m_corr[i,:],mu_array))

# plt.plot(T,(enrgy_correlator_1 - enrgy_correlator_2))
# plt.show()
