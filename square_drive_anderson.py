from __future__ import print_function, division

import numpy as np
import methods as func
import time
import matplotlib.pyplot as plt
import corr_init as cstart
from time_evolution import evolve_correlation_matrix

'''
	This script is for time evolution of Anderson insulator using correlators method. 
	The code is written in serial execution manner for single configuration.  
	https://doi.org/10.1103/PhysRevB.98.214202	
	The reults of this test code has to compared with time evolution using quspin package
	The Anderson insulator is driven such that the hopping apmlitude switches between (J+dJ) and (J-dJ) with period T.
'''
L = 4	#system size
J = 1.
delta = 0.0
dJ = 0.1
T = 1.0
cyc = 4

nT = 2*cyc
save_data = False

# CC_0 = np.zeros((L,L),dtype=complex)
# CC_0[0,0] = 1.0
# CC_0[1,1] = 1.0
# CC_0[2,2] = 1.0
# CC_0 = cstart.left_spatial_occ(L)#cstart.alternate_occupy(L)
# CC_t = CC_0.copy()

'''
	#########################################  Main Code  ##############################################
'''
mu_array = np.load('ED/ED_mu.npy')
# np.save("WDIR/test/mu_0.npy",mu_array)
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
# HH_start = - (J+dJ)*np.diag(np.ones(L-1),1)- (J+dJ)*np.diag(np.ones(L-1),-1)
# HH_start[-1,0] = -(J+dJ)
# HH_start[0,-1] = -(J+dJ) 
# print(HH_h)
# print(HH_l)
CC_0 = cstart.alternate_occupy(L)#ground_state(num_p,HH_h)
# print('ground state energy = ',np.sum(np.multiply(HH_h,CC_0)).real)
CC_t = CC_0.copy()
#Initialize the storage matrices
m_energy = np.zeros(nT)
m_nbar = np.zeros(nT)
m_corr = np.zeros((nT,L))
m_spectrum = np.zeros((nT,L))
occ_indx = np.arange(L/2)-L//4
m_exact_energy = np.zeros(nT)
ham = np.zeros(nT)
m_nsub = np.zeros(nT)
m_cc0 = np.zeros(nT)
m_CC = np.zeros((nT,L,L),dtype=complex)
#extract Hamiltonian paramter from quspin simulation

'''
	#########################################  Loop over Time  ##############################################
'''
for i in range(nT):
	#Calculate Hamiltonian
	m_CC[i,:] = CC_t.copy()		 
	if i%2 == 0:
		m_energy[i] = np.sum(np.multiply(HH_h,CC_t)).real
		m_nbar[i] = np.sum(np.diag(CC_t)[0:L]).real/L
		m_nsub[i] = np.sum(np.diag(CC_t)[0:3]).real
		m_cc0[i] = CC_t[1,1].real
		CC_next = np.dot(np.conj(UU_h.T),np.dot(CC_t,UU_h))
		# m_exact_energy[i] = sum([-2*(J+dJ)*np.cos(2*np.pi*m/L)+mu0 for m in occ_indx]).real
	else:
		m_energy[i] = np.sum(np.multiply(HH_l,CC_t)).real
		m_nbar[i] = np.sum(np.diag(CC_t)[0:L]).real/L
		m_nsub[i] = np.sum(np.diag(CC_t)[0:3]).real
		m_cc0[i] = CC_t[1,1].real
		CC_next = np.dot(np.conj(UU_l.T),np.dot(CC_t,UU_l))
		# m_exact_energy[i] = sum([-2*(J-dJ)*np.cos(2*np.pi*m/L)+mu0 for m in occ_indx]).real
	CC_t = CC_next.copy()
print(m_nbar[::2])
print(m_energy[::2])

ED_GG = np.load('ED/ED_GG.npy')
ED_CC = ED_GG[:,L:2*L,L:2*L]
print('Initial',np.allclose(ED_CC[0,:],m_CC[0,:]))
print('All',np.allclose(ED_CC,m_CC[::2,:,:]))
plt.plot(ED_CC[:,0,0].real,'--')
plt.plot(m_CC[::2,0,0].real)
plt.show()


np.save('ED/and_CC.npy',m_CC)
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
# ED_nbar = np.load('ED/ED_nbar.npy')
# ED_nsub = np.load('ED/ED_nsub.npy')
# ED_cc0 = np.load('ED/ED_cc0.npy')
# ED_energy = np.load('ED/ED_energy.npy')

# plt.plot(range(cyc),ED_energy,'--',label='ED')
# plt.plot(range(cyc),m_energy[::2],label='CORR')
# plt.legend()
# plt.xlabel('t/T')
# plt.title('energy')
# plt.savefig('ED/energy_L_%d.png'%L)
# plt.show()

# plt.plot(range(cyc),ED_nbar,'--',label='ED')
# plt.plot(range(cyc),m_nbar[::2],label='CORR')
# plt.legend()
# plt.xlabel('t/T')
# plt.title('nbar')
# plt.show()

# plt.plot(range(100),ED_nsub[0:100],'--',label='ED')
# plt.plot(range(100),m_nsub[::2][0:100],label='CORR')
# plt.legend()
# plt.xlabel('t/T')
# plt.title('Subsystem of size 3')
# plt.savefig('ED/nsub_L_%d.png'%L)
# plt.show()

# plt.plot(range(cyc),ED_cc0,'--',label='ED')
# plt.plot(range(cyc),m_cc0[::2],label='CORR')
# plt.legend()
# plt.xlabel('t/T')
# plt.title('cc0')
# plt.savefig('ED/cc0_L_%d.png'%L)
# plt.show()

# plt.semilogy(range(cyc),np.divide(np.abs(ED_nsub - m_nsub[::2]),ED_nsub))
# plt.xlabel('t/T')
# plt.title('Relative error in subsystem occupation')
# plt.savefig('ED/nsubError_L_%d.png'%L)
# plt.show()