from __future__ import print_function, division
import numpy as np
import sys,os
from quspin.operators import hamiltonian # Hamiltonians and operators
from quspin.basis import spinless_fermion_basis_1d # Hilbert space fermion basis
from quspin.tools.evolution import evolve # nonlinear evolution 
import numpy as np # generic math functions
import matplotlib.pyplot as plt # plotting library
from six import iteritems # loop over elements of dictionary
import math
'''
	This script is for time evolution of Anderson insulator using QuSpin method. 
	https://doi.org/10.1103/PhysRevB.98.214202	
	The reults of this test code has to compared with time evolution using quspin package
	The Anderson insulator is driven such that the hopping apmlitude switches between (J+dJ) and (J-dJ) with period T.
'''
def fact(n):
	if n < 1:
		return 1
	else:
		return n*fact(n-1)
print('------------------------------------')
print('Anderson Insultor using QuSpin')
print('------------------------------------')
L = 10
J = 1.0
mu0 = 5.0
dJ = 0.2*J
T = 1.
tf = 100.0
dt = 0.001
nT = int(tf/dt)

fname = "WDIR/Quspin_SQ_L_%d_T_%g_mu0_%g_tf_%g_dJ_%g_"%(L,T,mu0,tf,dJ)
'''
	##############	Define the Hamiltonian 	###########################
'''
def drive(t,J,dJ):
	if math.fmod(t,T) <0.5*T:
		return J+dJ
	else:
		return J-dJ
# drive protocol parameters
drive_args=[J,dJ]
##### construct single-particle Hamiltonian #####
mu_array = np.load("Master_%g_mu.npy"%(mu0))
# define site-coupling lists
hopping=[[-1,i,(i+1)%L] for i in range(L)]
hopping_hc =[[-1,(i+1)%L,i] for i in range(L)]
chem=[[mu_array[i],i,i] for i in range(L)]
# define static and dynamic lists
dynamic = [["+-",hopping,drive,drive_args],["+-",hopping_hc,drive,drive_args]]
static = [["+-",chem]]
# define basis
basis = spinless_fermion_basis_1d(L=L,Nf=L//2)
basis_dim = int((fact(L))/(fact(L//2))**2)
# build Hamiltonian
H=hamiltonian(static,dynamic,basis=basis,dtype=np.float64)
'''
	################	Initial state 	#######################
'''
psi0 = np.zeros(basis_dim)
psi0[0] = 1.0 #Initial state
hopping_im = [[-(J+dJ),i,(i+1)%L] for i in range(L)]
hopping_hc_im = [[-(J+dJ),(i+1)%L,i] for i in range(L)]
chem_im = [[mu_array[i],i,i] for i in range(L)]
# define static and dynamic lists
static_im = [["+-",hopping_im],["+-",hopping_hc_im],["+-",chem_im]]
H_im=hamiltonian(static_im,[],basis=basis,dtype=np.float64)
tau = np.linspace(0.0,70.0,100)
psi_imag_t = H_im.evolve(psi0,tau[0],tau,iterate=True,atol=1E-12,rtol=1E-12,imag_time=True)
for k,psi_ground in enumerate(psi_imag_t):
	pass 
'''
	################	Real time evolve 	#########################
'''
#time vector
t = np.linspace(0.0,tf,nT,endpoint=False)
EE_q = []

#Time evolution under the hamiltonian
psi_t = H.evolve(psi_ground,t[0],t,iterate=True,atol=1E-12,rtol=1E-12)
print('calculated psi(t)')
tlist = np.linspace(0,tf,2*int(tf/T),endpoint=False)
print (150*dt in tlist,150*dt,tlist[3])
# print(tlist)
for i,psi in enumerate(psi_t):

	# if i%100==0:
	# 	print('t = %g'%(i*dt))
	if i*dt in tlist:#np.allclose(math.fmod(i*dt,T),0) or np.allclose(math.fmod(i*dt,T),0.5*T):
		energy=H.matrix_ele(psi,psi,time=t[i]).real
		EE_q.append(energy)
		print(i*dt,t[i])

np.save(fname+"energy.npy",EE_q)
np.save("Master_%g_mu.npy"%(mu0),mu_array)

print("-------------------------")
print('Files saved at : ',fname)
print("QuSpin Simulation Done!!")
print("-------------------------")
print(repr(EE_q[0]))
plt.plot(tlist,EE_q)
plt.show()