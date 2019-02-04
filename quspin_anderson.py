from __future__ import print_function, division
import numpy as np
import sys,os
from quspin.operators import hamiltonian # Hamiltonians and operators
from quspin.basis import spinless_fermion_basis_1d # Hilbert space fermion basis
from quspin.tools.evolution import evolve # nonlinear evolution 
import numpy as np # generic math functions
import matplotlib.pyplot as plt # plotting library
from six import iteritems # loop over elements of dictionary
#####################################################################
#                            example 4                              #
#    In this script we demonstrate time evolution of anderson       #
#    insulator using quspin package							        #
#####################################################################
##### define model parameters #####
def fact(n):
	if n < 1:
		return 1
	else:
		return n*fact(n-1)
print('----------------------')
print('QuSPin')
print('----------------------')
L = 10
J = 1.0
mu0 = 5.0
A = 1.
w = 0.5
dt = 0.001
tf = 10.0
nT = int(tf/dt)	

fname = "WDIR/QUSPIN_L_%d_A_%g_w_%g_mu0_%g_num_%d_tf_%g_dt_%g_"%(L,A,w,mu0,1,tf,dt)
# drive protocol
def drive(t,A,w):
	return  np.exp(-1j*A*np.cos(w*t))

def drive_hc(t,A,w):
	return np.exp(1j*A*np.cos(w*t))
# drive protocol parameters
drive_args=[A,w]
##### construct single-particle Hamiltonian #####
mu_array = mu0*np.ones(L)#np.random.uniform(-1*mu0,mu0,L)

# define site-coupling lists
hopping=[[-J,i,(i+1)%L] for i in range(L)]
hopping_hc =[[-J,(i+1)%L,i] for i in range(L)]
chem=[[mu_array[i],i,i] for i in range(L)]
# define static and dynamic lists
static = [["+-",chem]]
dynamic = [["+-",hopping,drive,drive_args],["+-",hopping_hc,drive_hc,drive_args]]
# define basis
basis = spinless_fermion_basis_1d(L=L,Nf=L//2)
basis_dim = int((fact(L))/(fact(L//2))**2)

# build Hamiltonian
H=hamiltonian(static,dynamic,basis=basis,dtype=np.float64)
#build operator for correlator matrix diagonal
static_corr = []
cc_op = []
for n in range(L):
	static_corr.append([["+-",[[1.0,n,n]]]])	#cn_dagger_cn coupling
	cc_op.append(hamiltonian(static_corr[-1],[],basis=basis,dtype=np.float64))

#Nbar operator
static_nb = [["+-",[[1.0,i,i] for i in range(L)]]]
NBAR_q = hamiltonian(static_nb,[],basis=basis,dtype=np.float64)
#Initial state 
psi0 = np.zeros(basis_dim,dtype=complex)
psi0[0] = 1.0 #Initial state
#time vector
t = np.linspace(0.0,tf,nT)
EE_q = np.zeros(nT)
nb_q = np.zeros(nT)
cc_q = np.zeros((nT,L))
#Time evolution under the hamiltonian
psi_t = H.evolve(psi0,t[0],t,iterate=True,atol=1E-12,rtol=1E-12)
print('calculated psi(t)')
for i,psi in enumerate(psi_t):
	EE_q[i]=H.matrix_ele(psi,psi,time=t[i]).real
	for n in range(L):
		cc_q[i,n]=cc_op[n].matrix_ele(psi,psi,time=t[i]).real
	if i%100==0:
		print('t = %g'%(i*dt))
np.save(fname+"nbar.npy",NBAR_q)
np.save(fname+"energy.npy",EE_q)
np.save(fname+"mu.npy",mu_array)
np.save(fname+"corr.npy",cc_q)
# np.save('Dual-Degree-Project/NonEqModels/v_new_quspin.npy',EE_q)
print("-------------------------")
print('Files saved at : ',fname)
print("QuSpin Simulation Done!!")
print("-------------------------")
print(repr(EE_q[-1]))
