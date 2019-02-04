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

L = 10
J = 1.0
mu0 = 5.0
A = 0.001
w = 0.5
dt = 0.001
tf = 100.0
nT = int(tf/dt)	
cc0_ser = np.load("WDIR/CORR_L_%d_A_%g_w_%g_mu0_%g_num_%d_tf_%g_dt_%g_corr.npy"%(L,A,w,mu0,1,tf,dt))

# drive protocol
def drive(t,A,w):
	return  np.exp(-1j*A*np.cos(w*t))

def drive_hc(t,A,w):
	return np.exp(1j*A*np.cos(w*t))
# drive protocol parameters
drive_args=[A,w]
##### construct single-particle Hamiltonian #####
mu_array = np.load("WDIR/CORR_L_%d_A_%g_w_%g_mu0_%g_num_%d_tf_%g_dt_%g_mu.npy"%(L,A,w,mu0,1,tf,dt))

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
# E,V=H.eigsh(time=0.0,k=1,which='SA')
#build correlator 00
cdc = [[1.0,0,0]]
static_corr = [["+-",cdc]]
cc0 = hamiltonian(static_corr,[],basis=basis,dtype=np.float64)
#Nbar operator
static_nb = [["+-",[[1.0,i,i] for i in range(L)]]]
NBAR_q = hamiltonian(static_nb,[],basis=basis,dtype=np.float64)

psi0 = np.zeros(basis_dim,dtype=complex)
psi0[0] = 1.0 #Initial state

phi0 = np.zeros(basis_dim,dtype=complex)
phi0[-1] = 1.0 #Initial state
phi1 = np.zeros(basis_dim,dtype=complex)
phi1[0] = 1.0 #Initial state
t = np.linspace(0.0,tf,nT)
EE = np.zeros(nT)
EE_id = np.zeros(nT)
cc0_t = np.zeros(nT)
nb_q = np.zeros(nT)
# spectrum_qspin = np.zeros((nT,L))
# for j in range(L):
# 	for k in range(L):
# 	    pass# print(repr(H.toarray(time=0.0)[j,k]))
psi_t = H.evolve(psi0,t[0],t,iterate=True,atol=1E-12,rtol=1E-12)
print('calculated psi(t)')
for i,psi in enumerate(psi_t):
	EE[i]=H.matrix_ele(psi,psi,time=t[i]).real
	cc0_t[i]=cc0.matrix_ele(psi,psi,time=t[i]).real
	if i%100==0:
		print(i)
		# print(i,repr(cc0_t[i]),repr(cc0_ser[i]),repr(NBAR_q.matrix_ele(psi,psi,time=t[i]).real))
    # spectrum_qspin[i,:] = H.eigvalsh(time=t[i])
    # nb_q[i] = NBAR_q.matrix_ele(psi,psi,time=t[i]).real
    # print(repr(np.linalg.eigvalsh(H.toarray(time=t[i]))[-1]))
# for i in range(nT):
#     EE[i] = H.eigvalsh(time=t[i])[0]
#     EE_id[i] = min([-(J*(np.exp(1j*A*np.cos(w*t[i]))*np.exp(1j*2*np.pi*m/L)) +  \
#                np.conj(J)*(np.exp(-1j*A*np.cos(w*t[i]))*np.exp(-1j*2*np.pi*m/L))) + mu0 for m in range(L)]).real

# np.save("WDIR/QSPIN_L_%d_A_%g_w_%g_mu0_%g_num_%d_tf_%g_dt_%g_energy.npy"%(L,A,w,mu0,1,tf,dt),EE)

EE_CORR = np.load("WDIR/CORR_L_%d_A_%g_w_%g_mu0_%g_num_%d_tf_%g_dt_%g_energy.npy"%(L,A,w,mu0,1,tf,dt))

# spectrum_ser = np.load("WDIR/CORR_L_%d_A_%g_w_%g_mu0_%g_num_%d_tf_%g_dt_%g_spectrum.npy"%(L,A,w,mu0,1,tf,dt))

# print('-------------------------------')
# if np.allclose(EE,EE_CORR):
# 	print('Quspin Energy check PASSED !!')
# else:
# 	print('Quspin Energy check FAILED !!')
# print()
# if np.allclose(cc0_t,cc0_ser):
# 	print('Quspin CC0 check PASSED !!')
# else:
# 	print('Quspin CC0 check FAILED !!')
# print()
# if np.allclose(spectrum_ser,spectrum_qspin):
# 	print('Quspin spectrum check PASSED !!')
# else:
# 	print('Quspin spectrum check FAILED !!')
# print('-------------------------------')

# plt.plot(t,cc0_t,label='quspin')
# plt.plot(t,cc0_ser,label='serial')
# plt.legend()
plt.plot(t,np.abs(EE-EE_CORR))
plt.title('L = %d , w = %g , A = %g'%(L,w,A))
plt.xlabel('time in seconds')
plt.ylabel('|E_qspin - E|')
#plt.legend()
plt.show()
# print('Analytical : ',EE_id[0])
# print('Quspin : ',EE[0])
# print('Correlation : ',EE_CORR[0])
# plt.plot(t,np.abs(cc0_t-cc0_ser))
plt.plot(t,np.abs(cc0_t-cc0_ser))
plt.title('L = %d , w = %g , A = %g'%(L,w,A))
plt.xlabel('time in seconds')

# plt.plot(t,0.001*np.cos(w*t))
plt.show()