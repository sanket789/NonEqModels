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
	Exact diagonalization solution of Aubry-Andre model with pairing term
'''
#factorial
def fact(n):
	if n < 1:
		return 1
	else:
		return n*fact(n-1)

def drive(t,T,J,dJ):
	if math.fmod(t,T) <0.5*T:
		return J+dJ
	else:
		return J-dJ

L = 4
J = 1.0
mu0 = 3.0
dJ = 0.0*J
delta = 53.
T = 1.7
cyc = 2
nT = int(2*cyc)
mu_array = np.arange(L)#np.random.uniform(-mu0,mu0,L)

'''
	##############	Define the Hamiltonian and Correlators 	###########################
'''
# drive protocol parameters
drive_args = [T,J,dJ]
# define site-coupling lists
hopping = [[-1,i,(i+1)%L] for i in range(L)]
hopping_hc = [[-1,(i+1)%L,i] for i in range(L)]
chem = [[mu_array[i],i,i] for i in range(L)]
paircDcD = [[delta,i,(i+1)%L] for i in range(L)]
paircc = [[delta,(i+1)%L,i] for i in range(L)]
# define static and dynamic lists
dynamic = [["+-",hopping,drive,drive_args],["+-",hopping_hc,drive,drive_args]]
static = [["+-",chem],["++",paircDcD],["--",paircc]]
# define basis
basis = spinless_fermion_basis_1d(L=L,Nf=range(0,L+1))
basis_dim = 2**L#int((fact(L))/(fact(L//2))**2)
# build Hamiltonian
HH = hamiltonian(static,dynamic,basis=basis,dtype=np.float64,check_pcon=False)
# print(basis)
#Initial state
psi0 = np.zeros(basis_dim,dtype=complex)
# psi0[5] = 1.0
psi0[0] = 1./np.sqrt(2.)
psi0[3] = 1./np.sqrt(2.)
# print(basis[1])
#Evolve the state
tlist = 0.5*T*np.arange(2*cyc)
psi_t = HH.evolve(psi0,tlist[0],tlist,iterate=True,atol=1E-12,rtol=1E-12)
'''
	################## Correlators ##############################
'''
cdc = []
for i in range(L):
	for j in range(L):
		A = hamiltonian([["+-",[[1,i,j]]]],[],basis=basis,dtype=np.float64,check_herm=False)

		cdc.append(A)
ccd = []
for i in range(L):
	for j in range(L):
		A = hamiltonian([["-+",[[1,i,j]]]],[],basis=basis,dtype=np.float64,check_herm=False)
		ccd.append(A)

cdcd = []
for i in range(L):
	for j in range(L):
		A = hamiltonian([["++",[[1,i,j]]]],[],basis=basis,dtype=np.float64,check_herm=False,check_pcon=False)
		cdcd.append((A,i,j))

cc = []
for i in range(L):
	for j in range(L):
		A = hamiltonian([["--",[[1,i,j]]]],[],basis=basis,dtype=np.float64,check_herm=False,check_pcon=False)
		cc.append(A)

'''
	################### Calculation of expectation values
'''
GG = np.zeros((nT,2*L,2*L),dtype=complex) 
energy = np.zeros(nT)
# print(np.shape(GG))
for i,psi in enumerate(psi_t):	
	l = 0
	energy[i] = HH.matrix_ele(psi,psi,time=tlist[i]).real
	for j in range(L):
		for k in range(L):
			# print(i,j,k,l)
			GG[i,j,k] = ccd[l].matrix_ele(psi,psi,time=tlist[i])
			GG[i,j,L+k] = cc[l].matrix_ele(psi,psi,time=tlist[i])
			GG[i,L+j,k] = cdcd[l][0].matrix_ele(psi,psi,time=tlist[i])
			GG[i,L+j,L+k] = cdc[l].matrix_ele(psi,psi,time=tlist[i])
			l = l+1

# plt.plot(tlist,energy)
# plt.show()
np.save("ED/ED_GG.npy",GG)
np.save("ED/ED_mu.npy",mu_array)
energy_old = np.zeros(nT)
nb = np.zeros(nT)
for i in range(nT):
	nb[i] = np.sum(np.diag(GG[i,L:,L:].real))
# print(np.allclose(nb,0.5*L*np.ones(nT)))
# plt.plot(tlist,nb)
# plt.show()
# print(basis)
# print(cdcd[14][1],cdcd[14][2])
# print(cdcd[14][0].toarray())
# print(cdcd)
# print(GG[0,:,:].real)
# m_CC_old = np.zeros((nT,L,L),dtype=complex)
# HH_old = hamiltonian([["+-",chem]],dynamic,basis=basis,dtype=np.float64,check_pcon=False)
# psi_t = HH_old.evolve(psi0,tlist[0],tlist,iterate=True,atol=1E-12,rtol=1E-12)
# for i,psi in enumerate(psi_t):	
# 	l = 0
# 	energy_old[i] = HH_old.matrix_ele(psi,psi,time=tlist[i]).real
# 	for j in range(L):
# 		for k in range(L):
# 			# print(i,j,k,l)
# 			m_CC_old[i,j,k] = cdc[l].matrix_ele(psi,psi,time=tlist[i])
# 			l = l+1
# np.save("ED/ED_CC_old.npy",m_CC_old)

# plt.plot(tlist,energy_old,label='old')
# plt.plot(tlist,energy,label='new')
# plt.show()

# plt.plot(energy - energy_old)
# plt.show()
# print(np.allclose(HH.toarray(),HH_old.toarray()))
# print(HH_old.toarray())
b1 = spinless_fermion_basis_1d(L=L,Nf=range(0,L+1,2))
print(b1)
JJ = [[4.,2,3]]
c_n = hamiltonian([["++",JJ]],[],basis=b1,dtype=np.float64,check_herm=False,check_pcon=False)
# print(c_n.toarray())
c_n = hamiltonian([["++",JJ]],[],basis=b1,dtype=np.float64,check_herm=False,check_pcon=False)
# print(c_n.toarray())