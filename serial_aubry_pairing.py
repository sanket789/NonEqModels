from __future__ import print_function, division
import numpy as np 
import matplotlib.pyplot as plt
def alt_initial_state(L):
	C = np.zeros((L,L),dtype=complex)
	for j in range(L//2):
		C[2*j,2*j] = 1.0
	F = np.zeros((L,L),dtype=complex)

	return C,F


def ham(L,J,delta,mu_array):
	h11 = np.diag(-J*np.ones(L-1,dtype=complex),1) + np.diag(-np.conjugate(J)*np.ones(L-1,dtype=complex),-1) + np.diag(mu_array)
	h11[-1,0] = -J
	h11[0,-1] = -np.conjugate(J)
	h22 = -h11.conj()

	h12 = np.diag(0.5*delta*np.ones(L-1,dtype=complex),1) + np.diag(-0.5*np.conjugate(delta)*np.ones(L-1,dtype=complex),-1) 
	h12[-1,0] = 0.5*delta
	h12[0,-1] = -np.conjugate(0.5*delta)
	h21 = -h12.conj()

	H = np.hstack((np.vstack((h11,h21)),np.vstack((h12,h22))))

	return H

def dynamics(H,dt):
	L = H.shape[0]//2
	eigval, eigvec = np.linalg.eigh(H)
	eigval = np.concatenate((eigval[L:2*L],np.flip(eigval[0:L])))
	gd = eigvec[0:L,L:2*L] #positive eigenvalue vectors
	hd = eigvec[L:2*L,L:2*L]
	Td = np.hstack((np.vstack((gd,hd)),np.vstack((hd.conj(),gd.conj()))))

	A = np.dot(Td,np.dot(np.diag(np.exp(-1j*dt*eigval)),Td.T.conj()))
	B = np.dot(Td,np.dot(np.diag(np.exp(1j*dt*eigval)),Td.T.conj()))
	return A,B

def construct_G(C,F):
	G = np.hstack((np.vstack((np.eye(L) - C.conj(),F)),np.vstack((-F.conj(),C)))) #My result
	# G = np.hstack((np.vstack((np.eye(L) - C,F.conj())),np.vstack((F,C))))
	return G
'''
	Simulation parameters
'''

L = 4
J = 1.0
mu0 = 3.0
dJ = 0.0*J
delta = 53.
T = 1.7
cyc = 2
nT = int(2*cyc)
mu_array = np.load("ED/ED_mu.npy")
tlist = 0.5*T*np.arange(2*cyc)

#data storing variables
m_CC = np.zeros((nT,L,L),dtype=complex)
m_FF = np.zeros((nT,L,L),dtype=complex)
m_GG = np.zeros((nT,2*L,2*L),dtype=complex)
m_nb = np.zeros(nT)

#Hamiltonian 1
HH_h = ham(L,J+dJ,delta,mu_array)
UU_h , VV_h = dynamics(HH_h,0.5*T)
#Hamiltonian 2
HH_l = ham(L,J-dJ,delta,mu_array)
UU_l , VV_l = dynamics(HH_l,0.5*T)
#Intial state
CC_0 , FF_0 = alt_initial_state(L)
GG_t = construct_G(CC_0,FF_0)
'''
	Main loop
'''

for i in range(nT):
	m_GG[i,:,:] = GG_t.copy()
	m_CC[i,:,:] = GG_t[L:2*L,L:2*L]
	m_FF[i,:,:] = GG_t[L:2*L,0:L]
	m_nb[i] = np.sum(np.diag(m_CC[i,:,:]).real)
	# print(m_nb[i])
	if i%2 == 0: #Hamiltonian 1
		GG_next = np.dot(UU_h,np.dot(GG_t,VV_h))
	else: #Hamiltonian 2
		GG_next = np.dot(UU_l,np.dot(GG_t,VV_l))
	GG_t = GG_next.copy()


ED_GG = np.load("ED/ED_GG.npy")
ED_CC_old = np.load("ED/ED_CC_old.npy")
# print('All :',np.allclose(ED_GG[:,L,L],m_GG[:,L,L]))
print('CC All:' ,np.allclose(ED_GG[:,L:2*L,L:2*L],m_GG[:,L:2*L,L:2*L]))
print('FF All:' ,np.allclose(ED_GG[:,L:2*L,0:L],m_GG[:,L:2*L,0:L]))
print('All Initial :' ,np.allclose(ED_GG[0,:,:],m_GG[0,:,:]))
print(np.allclose(ED_GG[:,0:L,L:2*L],-ED_GG[:,L:2*L,0:L].conj()))
# for i in range(nT):
# 	print(np.sum(np.diag(m_CC[i,:,:]).real),np.sum(np.diag(ED_GG[i,L:2*L,L:2*L]).real))
# plt.plot(tlist,ED_GG[:,L+1,L+1].real,label='ED-new')
# plt.plot(tlist,m_CC[:,1,1].real,label='my')
# plt.plot(tlist,ED_CC_old[:,1,1].real,label='ED - old')
# plt.legend()
# plt.show()
plt.plot(tlist,m_nb)
plt.show()
# plt.plot(tlist,ED_GG[:,L,L].real,'--')
# plt.plot(tlist,m_GG[:,1,1].real+m_GG[:,L+1,L+1].real)
# plt.show()
