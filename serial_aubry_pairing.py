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
	h11 = np.diag(-0.5*J*np.ones(L-1,dtype=complex),1) + np.diag(-np.conjugate(0.5*J)*np.ones(L-1,dtype=complex),-1) \
				+ np.diag(0.5*mu_array)
	h11[-1,0] = -0.5*J
	h11[0,-1] = -np.conjugate(0.5*J)
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
	# print(eigval)
	
	z = 0
	for i in range(2*L):
		if abs(eigval[i]) < 1E-8:
			z = z+1
	if z==0:
		gd = eigvec[0:L,L:2*L] #positive eigenvalue vectors.
		hd = eigvec[L:2*L,L:2*L]
	else:	
		gd = np.zeros((L,L),dtype=complex)
		hd = np.zeros((L,L),dtype=complex)
		j = 1
		gd[:,0] = eigvec[0:L,L+z//2-1]
		hd[:,0] = eigvec[L:2*L,L+z//2-1]
		for i in range(1,z):
			if j < z//2:
				print(i,j,z//2)	
				gdv = eigvec[0:L,L+z//2-1-i]
				hdv = eigvec[L:2*L,L+z//2-1-i]
				flag = 0
				for k in range(i):
					if np.array_equal(gdv.conj() , hd[:,k]):
						flag += 1
					else: 
						flag += 0 
				print(flag)
				if flag==0:
					gd[:,j] = gdv.copy()
					hd[:,j] = hdv.copy()
					j = j+1
		gd[:,z//2:] =  eigvec[0:L,L+z//2:]
		hd[:,z//2:] =  eigvec[L:2*L,L+z//2:]


	eigval = np.concatenate((eigval[L:2*L],np.flip(eigval[0:L])))
	# print(eigval)
	# print(np.dot(uu.conj().T,uu).real)
	print(gd[:,:].real,'\n',hd[:,:].real)
	print(eigvec.real)
	# print(np.allclose(np.eye(2*L),np.dot(eigvec,eigvec.T.conj())),'unitary eig' )
	Td = np.hstack((np.vstack((gd,hd)),np.vstack((hd.conj(),gd.conj()))))
	print(np.allclose(np.eye(2*L),np.dot(Td,Td.T.conj())),'unitary T' )
	A = np.dot(Td,np.dot(np.diag(np.exp(-2j*dt*eigval)),Td.T.conj()))
	B = np.dot(Td,np.dot(np.diag(np.exp(2j*dt*eigval)),Td.T.conj()))
	# print(np.dot(Td.T.conj(),Td).real)
	return A,B

def construct_G(C,F):
	G = np.hstack((np.vstack((np.eye(L) - C.conj(),F)),np.vstack((-F.conj(),C)))) #My result
	# G = np.hstack((np.vstack((np.eye(L) - C,F.conj())),np.vstack((F,C))))
	return G

def getEnergy(H,G):
	L = np.shape(H)[0]//2
	a = np.sum(np.multiply(H[0:L,0:L],G[L:2*L,L:2*L]))
	b = np.sum(np.multiply(H[0:L,L:2*L],G[L:2*L,0:L]))
	c = np.sum(np.multiply(H[L:2*L,0:L],G[0:L,L:2*L]))
	d = np.sum(np.multiply(H[L:2*L,L:2*L],G[0:L,0:L]))
	return (a+b+c+d).real - np.sum(np.diag(H[0:L,0:L])).real
'''
	Simulation parameters
'''
L = 4	#system size
J = 1.
delta = 0.
dJ = 0.1
T = 1.25
cyc = 10

nT = 2*cyc

mu_array = np.zeros(L)#np.load("ED/ED_mu.npy")
ED_GG = np.load("ED/ED_GG.npy")
tlist = 0.5*T*np.arange(2*cyc)

#data storing variables
m_CC = np.zeros((nT,L,L),dtype=complex)
m_FF = np.zeros((nT,L,L),dtype=complex)
m_GG = np.zeros((nT,2*L,2*L),dtype=complex)
m_nb = np.zeros(nT)
m_energy = np.zeros(nT)
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
# GG_t = ED_GG[0,:,:]
for i in range(nT):
	m_GG[i,:,:] = GG_t.copy()
	m_CC[i,:,:] = GG_t[L:2*L,L:2*L]
	m_FF[i,:,:] = GG_t[L:2*L,0:L]
	m_nb[i] = np.sum(np.diag(m_CC[i,:,:]).real)
	m_energy[i] = getEnergy(HH_h,m_GG[i,:,:])
	# print(m_nb[i])
	if i%2 == 0: #Hamiltonian 1
		GG_next = np.dot(UU_h,np.dot(GG_t,VV_h))
	else: #Hamiltonian 2
		GG_next = np.dot(UU_l,np.dot(GG_t,VV_l))
	GG_t = GG_next.copy()
	# print(max(GG_t.ravel()))

ED_GG = np.load("ED/ED_GG.npy")
# and_CC = np.load("ED/and_CC.npy")
# # ED_CC_old = np.load("ED/ED_CC_old.npy")
# print('All :',np.allclose(ED_GG[:,L,L],m_GG[:,L,L]))
# print('-------------------------------------------')
print('CC All:' ,np.allclose(ED_GG[:,L:2*L,L:2*L],m_GG[::2,L:2*L,L:2*L]))
print('FF All:' ,np.allclose(ED_GG[:,L:2*L,0:L],m_GG[::2,L:2*L,0:L]))
# print('--------------------------------------------')
# # print('cdc00 :' ,np.allclose(ED_GG[:,L,L],m_GG[:,L,L]))
# # print(np.allclose(m_GG[0,:,:],ED_GG[0,:,:]))
# print('nbar',m_nb[::2])
# print('energy',m_energy[::2])
# print(HH_h.real)
# print(m_GG[0,:].real)

# print(np.allclose(and_CC,m_CC),'and')

# for i in range(nT):
# 	print(np.sum(np.diag(m_CC[i,:,:]).real),np.sum(np.diag(ED_GG[i,L:2*L,L:2*L]).real))
# plt.plot(tlist,ED_GG[:,L+1,L+1].real,label='ED-new')
# plt.plot(tlist,m_CC[:,1,1].real,label='my')
# plt.plot(tlist,ED_CC_old[:,1,1].real,label='ED - old')
# plt.legend()
# plt.show()
# plt.plot(tlist,m_GG[:,L,L].real)
# plt.plot(tlist,ED_GG[:,L,L].real,'--')

# plt.show()
# plt.plot(tlist,ED_GG[:,L,L].real,'--')
# plt.plot(tlist,m_GG[:,1,1].real+m_GG[:,L+1,L+1].real)
# plt.show()
# plt.plot(m_energy)
# plt.show()