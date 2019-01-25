import numpy as np
import methods as f

def is_unitary(AA):
	size = np.shape(AA)[0]
	return np.allclose(np.dot(np.conj(AA.T),AA),np.eye(size))

L = 4

HH = np.diag(-1.*np.ones(L-1,dtype=complex),1) + np.diag(-1.*np.ones(L-1,dtype=complex),-1)
HH[-1,0] = -1.
HH[0,-1] = -1.

#print('Hermitian Hamiltonian = ',f.is_hermitian(HH))

eps0,DD0 = np.linalg.eigh(HH)
#print('Unitary DD0 = ',is_unitary(DD0))

indx = np.zeros(L)
for i in range (L//2):
	indx[2*i] = i
	indx[2*i+1] = -1-i

eps1 = [-2*np.cos(2*np.pi*n/L) for n in indx]
DD1 = np.zeros((L,L),dtype=complex)
for m in range(L):
	for n in range(L):
		DD1[m,n] = (1/np.sqrt(L))*np.exp(1j*2*np.pi*indx[n]*m/L)

#print('Unitary DD1 = ',is_unitary(DD1))
#print('Eigenvalues are same = ',np.allclose(eps1,eps0))
#print('DD matrices are same = ',np.allclose(DD0,DD1))

##print(np.diag(np.dot(np.conj(DD0.T),np.dot(HH,DD0)).real))
##print(np.diag(np.dot(np.conj(DD1.T),np.dot(HH,DD1)).real))

CC0 = np.dot(np.conj(DD0)[:,0:L//2],DD0.T[0:L//2,:])
CC1 = np.zeros((L,L),dtype=complex)
CC2 = np.dot(np.conj(DD1)[:,0:L//2],DD1.T[0:L//2,:])
for p in range(L):
	for q in range(L):
		CC1[p,q] = (1./L)*sum([np.exp(-1j*2*np.pi*indx[n]*(p-q)/L) for n in range(L//2)])

#print('CC0 is Hermitian = ',f.is_hermitian(CC0))
#print('CC1 is Hermitian = ',f.is_hermitian(CC1))
#print(np.allclose(np.diag(CC0),np.diag(CC1)))
#print(np.diag(CC0),np.diag(CC2))
m_deln = np.zeros(L,dtype=complex)
for m in range(L):
	if eps0[m] < 0:
		m_deln[m] = np.dot(DD0.T,np.dot(CC0,np.conj(DD0)))[m,m]
		print(1)
	else:
		m_deln[m] = np.dot(DD0.T,np.dot(CC0,np.conj(DD0)))[m,m] 
		print(2)
print(m_deln.real)
print(np.dot(DD0.T,np.dot(CC0,np.conj(DD0))).real)