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
	eigval = np.concatenate((eigval[L:2*L],np.flip(eigval[0:L])))
	gd = eigvec[0:L,L:2*L] #positive eigenvalue vectors
	hd = eigvec[L:2*L,L:2*L]
	Td = np.hstack((np.vstack((gd,hd)),np.vstack((hd.conj(),gd.conj()))))
	# print(np.allclose(np.eye(2*L),np.dot(Td,Td.T.conj())))
	A = np.dot(Td,np.dot(np.diag(np.exp(-2j*dt*eigval)),Td.T.conj()))
	B = np.dot(Td,np.dot(np.diag(np.exp(2j*dt*eigval)),Td.T.conj()))
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
