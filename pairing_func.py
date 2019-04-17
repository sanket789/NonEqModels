import numpy as np 
import matplotlib.pyplot as plt
import math
def alt_initial_state(L):
	C = np.zeros((L,L),dtype=complex)
	for j in range(L//2):
		C[2*j,2*j] = 1.0
	F = np.zeros((L,L),dtype=complex)

	return C,F


def ground_initial_state(L,H):
	'''
	Construct the G matrix such that the state is ground state of hamiltonian H
	'''
	eigval,eigvec = np.linalg.eigh(H)
	C = np.zeros((L,L),dtype=complex)
	for j in range(L//2):	
		C[j,j] = 1.
	F = np.zeros((L,L),dtype=complex)
	G_0 = construct_G(C,F)
	eigval = np.concatenate((eigval[L:2*L],np.flip(eigval[0:L])))
	z = 0
	for i in range(2*L):
		if abs(eigval[i]) < 1E-8:
			z = z+1
	if z>0:
		print('-------------------------------------------------------------------')
		print('WARNING!!! : The Hamiltonian has zero eigenvalues - WIP')
		print('Please choose different set of parameters to avoid this issue.')
		print('Terminating the simulation')
		print('-------------------------------------------------------------------')
		quit()
	gd = eigvec[0:L,L:2*L] #positive eigenvalue vectors
	hd = eigvec[L:2*L,L:2*L]
	Td = np.hstack((np.vstack((gd,hd)),np.vstack((hd.conj(),gd.conj()))))
	return np.dot(Td,np.dot(G_0,Td.T.conj()))


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
	z = 0
	for i in range(2*L):
		if abs(eigval[i]) < 1E-8:
			z = z+1
	if z>0:
		print('-------------------------------------------------------------------')
		print('WARNING!!! : The Hamiltonian has zero eigenvalues - WIP')
		print('Please choose different set of parameters to avoid this issue.')
		print('Terminating the simulation')
		print('-------------------------------------------------------------------')
		quit()
	gd = eigvec[0:L,L:2*L] #positive eigenvalue vectors
	hd = eigvec[L:2*L,L:2*L]
	Td = np.hstack((np.vstack((gd,hd)),np.vstack((hd.conj(),gd.conj()))))
	A = np.dot(Td,np.dot(np.diag(np.exp(-2j*dt*eigval)),Td.T.conj()))
	B = A.T.conj() #np.dot(Td,np.dot(np.diag(np.exp(2j*dt*eigval)),Td.T.conj()))
	#print(np.allclose(A,B.T.conj()))
	return A,B

def construct_G(C,F):
	L = C.shape[0]
	G = np.hstack((np.vstack((np.eye(L) - C.conj(),F)),np.vstack((-F.conj(),C)))) #My result
	return G

def getEnergy(H,G):
	L = np.shape(H)[0]//2
	a = np.sum(np.multiply(H[0:L,0:L],G[L:2*L,L:2*L]))
	b = np.sum(np.multiply(H[0:L,L:2*L],G[L:2*L,0:L]))
	c = np.sum(np.multiply(H[L:2*L,0:L],G[0:L,L:2*L]))
	d = np.sum(np.multiply(H[L:2*L,L:2*L],G[0:L,0:L]))
	return (a+b+c+d).real - np.sum(np.diag(H[0:L,0:L])).real

def xlogx(x):
	try:
		val = x*math.log(x) 
	except ValueError as ve:
		val = 0.
	return val

def pairing_entropy(G,l):
	L = G.shape[0]//2
	G_l = np.hstack((np.vstack((G[0:l,0:l],G[L:L+l,0:l])),np.vstack((G[0:l,L:L+l],G[L:L+l,L:L+l]))))
	eig = np.linalg.eigvalsh(G_l)
	Sl = sum([-xlogx(p) for p in list(eig)])

	return Sl
