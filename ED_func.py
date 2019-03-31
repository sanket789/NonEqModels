import numpy as np
from scipy import special
import scipy.sparse as sps
import scipy.sparse.linalg as spsla
import matplotlib.pyplot as plt
'''
	Exact diagonalisation routine to calculate the dynamics of tight binding fermion model
'''
def construct_full_basis(L):
	'''
		Construct many body basis for fermions 
		L	:	System size
		num_p	: Number of particles
		returns dictionary with {key : basis state; value : sequence}
	'''
	return np.arange(2**L)

def calc_sign(state,i):
	'''
		Calculate sign corresponding to action of c dagger
		state: Binary representation in list form
		i : index of operator operation
	'''
	count = 0
	for j in state[0:i]:
		if j==1:
			count = count+1
	return (-1)**count

def hop_op(basis,L,J): 
	'''
		Matrix element of hopping term: -J c^+_{i}c_{i+1} + h.c.
		basis 	:	list of basis states (binary)
		L 		: 	System size
		J 		: 	Hopping amplitude
	'''
	#(c_idag c_i+1) + (c_i+1dag c_i)
	#do first (c_idag c_i+1)
	basis_dim = len(basis)
	op = sps.lil_matrix((basis_dim,basis_dim))
	for i in range(L):	#iterate over L sites
		j = np.mod(i+1,L) #nearest neighbour
		for k in basis:#iterate over the basis 
			state_bin = bin(k)[2:].zfill(L)	#binary representation of the state |101110...>
			state_list = [int(x) for x in list(state_bin)] #convert it to list of int form
			if state_list[i] == 0 and state_list[j] == 1:	#if matrix element is finite
				sign_j = calc_sign(state_list,j)
				state_list[j] = 0
				sign_i = calc_sign(state_list,i)
				state_list[i] = 1
				ket_index = basis[int(''.join(str(x) for x in state_list),2)]
				op[basis[ket_index],basis[k]] += -J*sign_i*sign_j
	op = op + op.transpose()#.conjugate()
	return op.tocsc()

def onsite_pot_op(basis,L,mu):
	'''
		Matrix element of onsite potential term:  mu_i c^+_{i}c_{i} 
		basis 	:	list of basis states (binary)
		L 		: 	System size
		mu 		: 	Onsite potential (site-dependent)
	'''
	basis_dim = len(basis)
	op = sps.lil_matrix((basis_dim,basis_dim))
	for i in range(L):	#iterate over L sites
		for k in basis:	#iterate over keys in the basis 
			state_bin = bin(k)[2:].zfill(L)	#binary representation of the state |101110...>
			state_list = [int(x) for x in list(state_bin)] #convert it to list of int form
			if state_list[i]==1:
				op[basis[k],basis[k]] += mu[i]
	return op.tocsc()
 
def pairing_op(basis,L,delta):
	'''
		Matrix element pairing term:  delta c^+_{i}c^+_{i+1} + h.c. 
		basis 	:	list of basis states (binary)
		L 		: 	System size
		delta 		: 	Onsite potential (site-dependent)
	'''
	#Calculate cdag_cdag term and add h.c.
	basis_dim = len(basis)
	op = sps.lil_matrix((basis_dim,basis_dim))
	for i in range(L):	#iterate over L sites
		j = np.mod(i+1,L) #nearest neighbour
		for k in basis:#iterate over the basis 
			state_bin = bin(k)[2:].zfill(L)	#binary representation of the state |101110...>
			state_list = [int(x) for x in list(state_bin)] #convert it to list of int form
			if state_list[i] == 0 and state_list[j] == 0:	#if matrix element is finite
				sign_j = calc_sign(state_list,j)
				state_list[j] = 1
				sign_i = calc_sign(state_list,i)
				state_list[i] = 1
				ket_index = basis[int(''.join(str(x) for x in state_list),2)]
				op[basis[ket_index],basis[k]] += delta*sign_i*sign_j
	op = op + op.transpose()#.conjugate()
	return op.tocsc()

def getUnitary(ham1,ham2,dt):
	'''
		Unitary operator construction
		ham1 is Hamiltonian for t = [0,T/2) and ham2 is for t = [T/2,T)
	'''
	U1 = (spsla.expm(-1j*dt*ham1)).toarray()
	U2 = (spsla.expm(-1j*dt*ham2)).toarray()
	return np.dot(U2,U1)

def cdc_op(basis,L,amp,i,j):
	'''
		c dagger_i c_j operator
	'''
	basis_dim = len(basis)
	op = sps.lil_matrix((basis_dim,basis_dim))
	for k in basis:									#iterate over the basis 
		state_bin = bin(k)[2:].zfill(L)				#binary representation of the state |101110...>
		state_list = [int(x) for x in list(state_bin)] 	#convert it to list of int form
		if i==j:
			if state_list[i]==1:
				op[basis[k],basis[k]] += amp
		elif state_list[i] == 0 and state_list[j] == 1:
			sign_j = calc_sign(state_list,j)
			state_list[j] = 0
			sign_i = calc_sign(state_list,i)
			state_list[i] = 1
			ket_index = basis[int(''.join(str(x) for x in state_list),2)]
			op[basis[ket_index],basis[k]] += amp*sign_i*sign_j
	return op.tocsc()

def ccd_op(basis,L,amp,i,j):
	'''
		c_i c dagger_j operator
	'''
	basis_dim = len(basis)
	op = sps.lil_matrix((basis_dim,basis_dim))
	for k in basis:									#iterate over the basis 
		state_bin = bin(k)[2:].zfill(L)				#binary representation of the state |101110...>
		state_list = [int(x) for x in list(state_bin)] 	#convert it to list of int form
		if i==j:
			if state_list[i]==0:
				op[basis[k],basis[k]] += amp
		elif state_list[i] == 1 and state_list[j] == 0:
			sign_j = calc_sign(state_list,j)
			state_list[j] = 1
			sign_i = calc_sign(state_list,i)
			state_list[i] = 0
			ket_index = basis[int(''.join(str(x) for x in state_list),2)]
			op[basis[ket_index],basis[k]] += amp*sign_i*sign_j
	return op.tocsc()

def cdcd_op(basis,L,amp,i,j):
	'''
		c dagger_i c dagger_j operator
	'''
	basis_dim = len(basis)
	op = sps.lil_matrix((basis_dim,basis_dim))
	if i==j:
		return op.tocsc()
	else:
		for k in basis:									#iterate over the basis 
			state_bin = bin(k)[2:].zfill(L)				#binary representation of the state |101110...>
			state_list = [int(x) for x in list(state_bin)] 	#convert it to list of int form
			if state_list[i] == 0 and state_list[j] == 0 :
				sign_j = calc_sign(state_list,j)
				state_list[j] = 1
				sign_i = calc_sign(state_list,i)
				state_list[i] = 1
				ket_index = basis[int(''.join(str(x) for x in state_list),2)]
				op[basis[ket_index],basis[k]] += amp*sign_i*sign_j
	return op.tocsc()

def cc_op(basis,L,amp,i,j):
	'''
		c_i c_j operator
	'''
	basis_dim = len(basis)
	op = sps.lil_matrix((basis_dim,basis_dim))
	if i==j:
		return op.tocsc()
	else:	
		for k in basis:									#iterate over the basis 
			state_bin = bin(k)[2:].zfill(L)				#binary representation of the state |101110...>
			state_list = [int(x) for x in list(state_bin)] 	#convert it to list of int form
			if state_list[i] == 1 and state_list[j] == 1:
				sign_j = calc_sign(state_list,j)
				state_list[j] = 0
				sign_i = calc_sign(state_list,i)
				state_list[i] = 0
				ket_index = basis[int(''.join(str(x) for x in state_list),2)]
				op[basis[ket_index],basis[k]] += amp*sign_i*sign_j
	return op.tocsc()

def expect_val(op,psi):
	'''
		Expectation value of operator in the state psi. Psi is assumed to be represented in full many-body basis
	'''
	val = 0. + 0.0j
	indx = op.nonzero()
	row = indx[0]
	col = indx[1]

	for i in range(row.shape[0]):
		val = val + op[row[i],col[i]]*np.conjugate(psi[row[i]])*psi[col[i]]
	return val


