import numpy as np
import methods as func
L = 6
dt = 0.1
J = 1.
CC_t = np.zeros((L,L),dtype=complex)
for j in range(0,L,2):
 	CC_t[j,j] = 1.0
delta = 1.

#HH = func.ham_anderson_insl_PBC(L,J,0.,np.zeros(L))
HH = np.diag(np.arange(1,L+1))
v_eps , DD = np.linalg.eigh(HH)

EE = np.diag(np.exp(dt*v_eps))
UU = np.dot(np.conj(DD),np.dot(EE,DD.T))
EEp = np.diag(np.exp(dt*v_eps))
UUp = np.dot(np.conj(DD),np.dot(EEp,DD.T))

#CC_t = np.dot(np.conj(DD)[:,0:L//2],DD.T[0:L//2,:])
CC_0 = CC_t.copy()
while True:
	CC_next = np.dot(UUp,np.dot(CC_t,UU))
	delta = abs(np.sum(np.multiply(CC_t,HH)) - np.sum(np.multiply(CC_next,HH)))
	#print('delta = ',delta)
	#print(np.sum(np.multiply(CC_t,HH)))
	print(np.sum(np.multiply(CC_next,HH)))
	CC_t = CC_next.copy()

print(CC_t)
print(np.allclose(CC_t,CC_0))
