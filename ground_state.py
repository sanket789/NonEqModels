import numpy as np 
import methods as func

# L = 100
# CC_0 = np.zeros((L,L),dtype=complex)
# for j in  range(L):
# 	for k in range(L):
# 		if j==0 and k==0:
# 			CC_0[j,k] = L/4.
# 		elif k==0:
# 			CC_0[j,k] = 0.5*np.exp(-1j*np.pi*j*0.5)*(np.exp(1j*np.pi*j)-1)/(np.exp(1j*np.pi*j*2/L)-1)
# 		elif j == 0:
# 			CC_0[j,k] = 0.5*np.exp(1j*np.pi*k*0.5)*(np.exp(-1j*np.pi*k)-1)/(np.exp(-1j*np.pi*k*2/L)-1)
# 		else:
# 			CC_0[j,k] =(1/L)*np.exp(1j*np.pi*(k-j)*0.5)*(np.exp(1j*np.pi*j)-1)*(np.exp(-1j*np.pi*k)-1)/ \
# 												((np.exp(1j*np.pi*j*2/L)-1)*(np.exp(-1j*np.pi*k*2/L)-1))

# HH = func.ham_anderson_insl_PBC(L,1.,0.,0.)
# EE = np.sum(np.multiply(HH,CC_0)).real
# print(func.is_hermitian(CC_0))
# #print(HH.real)
# #print(CC_0)
# print(EE , -2/np.tan(np.pi/L),-2*L/np.pi)          
# # print(np.sum((np.diag(CC_0,-1))))
# # print(CC_0[0,-1],CC_0[-1,0])
# diagonal = np.diag(CC_0).real
# # print(np.sum(diagonal))

# j = L-1
# # print(0.5*np.exp(-1j*np.pi*j*0.5)*(np.exp(1j*np.pi*j)-1)/(np.exp(1j*np.pi*j*2/L)-1))
# # print((np.exp(1j*np.pi*j*2/L)-1))
# # print(0.866025403784*4)
# indx = np.arange(L/2)-L//4
# CC_1 = np.zeros((L,L),dtype=complex)

# for j in  range(L):
# 	for k in range(L):
# 		if j==0 and k==0:
# 			CC_1[j,k] = L/4. 
# 		elif k==0:
# 			CC_1[j,k] = sum([0.5*np.exp(1j*2*j*n*np.pi/L) for n in indx])
# 		elif j == 0:
# 			CC_1[j,k] = sum([0.5*np.exp(-1j*2*k*n*np.pi/L) for n in indx])
# 		else:
# 			CC_1[j,k] = (1./L)*sum([np.exp(1j*2*j*n*np.pi/L) for n in indx])*sum([np.exp(-1j*2*k*n*np.pi/L) for n in indx])
# print(np.sum(np.multiply(HH,CC_1)).real, -2/np.tan(np.pi/L),-2*L/np.pi)

# print(np.allclose(CC_0,CC_1))

L = 4
CC_0 = np.zeros((L,L),dtype=complex)
CC_1 = np.zeros((L,L),dtype=complex)
indx = np.arange(L/2)-L//4
print(indx)
for j in  range(L):
	for k in range(L):
		CC_0[j,k] =(1./L)*(sum([np.exp(1j*(k-j)*2*np.pi*n/L) for n in indx]))

ham = -1.*np.diag(np.ones(L-1,dtype=complex),-1)  -1.*np.diag(np.ones(L-1),1)
ham[0,-1] = -1
ham[-1,0] = -1
eps,DD = np.linalg.eigh(ham)
CC_1 = np.dot(np.conj(DD)[:,0:L//2],DD.T[0:L//2,:])
print(np.allclose(CC_0,CC_1))
print(np.sum(np.multiply(ham,CC_0)).real, -2/np.tan(np.pi/L))
print(np.sum(np.multiply(ham,CC_1)).real, -2/np.tan(np.pi/L))
print(np.sum(np.diag(CC_0).real),np.sum(np.diag(CC_1).real))
#print(np.allclose(np.sum(np.diag(CC_0,1)),np.sum(np.diag(CC_1,1))))
ind = [0,1,-1,2]
DD1 = np.zeros((L,L),dtype=complex)
for m in range(L):
	for n in range(L):
		DD1[m,n] = (1/np.sqrt(L))*np.exp(1j*m*ind[n]*2.*np.pi/L)
print(np.allclose(np.dot(np.conj(DD.T),DD),np.eye(L)))
#print(np.dot(np.conj(DD.T),np.dot(ham,DD)).real)
#print(np.dot(np.conj(DD1.T),np.dot(ham,DD1)).real)
print(np.allclose(DD,DD1))
print(DD)
print()
print(DD1)
print(np.allclose(CC_0,CC_1))