import numpy as np
import time
from mpi4py import MPI
import argparse
import pairing_func as pf
import methods as methods
import matplotlib.pyplot as plt

'''
	This is main script for time evolution of Aubry-Andre model with pairing term.  
	The code is written in MPI framework to average over multiple disorder configurations
	
'''

def main_routine(arg,c,alpha):
	start_time = time.time()
	#Extract MPI paramteres
	mpi_size = c.Get_size()
	mpi_rank = c.Get_rank()
	#Define simulation paramteres
	L = arg.L
	J = 1.0
	mu0 = arg.mu0
	dJ = arg.delJ*J
	delta = arg.delta
	T = arg.T
	cyc = arg.cyc
	num_conf = arg.num_conf
	nT = 2*cyc
	#Hamiltonian paramteres
	sigma = 0.5*(1.+np.sqrt(5.))
	list_sub = np.array([10,30,50,70,100])	#list of subsystems for entropy calculation
	num_sub = np.shape(list_sub)[0]     #number of subsystems for entropy calculation
	save_data = True

	'''
		#########################################  Main Code  ##############################################
	'''
	#Initial correlation matrix
	#data storing variables
	fname = "WDIR/Apr1/Data/PAIR_ALT_AUBRY_L_%d_dJ_%g_mu0_%g_delta_%g_T_%g_cyc_%d_ncnf_%g_"%(L,dJ,mu0,delta,T,cyc,num_conf*mpi_size)
	m_diag = np.zeros((num_conf,nT,L))
	m_imb = np.zeros((num_conf,nT))
	m_entropy = np.zeros((num_conf,nT,num_sub))
	# m_mu = np.zeros((num_conf,L))
	# m_GG = np.zeros((num_conf,nT,2*L,2*L),dtype=complex)
	'''
		#########################################  Loop over configurations and time  ##############################################
	'''
	for k in range(num_conf):
		#Define onsite potential
		if mu0 == 0.0:
			mu_array = J*np.ones(L)
		else:
			mu_array = mu0*np.asarray([np.cos(2*np.pi*sigma*nsite + alpha[k]) for nsite in range(L) ])

		# print(mu_array)
		#Define two Hamiltonians
		#Hamiltonian with (J+dJ)
		HH_h = pf.ham(L,J+dJ,delta,mu_array)
		UU_h , VV_h = pf.dynamics(HH_h,0.5*T)	#C(t+1) = UU*C*VV
		#Hamiltonian with (J-dJ)
		HH_l = pf.ham(L,J-dJ,delta,mu_array)
		UU_l , VV_l = pf.dynamics(HH_l,0.5*T)	#C(t+1) = UU*C*VV
		#initialize correlation matrix to the alternate occupied half-filled state		
		CC_0 , FF_0 = pf.alt_initial_state(L)
		GG_t = pf.construct_G(CC_0,FF_0)
		# print(HH_h,HH_l)

		for i in range(nT):
			m_imb[k,i] = methods.imbalance(np.diag(GG_t[L:2*L,L:2*L]).real)
			m_diag[k,i,:] = np.diag(GG_t[L:2*L,L:2*L]).real
			for ent_iter in range(num_sub):
				m_entropy[k,i,ent_iter] = pf.pairing_entropy(GG_t,list_sub[ent_iter])
			if i%2 == 0: 	#Hamiltonian 1
				GG_next = np.dot(UU_h,np.dot(GG_t,VV_h))
			else: 			#Hamiltonian 2
				GG_next = np.dot(UU_l,np.dot(GG_t,VV_l))
			GG_t = GG_next.copy()
			
	'''
		############################	Gather the data to be saved 	##################################################
	'''
	recv_imb = None
	recv_diag = None
	recv_entropy = None
	recv_GG = None
	if mpi_rank	== 0:
		recv_imb = np.empty([mpi_size,nT])
		recv_diag = np.empty([mpi_size,nT,L])
		recv_entropy = np.empty([mpi_size,nT,num_sub])
	c.Gather(np.mean(m_imb,0),recv_imb,root=0)
	c.Gather(np.mean(m_diag,0),recv_diag,root=0)
	c.Gather(np.mean(m_entropy,0),recv_entropy,root=0)
	if mpi_rank	== 0:
		recv_diag = np.mean(recv_diag,0)
		recv_imb = np.mean(recv_imb,0)
		recv_entropy  = np.mean(recv_entropy,0)
		if save_data == True:
			np.save(fname+"imb.npy",recv_imb)
			np.save(fname+"diag.npy",recv_diag)
			np.save(fname+"entropy.npy",recv_entropy)
			#np.save(fname+"mu.npy",recv_mu)
			print("Data files saved at : %s"%fname)
	end_time = time.time()
	print('Time taken by rank %d : %g seconds '%(mpi_rank,end_time - start_time))
'''
	#############################	Argument parsing 	#####################################
'''

if __name__ == '__main__':	
	comm = MPI.COMM_WORLD
	rank = comm.Get_rank()
	size = comm.Get_size()
	if rank==0:
		print('----------------------------------------------------------')
		print('Aubry-Andre + Pairing term  with alternately occupied initial state')
		print('----------------------------------------------------------')
		print()
		## parsing at rank =0
		parser = argparse.ArgumentParser(
				  description='Time evolution of fermion chain',
				  prog='main', usage='%(prog)s [options]')
		parser.add_argument("-L", "--L", help="System Size",default=100,type=int, nargs='?')
		parser.add_argument("-delJ","--delJ",help="Drive strength; hopping paramtere = (1+delJ)*J",default=.1,type=float,nargs='?')
		parser.add_argument("-delta", "--delta", help="Amplitude of pairing term",default=2.0,type=float, nargs='?')
		parser.add_argument("-T", "--T", help="Time period",default=1.0,type=float, nargs='?')
		parser.add_argument("-mu0", "--mu0", help="Strength of chemical potential",default=2.0,type=float, nargs='?')
		parser.add_argument("-num_conf","--num_conf",help = "number of disorder config per MPI_rank",default=100,type=int,nargs='?')
		parser.add_argument("-cyc","--cyc",help="Number of drive cycles",default=100,type=int,nargs='?')
		#-- add in the argument
		arg=parser.parse_args()
		print(arg)

	else:
		arg = None

	bcasted_args = comm.bcast(arg, root=0)

	#Scatter disorder configurations
	sendbuf = None
	if rank == 0:
		np.random.seed(0)
		#generate num_conf*mpi_size random numbers in [0,2-pi)
		sendbuf = 2*np.pi*np.random.random_sample((size,bcasted_args.num_conf))
	recvbuf = np.empty(bcasted_args.num_conf, dtype=float)
	comm.Scatter(sendbuf, recvbuf, root=0)
	
	# run main code
	main_routine(bcasted_args,comm,recvbuf)
