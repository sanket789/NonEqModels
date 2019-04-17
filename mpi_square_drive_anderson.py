import numpy as np
import methods as func
import time
from mpi4py import MPI
import argparse
# import matplotlib.pyplot as plt
from corr_init import ground_state,alternate_occupy
from time_evolution import evolve_correlation_matrix
'''
	This is main script for time evolution of Anderson insulator.  
	The code is written in MPI framework to average over multiple disorder configurations
	https://doi.org/10.1103/PhysRevB.98.214202	
'''
def ham(J,dJ,L,chem):
	HH = np.diag(chem) - (J+dJ)*np.diag(np.ones(L-1),1)- (J+dJ)*np.diag(np.ones(L-1),-1)
	HH[-1,0] = -(J+dJ)
	HH[0,-1] = -(J+dJ)
	return HH

def main_routine(arg,c):
	start_time = time.time()
	#Extract MPI paramteres
	mpi_size = c.Get_size()
	mpi_rank = c.Get_rank()
	#Define simulation paramteres
	L = arg.L
	J = 1.0
	mu0 = arg.mu0
	dJ = arg.delJ*J
	T = arg.T
	cyc = arg.cyc
	num_conf = arg.num_conf
	nT = 2*cyc
	num_sub = 10	#number of subsystem 
	list_sub = 10*np.arange(1,num_sub+1)
	save_data = True
	if nT<40:
		num_CC_samples = cyc
	else:
		num_CC_samples = 20
	'''
		#########################################  Main Code  ##############################################
	'''
	#Initial correlation matrix
	HH_start = ham(J,dJ,L,np.zeros(L))
	CC_0 = alternate_occupy(L)
	#Initialize the storage matrices
	fname = "WDIR/Mar8/Data/SQ_ALT_ANDERSON_L_%d_dJ_%g_mu0_%g_T_%g_cyc_%d_ncnf_%g_"%(L,dJ,mu0,T,cyc,num_conf*mpi_size)
	#m_energy = np.zeros((num_conf,nT))
	#m_nbar = np.zeros((num_conf,nT))
	m_imb = np.zeros((num_conf,nT))
	#m_absb_energy = np.zeros((num_conf,nT))
	m_diag = np.zeros((num_conf,nT,L))
	#m_curr = np.zeros((num_conf,nT,L))
	#m_excit = np.zeros((num_conf,L))
	m_entropy = np.zeros((num_conf,nT,num_sub))
	#m_CC = np.zeros((num_conf,num_CC_samples,L,L),dtype=complex)
	#m_mu = np.zeros((num_conf,L))
	#tlist = []
	'''
		#########################################  Loop over configurations and time  ##############################################
	'''
	for k in range(num_conf):
		mu_array = np.random.uniform(-mu0,mu0,L)
		#m_mu[k,:] = mu_array.copy()
		#Define two Hamiltonians
		#Hamiltonian with (J+dJ)
		HH_h = ham(J,dJ,L,mu_array)
		v_eps_h , DD_h = np.linalg.eigh(HH_h)
		EE_h = np.diag(np.exp(-1j*0.5*T*v_eps_h))
		UU_h = np.dot(np.conj(DD_h),np.dot(EE_h,DD_h.T))
		#Hamiltonian with (J-dJ)
		HH_l = ham(J,-dJ,L,mu_array)
		v_eps_l , DD_l = np.linalg.eigh(HH_l)
		EE_l = np.diag(np.exp(-1j*0.5*T*v_eps_l))
		UU_l = np.dot(np.conj(DD_l),np.dot(EE_l,DD_l.T))
		#initialize correlation matrix to the ground state of HH_start		
		CC_t = CC_0.copy()
		#j=0
		for i in range(nT):
			#if i in np.linspace(2,nT-1,num_CC_samples,endpoint=True):
			#	m_CC[k,j,:,:] = CC_t.copy()
			#	j = j+1
			#	tlist.append(i*T/2)
			#Calculate Hamiltonian
			#m_nbar[k,i] = np.trace(CC_t).real		 
			m_imb[k,i] = func.imbalance(np.diag(CC_t).real)
			m_diag[k,i,:] = np.diag(CC_t).real
			for ent_iter in range(num_sub):
				ent,eig_ent = func.vonNeumann_entropy(CC_t[0:list_sub[ent_iter],0:list_sub[ent_iter]])
				m_entropy[k,i,ent_iter] = ent
			if i%2 == 0:
				#m_energy[k,i] = np.sum(np.multiply(HH_h,CC_t)).real	
				# m_absb_energy[k,i] = (m_energy[k,i] - m_energy[k,0])			
				#m_curr[k,i,:] = func.charge_current(J+dJ,0.,CC_t)
				CC_next = np.dot(np.conj(UU_h.T),np.dot(CC_t,UU_h))
			else:
				#m_energy[k,i] = np.sum(np.multiply(HH_l,CC_t)).real
				# m_absb_energy[k,i] = (m_energy[k,i] - m_energy[k,0])
				#m_curr[k,i,:] = func.charge_current(J-dJ,0.,CC_t)
				CC_next = np.dot(np.conj(UU_l.T),np.dot(CC_t,UU_l))
			CC_t = CC_next.copy()

		#m_absb_energy[k,:] = func.absorbed_energy(m_energy[k,:],HH_h)
		#m_excit[k,:] = func.energy_excitations(v_eps_h,DD_h,CC_t)
	'''
		############################	Gather the data to be saved 	##################################################
	'''
	#recv_energy = None
	#recv_nbar = None
	recv_imb = None
	#recv_absb_energy = None
	recv_diag = None
	#recv_curr = None
	#recv_excit = None
	recv_entropy = None
	#recv_CC = None
	#recv_mu = None
	if mpi_rank	== 0:
		#recv_energy = np.empty([mpi_size,num_conf,nT])
		#recv_nbar = np.empty([mpi_size,num_conf,nT])
		recv_imb = np.empty([mpi_size,num_conf,nT])
		#recv_absb_energy = np.empty([mpi_size,num_conf,nT])
		recv_diag = np.empty([mpi_size,num_conf,nT,L])
		#recv_curr = np.empty([mpi_size,num_conf,nT,L])
		#recv_excit = np.empty([mpi_size,num_conf,L])
		recv_entropy = np.empty([mpi_size,num_conf,nT,num_sub])
		#recv_CC = np.empty([mpi_size,num_conf,num_CC_samples,L,L],dtype=complex)
		#recv_mu = np.empty([mpi_size,num_conf,L])
	#c.Gather(m_energy,recv_energy,root=0)
	#c.Gather(m_nbar,recv_nbar,root=0)
	c.Gather(m_imb,recv_imb,root=0)
	#c.Gather(m_absb_energy,recv_absb_energy,root=0)
	c.Gather(m_diag,recv_diag,root=0)
	#c.Gather(m_curr,recv_curr,root=0)
	#c.Gather(m_excit,recv_excit,root=0)
	c.Gather(m_entropy,recv_entropy,root=0)
	#c.Gather(m_CC,recv_CC,root=0)
	#c.Gather(m_mu,recv_mu,root=0)
	if mpi_rank	== 0:
		recv_diag = np.mean(recv_diag,(0,1))
		#recv_curr = np.mean(recv_curr,(0,1))
		if save_data == True:
			#np.save(fname+"tlist.npy",tlist)
			#np.save(fname+"energy.npy",recv_energy)
			#np.save(fname+"nbar.npy",recv_nbar)
			np.save(fname+"imb.npy",recv_imb)
			#np.save(fname+"absb.npy",recv_absb_energy)
			np.save(fname+"diag.npy",recv_diag)
			#np.save(fname+"curr.npy",recv_curr)
			#np.save(fname+"excit.npy",recv_excit)
			np.save(fname+"entropy.npy",recv_entropy)
			#np.save(fname+"CC.npy",recv_CC)
			#np.save(fname+"mu.npy",recv_mu)
			print("Data files saved at : %s"%fname)
	end_time = time.time()
	print('Time taken by rank %d : %g seconds'%(mpi_rank,end_time - start_time))
'''
	#############################	Argument parsing 	#####################################
'''

if __name__ == '__main__':	
	comm = MPI.COMM_WORLD
	rank = comm.Get_rank()
	if rank==0:
		print('----------------------------------------------------------')
		print('Anderson insulator with alternately occupied initial state')
		print('----------------------------------------------------------')
		print()
		## parsing at rank =0
		parser = argparse.ArgumentParser(
				  description='Time evolution of fermion chain',
				  prog='main', usage='%(prog)s [options]')
		parser.add_argument("-L", "--L", help="System Size",default=100,type=int, nargs='?')
		parser.add_argument("-delJ","--delJ",help="Drive strength; hopping paramtere = (1+delJ)*J",default=.1,type=float,nargs='?')
		parser.add_argument("-T", "--T", help="Time period",default=1.0,type=float, nargs='?')
		parser.add_argument("-mu0", "--mu0", help="Strength of chemical potential",default=2.0,type=float, nargs='?')
		parser.add_argument("-num_conf","--num_conf",help = "number of disorder config per MPI_rank",default=100,type=int,nargs='?')
		parser.add_argument("-cyc","--cyc",help="Number of drive cycles",default=100,type=int,nargs='?')
		#-- add in the argument
		args=parser.parse_args()
		print(args)

	else:
		args = None

	# broadcast
	bcasted_args = comm.bcast(args, root=0)

	# run main code
	main_routine(bcasted_args, comm)
