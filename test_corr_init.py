import corr_init as f
import numpy as np
import unittest
from ddt import ddt,file_data,data,unpack
import math

@ddt
class TestGround(unittest.TestCase):
	# @file_data('TestData/test_phase.json')
	def test_ground(self):
		L = 102
		result = np.zeros((L,L),dtype=complex)
		H = np.diag(np.arange(L))
		expected = f.ground_state(L,H)
		for j in range(L//2):
			result[j,j] = 1.0

		self.assertTrue(np.allclose(result,expected))

	def test_k_space_ground(self):
		L = 98
		result = np.zeros((L,L),dtype=complex)
		H = np.diag(5.*np.ones(L)) + np.diag(-1*np.ones(L-1),-1) + np.diag(-1*np.ones(L-1),1)
		H[-1,0] = -1
		H[0,-1] = -1
		expected = f.ground_state(L,H)
		indx = np.arange(L/2) - L//4
		for j in  range(L):
			for k in range(L):
				result[j,k] =(1./L)*(sum([np.exp(1j*(k-j)*2*np.pi*n/L) for n in indx]))
		self.assertTrue(np.allclose(result,expected))
if __name__=='__main__':

	unittest.main(verbosity=2)