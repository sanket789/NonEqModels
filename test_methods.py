import methods as f
import numpy as np
import unittest
from ddt import ddt,file_data,data,unpack
import math


@ddt
class TestPhase(unittest.TestCase):
	@file_data('TestData/test_phase.json')
	def test_Phase(self,value):
		L = value[0]
		A = value[1]
		w = value[2]
		t = value[3]
		expected = f.phase(L,A,w,t)
		result = value[4]
		self.assertAlmostEqual(result,expected)	

class TestHam(unittest.TestCase):
	def test_ham(self):
		L = 4
		J = 0.9
		phi = -59.45
		mu0 = 3.67
		sigma = np.sqrt(7)
		alpha = 0.43

		result = f.ham_drive_pseudo_PBC(L,J,phi,mu0,sigma,alpha)
		expected = np.array([[3.335904301,-J*np.exp(-1j*phi),0.,-J*np.exp(1j*phi)],
							[-J*np.exp(1j*phi),-0.8188498279,-J*np.exp(-1j*phi),0.],
							[0.,-J*np.exp(1j*phi),-2.338266455,-J*np.exp(-1j*phi)],
							[-J*np.exp(-1j*phi),0.,-J*np.exp(1j*phi),3.667654371]])
		
		self.assertTrue(np.allclose(result,expected))

@ddt
class Testxlogx(unittest.TestCase):
	@data(0.05,0.965,0.99947)
	def test_xlogx(self,value):
		result = f.xlogx(value)
		expected = value*math.log(value)+(1-value)*math.log(1-value)
		self.assertEqual(result,expected)
	@data(0,-40.,1.0005,-0.000008)
	def test_xlogx_exception(self,value):
		result = f.xlogx(value)
		expected = 0.
		self.assertEqual(result,expected)

class TestEntropy(unittest.TestCase):
	def test_entropy(self):
		AA = np.array([[ 0.68496971 -7.94093388e-23j , 0.08253590 +1.99931875e-03j,
						0.01708342 +3.89802395e-05j , 0.07183789 -1.73430033e-03j],
					 [ 0.08253590 -1.99931875e-03j , 0.02126937 -6.77626358e-21j,
   						0.10761136 +2.62667557e-03j , 0.01852958 +8.20031300e-05j],
 					[ 0.01708342 -3.89802395e-05j , 0.10761136 -2.62667557e-03j,
 					  0.98542522 -2.64697796e-23j , 0.09403497 +2.28980269e-03j],
 					[ 0.07183789 +1.73430033e-03j , 0.01852958 -8.20031300e-05j,
   						0.09403497 -2.28980269e-03j , 0.01618254 -6.77626358e-21j]])
		result1,result2 = f.vonNeumann_entropy(AA)
		expected1 = 0.6126380194835308
		expected2 = np.array([ -2.32932085e-09,   4.55044753e-09,   6.97894284e-01, 1.00995255e+00])
		self.assertAlmostEqual(result1,expected1)
		self.assertTrue(np.allclose(result2,expected2))

class TestCurrent(unittest.TestCase):
	def test_charge_current(self):
		J = 13.87
		phi = -413.6
		AA = np.array([[ 0.68496971 -7.94093388e-23j , 0.08253590 +1.99931875e-03j,
						0.01708342 +3.89802395e-05j , 0.07183789 -1.73430033e-03j],
					 [ 0.08253590 -1.99931875e-03j , 0.02126937 -6.77626358e-21j,
   						0.10761136 +2.62667557e-03j , 0.01852958 +8.20031300e-05j],
 					[ 0.01708342 -3.89802395e-05j , 0.10761136 -2.62667557e-03j,
 					  0.98542522 -2.64697796e-23j , 0.09403497 +2.28980269e-03j],
 					[ 0.07183789 +1.73430033e-03j , 0.01852958 -8.20031300e-05j,
   						0.09403497 -2.28980269e-03j , 0.01618254 -6.77626358e-21j]])
		result = f.charge_current(J,phi,AA)
		expected = np.array([-1.7448271717105557,-2.0045781413955264,-2.6133388875184131,-2.2837070106343336])
		self.assertTrue(np.allclose(result,expected))


class TestMainEntropy(unittest.TestCase):
	def test_main_entropy(self):
		result = [[[0.,0.11140758347149794]],[[0.,0.11140758347149794]]]
		self.assertTrue(np.allclose(result,sim_entropy))

class TestMainEigCC(unittest.TestCase):
	def test_main_eigCC(self):
		result = [[[[0.,1.],[ 0.00991571,  0.9900452 ]]],[[[0.0,1.0],[ 0.00991571,  0.9900452]]]]
		self.assertTrue(np.allclose(result,sim_eigCC))

class TestMainNbar(unittest.TestCase):
	def test_main_nbar(self):
		result = [[[0.5,0.49998046]],[[0.5,0.49998046]]]
		self.assertTrue(np.allclose(result,sim_nbar))

class TestMainImbalance(unittest.TestCase):
	def test_main_imbalance(self):
		result = [[[1.0,0.9608319799999999]],[[1.0,0.9608319799999999]]]
		self.assertTrue(np.allclose(result,sim_imbalance))

class TestMainCurrent(unittest.TestCase):
	def test_main_current(self):
		result = [[[[0.,0.,0.,0.],[-0.16452521725895811,0.18694022715261677,-0.15708617702388358,0.16257269298233326]]],
					[[[0.,0.,0.,0.],[-0.17576812185811258,0.13871157342608051,-0.1804843392520763,0.17711552727047405]]]]
		self.assertTrue(np.allclose(result,sim_current))

class TestMainEnergy(unittest.TestCase):
	def test_main_energy(self):
		result = [[[3.2622771741508854,3.2569353651522164]],[[-3.2622771741508858,-3.2500297945698748]]]
		self.assertTrue(np.allclose(result,sim_energy))

class TestMainNNsite(unittest.TestCase):
	def test_main_nn_site(self):
		result = [[1.0,0.98033441],[1.,0.98033441]]
		self.assertTrue(np.allclose(result,sim_nn_site))


if __name__=='__main__':
	#L = 4; J = 1; A = 128; w = 15; mu0 = 3.0 s_1.61803(golden ratio); tf = 0.2; dt = 0.1 
	sim_entropy = np.load('TestData/AVG_L_4_J_1_A_128_w_15_mu0_3_s_1.61803_conf_2_tf_0.2_dt_0.1_entropy.npy')
	sim_eigCC = np.load('TestData/AVG_L_4_J_1_A_128_w_15_mu0_3_s_1.61803_conf_2_tf_0.2_dt_0.1_CCspectrum.npy')
	sim_nbar = np.load('TestData/AVG_L_4_J_1_A_128_w_15_mu0_3_s_1.61803_conf_2_tf_0.2_dt_0.1_nbar.npy')
	sim_imbalance = np.load('TestData/AVG_L_4_J_1_A_128_w_15_mu0_3_s_1.61803_conf_2_tf_0.2_dt_0.1_imbalance.npy')
	sim_current = np.load('TestData/AVG_L_4_J_1_A_128_w_15_mu0_3_s_1.61803_conf_2_tf_0.2_dt_0.1_current.npy')
	sim_energy = np.load('TestData/AVG_L_4_J_1_A_128_w_15_mu0_3_s_1.61803_conf_2_tf_0.2_dt_0.1_energy.npy')
	sim_nn_site = np.load('TestData/AVG_L_4_J_1_A_128_w_15_mu0_3_s_1.61803_conf_2_tf_0.2_dt_0.1_nn_site.npy')

	unittest.main(verbosity=1)