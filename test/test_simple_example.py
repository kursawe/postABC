import unittest
import numpy as np
import postABC
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import os
from scipy import integrate

class TestABC1D(unittest.TestCase):
    def test_1D_rejection(self):
        # OK generate some samples from the Model
        # Y = x + N(0,1)
        my_samples = np.zeros((10000, 2), dtype = 'double')
        parameter_values = np.zeros_like(my_samples)
        my_samples[:,0] = np.linspace(1,10000, 10000)
        parameter_values[:,0] = my_samples[:,0]

        parameter_values[:,1] = np.random.rand(10000)*13.0 - 3.0
        my_samples[:,1] = np.random.randn(10000)
        my_samples[:,1] += parameter_values[:,1]
        
        data = 3.0 

        abc_processor = postABC.ABCPostProcessor(data,
                                                 my_samples,
                                                 parameter_values)

        closest_samples = abc_processor.find_closest_samples( num_samples = 1000 )
        self.assertEqual( closest_samples.shape, (1000,2) )

        # test that all of these samples are closer in data space than all other samples
        closest_ids = closest_samples[:,0]
        distance_table = np.zeros_like(my_samples)
        distance_table[:,0] = my_samples[:,0]
        distance_table[:,1] = np.abs(data - my_samples[:,1])
        accepted_distances = abc_processor.get_table_entries_for_ids(closest_ids, distance_table)
        maximal_accepted_distance = np.max(accepted_distances[:,1])
        accepted_integer_ids = np.array(closest_ids, dtype = 'int')
        for entry in distance_table:
            if int(entry[0]) not in accepted_integer_ids:
                self.assertGreater(entry[1], maximal_accepted_distance)
        
        
        # calculate a density estimate at the positions
        x_values = np.linspace(-3,10,1000)
        y_values = abc_processor.estimate_posterior_at_positions(x_values)

        # make a figure
        
        figuresize = (3.4,2.5)
        plt.figure(figsize = figuresize)
        plt.plot(x_values, mlab.normpdf(x_values, 3.0, 1.0), color = 'black')
        plt.plot(x_values, y_values, color = 'black', linestyle = '--')
        plt.savefig(os.path.join(os.path.dirname(__file__),'output', '1D_rejection_only.pdf'))
        
        # Now, we need to run the regression_adjustment
        regression_samples, weights = abc_processor.perform_regression_adjustment()
        np.testing.assert_allclose(regression_samples[:,0], closest_ids)
        
        new_y_values = abc_processor.estimate_posterior_at_positions(x_values)

        plt.plot(x_values, new_y_values, color = 'black', linestyle = ':')
        plt.savefig(os.path.join(os.path.dirname(__file__),'output', '1D_rejection_and_regression.pdf'))
        
    def test_1D_rejection_multiple_stats(self):
        # OK generate some samples from the Model
        # Y = x + N(0,1)
        my_samples = np.zeros((10000, 3), dtype = 'double')
        parameter_values = np.zeros( ( my_samples.shape[0], 2 ) )
        my_samples[:,0] = np.linspace(1,10000, 10000)
        parameter_values[:,0] = my_samples[:,0]

        parameter_values[:,1] = np.random.rand(10000)*13.0 - 3.0
        my_samples[:,1] = np.random.randn(10000)
        my_samples[:,1] += parameter_values[:,1]
        my_samples[:,2] = np.power(my_samples[:,1],2)
        
        data = np.array([3.0, 9.0]) 

        abc_processor = postABC.ABCPostProcessor(data,
                                                 my_samples,
                                                 parameter_values)

        closest_samples = abc_processor.find_closest_samples( num_samples = 1000 )
        self.assertEqual( closest_samples.shape, (1000,2) )

        # test that all of these samples are closer in data space than all other samples
        closest_ids = closest_samples[:,0]
        distance_table = np.zeros_like(my_samples)
        distance_table[:,0] = my_samples[:,0]
        distance_table[:,1] = np.linalg.norm(data - my_samples[:,1:], axis = 1)
        accepted_distances = abc_processor.get_table_entries_for_ids(closest_ids, distance_table)
        maximal_accepted_distance = np.max(accepted_distances[:,1])
        accepted_integer_ids = np.array(closest_ids, dtype = 'int')
        for entry in distance_table:
            if int(entry[0]) not in accepted_integer_ids:
                self.assertGreater(entry[1], maximal_accepted_distance)
        
        
        # calculate a density estimate at the positions
        x_values = np.linspace(-3,10,1000)
        y_values = abc_processor.estimate_posterior_at_positions(x_values)
#         y_values /= integrate.simps(y_values, x_values)

        # make a figure
        
        figuresize = (3.4,2.5)
        plt.figure(figsize = figuresize)
        plt.plot(x_values, mlab.normpdf(x_values, 3.0, 1.0), color = 'black')
        plt.plot(x_values, y_values, color = 'black', linestyle = '--')
        plt.savefig(os.path.join(os.path.dirname(__file__),'output', '1D_rejection_only.pdf'))
        
        # Now, we need to run the regression_adjustment
        regression_samples, weights = abc_processor.perform_regression_adjustment()
        np.testing.assert_allclose(regression_samples[:,0], closest_ids)
        
        new_y_values = abc_processor.estimate_posterior_at_positions(x_values)
#         new_y_values /= integrate.simps(new_y_values, x_values)

        plt.plot(x_values, new_y_values, color = 'black', linestyle = ':')
        plt.savefig(os.path.join(os.path.dirname(__file__),'output', '1D_multiple_stats.pdf'))
