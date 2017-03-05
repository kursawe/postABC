import unittest
import numpy as np
import postABC
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import os
from scipy import integrate

class TestABC2D(unittest.TestCase):
    def test_2D_rejection(self):
        # OK generate some samples from the Model
        # Y = x + N(0,1)
        my_samples = np.zeros((10000, 3), dtype = 'double')
        parameter_values = np.zeros_like(my_samples)
        my_samples[:,0] = np.linspace(1,10000, 10000)
        parameter_values[:,0] = my_samples[:,0]

        parameter_values[:,1] = np.random.rand(10000)*13.0 - 3.0
        parameter_values[:,2] = np.random.rand(10000)*13.0 - 3.0
        ya = np.random.randn(10000) + parameter_values[:,1]
        yb = np.random.randn(10000) + parameter_values[:,2]
        my_samples[:,1] = ya + 2*yb
        my_samples[:,2] = ya - yb
        
        data = [ 13, -2 ] 

        abc_processor = postABC.ABCPostProcessor(data,
                                                 my_samples,
                                                 parameter_values)

        closest_samples = abc_processor.find_closest_samples( num_samples = 1000 )
        self.assertEqual( closest_samples.shape, (1000,3) )

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
        X, Y = np.mgrid[-3:10:1000j, 0:10:1000j]
        positions = np.vstack([X.ravel(), Y.ravel()]).transpose()
        values = abc_processor.estimate_posterior_at_positions(positions)
        Z = np.reshape(values.T, X.shape)
        Z_true = mlab.bivariate_normal(X, Y, 1.0, 1.0, 3.0, 5.0)


        # make a figure
        
        figuresize = (3.4,2.5)
        plt.figure(figsize = figuresize)
        plt.contour(X,Y,Z_true, colors = 'black', alpha = 0.5)
        rejection_contour = plt.contour(X,Y,Z, colors = 'black', linestyles = 'dashed', alpha = 0.5)
        for line in rejection_contour.collections:
            line.set_dashes([(0, (2.0, 2.0))])
#         plt.plot(x_values, mlab.normpdf(x_values, 3.0, 1.0), color = 'black')
#         plt.plot(x_values, y_values, color = 'black', linestyle = '--')
        plt.savefig(os.path.join(os.path.dirname(__file__),'output', '2D_rejection_only.pdf'))
        
        # Now, we need to run the regression_adjustment
        regression_samples, weights = abc_processor.perform_regression_adjustment()
        np.testing.assert_allclose(regression_samples[:,0], closest_ids)
        
        new_values = abc_processor.estimate_posterior_at_positions(positions)
        new_Z = np.reshape(new_values.T, X.shape)

        regression_contour = plt.contour(X, Y, new_Z, colors = 'black', linestyles = 'dashed', alpha = 0.5)
#                   color = 'black', linestyle = ':')
        for line in regression_contour.collections:
            line.set_dashes([(0, (0.5, 0.5))])
        plt.savefig(os.path.join(os.path.dirname(__file__),'output', '2D_rejection_and_regression.pdf'))
        