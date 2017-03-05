import numpy as np
from scipy import optimize

class ABCPostProcessor():
    def __init__(self, data, sample_data, sample_parameters):
        """Create an instance of the ABCPostProcessor
        
        Parameters
        ----------
        
        data : array
            data point for inference
        
        sample_data : array
            each row contains the summary statistics of one sample, preceded
            by an integer value identifying the sample.
            each column corresponds to a different summary statistic.
            Note, that ABCPostProcessor will not conduct any weighting or scaling
            of the summary statistics. This needs to be done by the user 
            before creating an instance of the ABCPostProcessor
            
        sample_parameters : array
            each row contains the parameter value of one sample, preceded 
            by an integer value identifying the sample
            each column corresponds to a different parameter
        """
        
        self.sample_data = sample_data
        self.sample_parameters = sample_parameters
        self.data = data
        self.bandwidth = None
        
        if not isinstance(data, float): 
            assert(len(data) == len(sample_data[0,1:]))
        else:
            assert (sample_data.shape[1] == 2)
           
    def find_closest_samples(self, num_samples ):
        """Find and return the parameter values of the closest samples.
        Needs to be run before estimating posterior densities. 
        This method will set the member variable self.accepted_parameters
        to closest_samples (the return value of this function)
        and also set the member variable self.accepted_weights to an array of ones.
        
        Parameters
        ----------
        
        num_samples : int
            The number of samples that should be considered.
            
        Returns
        -------
        
        closest_samples : array
            Each row corresponds to one accepted parameter.
            The row contains the id of the accepted sample and its parameter value
        """
        difference_table = self.sample_data[:,1:] - self.data
        self.difference_table = np.column_stack((self.sample_data[:,0], difference_table))

        distances = np.linalg.norm(self.sample_data[:,1:] - self.data, axis = 1)
        self.distance_table = np.column_stack((self.sample_data[:,0], distances))

        distance_table_sorted = self.distance_table[self.distance_table[:,1].argsort()]
        accepted_ids = distance_table_sorted[:num_samples,0]
        accepted_parameters = self.get_table_entries_for_ids(accepted_ids, self.sample_parameters)
        self.accepted_parameters = accepted_parameters

        weights = np.ones((len(accepted_ids), 2))
        weights[:,0] = accepted_ids
        self.accepted_weights = weights
        
        return accepted_parameters

    def perform_regression_adjustment(self):
        """Run this method only after having run find_closest_samples. This method will
        perform the regression adjustment of the accepted parameters as described by Beaumont et al
        (2002). This method will rewrite the the member variables self.accepted_parameters
        and self.accepted_weights. Pre-regression parameter values and weights will
        be accessible through the new member variables self.accepted_original_parameters 
        and self.original_weights.
        
        This method will also delete the the current value of the optimal bandwidth for
        posterior density estimation, self.bandwidth, since this value may have changed.
        
        Returns
        -------
        
        regression_parameters : array
            each row contains one adjusted parameter set, preceded by the sample id in the first
            column
            
        weights : array
            array of accepted sample ids and their according epanechnikov weights.
        """
        self.accepted_original_parameters = np.array(self.accepted_parameters)
        self.accepted_original_weights = np.array(self.accepted_weights)

        accepted_ids = self.accepted_parameters[:,0]
        accepted_parameters = self.get_table_entries_for_ids(accepted_ids, self.sample_parameters)[:,1:]
        accepted_distances = self.get_table_entries_for_ids(accepted_ids, self.distance_table)[:,1:]
        accepted_differences = self.get_table_entries_for_ids(accepted_ids, self.difference_table)[:,1:] 
        
        regression_delta = np.max(accepted_distances)
        kernel_distances = self.multivariate_epanechnikov( accepted_distances, 
                                              regression_delta )
        
        W = np.diag( kernel_distances.flatten() )
        X = np.zeros( ( len(kernel_distances), accepted_differences.shape[1] + 1 ) )
        X[:,0] = 1
        X[:,1:] = np.nan_to_num(accepted_differences)
        
        matrix_to_invert =  X.transpose().dot( W.dot(X) )

        regression_matrix = np.linalg.pinv(np.nan_to_num(matrix_to_invert)).dot(X.transpose().dot(W))
        
        regression_coefficients = regression_matrix.dot( accepted_parameters )
        
        print 'regression coefficients are'
        print regression_coefficients
        adjusted_parameters = accepted_parameters - np.nan_to_num(accepted_differences).dot(regression_coefficients[1:,:])

        self.accepted_parameters[:,1:] = adjusted_parameters
        self.accepted_weights[:,1] = kernel_distances
        
        self.bandwidth = None

        return self.accepted_parameters, self.accepted_weights

    def get_table_entries_for_ids(self, sample_ids, table):
        """Get all table entries with the listed ids
        
        Parameters
        ----------
        sample_ids : list or array
           ids for which the table entries should be returned
           
        table : array
           table containing some data, first column contains integer ids
           
        Returns
        -------
        table_entries : array
           the rows of table for which the first entry corresponds to one of the ids in the list
        """
        
        translated_indices = []
        for sample_id in sample_ids:
            this_index = np.where(np.isclose(table[:,0],sample_id))[0][0]
            translated_indices.append(this_index)
        indices = (translated_indices,)
        table_entries = table[indices[0]]
        
        return table_entries
    
    def estimate_posterior_at_positions(self, positions):
        """find_closest_samples must have been called first. Will return
        posterior estimate at the paramter values for positions.
        Will perform autamtic bandwidth selection only if the bandwidth
        has not previously been set.
        
        Parameters
        ----------
        
        positions: array
            each entry corresponds to one position at which we would like the posterior
            to be estimated
            
        Returns
        -------
        
        posterior_values : array
            an estimate of the posterior for all provided positions
        """
        if self.bandwidth is None:
            bandwidth = self.select_density_estimation_bandwidth()

#         positions_as_columns = positions.transpose()
        # posterior_estimate
        density_values = np.zeros(len(positions))
        for index, position in enumerate(positions):
            density_values[index] = np.sum(self.multivariate_epanechnikov(position - self.accepted_parameters[:,1:],
                                                                          self.bandwidth)*
                                           self.accepted_weights[:,1])
        density_values/=np.sum(self.accepted_weights[:,1])
        return density_values

    def select_density_estimation_bandwidth(self):
        """Find the optimal bandwidth for the estimation of the posterior distribution.
        
        Performs least-squares slection of the density estimation bandwidth using the
        epanechnikov kernel
        """
        rule_of_thumb_delta = np.std(self.accepted_parameters[:,1:], axis = 0)
        rule_of_thumb_delta /= 5.

        print "performing bandwidth selection for density estimation"
        optimal_bandwidth = optimize.fmin(self.error_function, x0=rule_of_thumb_delta,
                                         maxiter=1e3, maxfun=1e3, disp=True, xtol=1e-4)
    
#         if ( optimal_bandwidth.shape == (1,) ):
#             optimal_bandwidth = optimal_bandwidth[0]
        self.bandwidth = optimal_bandwidth
        
    def error_function(self, delta):
        """Value of the error function for least-likelyhood density estimation at the value
        delta.
        
        Parameters
        ----------
            delta : double
                value of the bandwidth for which the error function should be calculated
        
        Returns
        -------
            error : double
                value of the error function at delta
        """
        
        # calculate first term:
        weights = self.accepted_weights[:,1]
        data_points = self.accepted_parameters[:,1:]
        assert(len(data_points) == len(weights))
        first_term = 0.
        for data_index, data_point in enumerate(data_points):
            first_term += weights[data_index]*np.sum(weights*self.epanechnikov_convolution( data_point - data_points, delta))
    
        first_term /= np.power(np.sum(weights),2)
        
        # calculate second term
        second_term = 0.0
        leave_one_out_mask = np.zeros_like(weights, dtype = 'bool')
        for data_index, data_point in enumerate(data_points):
            leave_one_out_mask[:] = True 
            leave_one_out_mask[data_index] = False
            leave_one_out_weights = weights[leave_one_out_mask]
            leave_one_out_differences = data_point - data_points[leave_one_out_mask,:]
            leave_one_out_kernel_values = self.multivariate_epanechnikov(leave_one_out_differences,
                                                                    delta)
            sub_summands = leave_one_out_weights*leave_one_out_kernel_values
            this_summand = weights[data_index]/np.sum(leave_one_out_weights)*np.sum(sub_summands)
            second_term += this_summand
        second_term *= -2./np.sum(weights)
    
        error_value = first_term + second_term
    
        return error_value

    def epanechnikov_convolution(self, differences, delta): 
        """Value of the epanechnikov convolution kernel for the differences rescaled by delta.
    
        Parameters:
        -----------
        
        differences : array
            the array for which the convolution kernel should be calculated
        
        delta : array or double
            the delta value for which the convolution kernel should be caclulated (if multidimensional
            needs to contain one value for each column of differences
        """

        if len(differences.shape) == 1:
            assert(delta.shape == differences.shape)
            values = self.epanechnikov_convolution_1d(differences, delta)
            assert(values.shape == differences.shape)
            print 'this line did actually get called'
            return values
        else:
            assert( differences.shape[1] == delta.shape[0])
            product_kernel_estimates = self.epanechnikov_convolution_1d(differences[:,0], delta[0])
            for dimension in range( 1, differences.shape[1] ):
                values_this_dim = self.epanechnikov_convolution_1d( differences[:, dimension], delta[dimension] ) 
                product_kernel_estimates *= values_this_dim
            return product_kernel_estimates

    def epanechnikov_convolution_1d(self, differences, delta):
        """Value for the convolution kernel for each entry of differences, using the double-float delta
        
        Parameters
        ----------
        
        differences : 1D array
            the values for which the convolution kernel should be calculated.
        
        delta : double
            the bandwidth for the convolution kernel
        """
        rescaled_differences = np.abs(differences/delta)
        to_calculate_mask = rescaled_differences < 2
        masked_differences = rescaled_differences[to_calculate_mask]
        convolution_values = np.zeros_like(differences)
        convolution_values[to_calculate_mask] = np.power(2.-masked_differences,3)*(
                                                np.power(masked_differences,2) + 
                                                6.*masked_differences + 4.)
        convolution_values*=0.75*0.75/(30.*delta)
        return convolution_values

    def multivariate_epanechnikov(self, differences, delta):
        """Get a (multivariate if necessary) density estimation using the Epanechnikov
        kernel
        
        Parameters
        ----------
        
        differences : array
            array of distances for which a multivariate epanechnikov kernel should be estimated
        
        Returns
        -------
        
        kernel_densities : array
            array of kernel densities, one entry for each row of differences
        """
#         if len(differences.shape) == 1 or differences.shape[1] == 1:
        if len(differences.shape) == 1:
            assert(isinstance(delta, float) or len(delta) == 1)
            values = self.epanechnikov_1D(differences, delta)
            return values
        else:
            if isinstance(delta, float):
                assert( differences.shape[1] == 1)
                delta = [delta]
            else:
                assert( differences.shape[1] == delta.shape[0])
            product_kernel_estimates = self.epanechnikov_1D(differences[:,0], delta[0])
            for dimension in range( 1, differences.shape[1] ):
                values_this_dim = self.epanechnikov_1D( differences[:, dimension], delta[dimension] ) 
                product_kernel_estimates *= values_this_dim

        return product_kernel_estimates

    def epanechnikov_1D(self, differences, delta):
        """Calculate the epanechnikov function on all difference values
        
        Parameters
        ----------
            differences : array
                array of dimension 1 containing the arguments for the epanechnikov function
                
            delta : double
                bandwidth of the kernel to be used
            
        Returns
        -------
        
        function_values : array
            evaluation of the epanechnikov function at each of the entries for differenes
        """
        rescaled_differences = differences/delta
        to_calculate_mask = np.abs(rescaled_differences) < 1
        masked_differences = rescaled_differences[to_calculate_mask]
        function_values = np.zeros_like(differences)
        function_values[to_calculate_mask] = 1 - np.power(masked_differences, 2)
        function_values*=3./(4.*delta)
        return function_values
