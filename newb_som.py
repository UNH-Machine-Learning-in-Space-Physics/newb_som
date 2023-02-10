#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 10:27:23 2023

@author: jedmond
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd






_RECTANGULAR = 'rectangular'
_ALLOWED_TOPOLOGY = [ _RECTANGULAR ]

_GAUSSIAN = 'gaussian'
_ALLOWED_NEIGHBORHOOD = [ _GAUSSIAN ]

_LINEAR = 'linear'
_ALLOWED_DECAY = [ _LINEAR ]

_EUCLIDEAN = 'euclidean'
_ALLOWED_METRIC = [ _EUCLIDEAN ]










class Neighborhood:
    
    """
    Tracks the various neighborhood functions that can be used in newb_som.
    Purely abstract class, can't be instantiated.
    """
    
    
    
    
    
    def _get_gaussian(distances, sigma_t):
        """
        Gaussian neighborhood function; distances are computed PRIOR
        to calling this function!

        Parameters
        ----------
        distances : 1d numpy array
            Array of distances between winning node and all other nodes
        sigma_t : number
            sigma value at iteration t

        Returns
        -------
        None.
        """
        return np.exp( -1*np.power(distances,2) / (2 * np.power(sigma_t,2)))
    
    
    
    
    
    def get_neighborhood_function(neighborhood_type):
        """
        Gets the desired neighborhood function
        """
        
        func = None
        if neighborhood_type == _GAUSSIAN:
            return Neighborhood._get_gaussian
        
        else:
            raise ValueError('Neighborhood function \'' + neighborhood_type
                             + '\' not recognized.')
            
        return func










class Distance:
    
    """
    Tracks the various distance functions that can be used in newb_som.
    Purely abstract class, can't be instantiated.
    """
    
    
    
    
    
    def _get_euclidean_distance(starting_pt, ending_pts):
        """
        Euclidean distance - the subtraction operation is applied over the
        LAST axis in the ending_pts n-d array

        Parameters
        ----------
        starting_pt: 1d numpy array (a vector)
            Point distances will be measureed relative to.
        ending_pts: n-d numpy array 
            Ending point(s) of vector(s)
            

        Returns
        -------
        1d numpy array
            Distance(s) from starting_pt to ending_pts
        """
        return np.linalg.norm( starting_pt - ending_pts,
                               axis=len(ending_pts.shape)-1 )
        
    
    
        
    
    def get_distance_function(distance_type):
        """
        Retrieves the desired distance function 
        """
        func = None
        if distance_type == _EUCLIDEAN:
            func = Distance._get_euclidean_distance
            
        else:
            raise ValueError('Distance function \'' + distance_type
                             + '\' not recognized.')
            
        return func










class Decay:
    
    """
    Tracks the various decay functions that can be used in newb_som.
    Purely abstract class, can't be instantiated.
    """
    
    
    
    
    
    def _get_linear_decay(start_val, end_val, current_iter, max_iter):
        """
        Linear decay function

        Parameters
        ----------
        start_val : number
            Starting value.
        end_val : number
            Ending value.
        current_iter : int
            Current iteration integer.
        max_iter : int
            Max iteration allowed.

        Returns
        -------
        number
            value at iteration current_iter
        """
        slope = (end_val - start_val) / max_iter   # always negative!
        intercept = start_val
        return slope * current_iter + intercept
        
    
    
        
    
    def get_decay_function(decay_type):
        """
        Retrieves the desired decay function 
        """
        func = None
        if decay_type == _LINEAR:
            func = Decay._get_linear_decay
            
        else:
            raise ValueError('Decay function \'' + decay_type
                             + '\' not recognized.')
            
        return func
            









class newb_som:
    
    """
    Personal implementation of a self-organizing map (som).
    Focuses on showcasing how it works rather than being efficient.
    Borrows heavily from minisom and XPySom.
    
    Params here
    
    Funcs here
    
    --- Notes about array structures ---
    Weights:
        Weights are saved in a single 3d array. First two dims are for
        x-nodes and y-nodes respectively. Remaining dim is the weight vector
        for the specified node.
        I.e. the weight vector for node x=2, y=0 can be indexed using
            self._weights[2,0,:]
            
    
    """
    
    
    def __init__(self, som_shape             = None,
                       data_size             = None,
                       max_iter              = None,
                       n_epoch               = None,
                       sigma_start           = None,
                       sigma_end             = None,
                       learning_rate_start   = None,
                       learning_rate_end     = None,
                       neighborhood          = None,
                       decay                 = None,
                       topology              = None,
                       distance              = None,
                       random_seed           = None):
        
        # class params determined at instantiation
        self.som_shape = som_shape
        self.data_size = data_size
        self.max_iter = max_iter
        self.n_epoch = n_epoch
        self.sigma_start = sigma_start
        self.sigma_end = sigma_end
        self.learning_rate_start = learning_rate_start
        self.learning_rate_end = learning_rate_end
        self.neighborhood = neighborhood
        self.decay = decay
        self.topology = topology
        self.random_seed = random_seed
        self.distance = distance
        
        # init values and check if user gave ones
        self._param_init_and_error_checking()
        
        # prepare other values
        nx, ny = self.som_shape
        self._x = np.arange(nx)
        self._y = np.arange(ny)
        # node 2d coords saved as nx*ny rows x 2 columns
        self._nodes_xy = np.vstack([
                            np.meshgrid(self._x, self._y)
                                   ]).reshape(2,nx*ny).T
        self._random_generator = np.random.RandomState(self.random_seed)
        self._weights = \
            self._random_generator.normal(
                    loc   = 0,
                    scale = 0.1,
                    size  = nx*ny*self.data_size
                                         ).reshape( (nx,ny,self.data_size) )
            
        # prepare functions needed based on params
        self._decay_function = Decay.get_decay_function(self.decay)
        self._distance_function = Distance.get_distance_function(self.distance)
        self._neighborhood_function = \
                Neighborhood.get_neighborhood_function(self.neighborhood)
        
        
        
        
        
    def _param_init_and_error_checking(self):
        
        """
        Initialize many class params given as None and check the various
        errors that could occur with class instantiation
        """
        
        
        ## Set som_shape and ensure SOM has shape 2
        if self.som_shape is None:
            self.som_shape = (5,5)
        if len(self.som_shape) != 2:
            raise ValueError('Can only create SOM of dimension 2')
        if (self.som_shape[0] < 0) or (self.som_shape[1] < 0):
            raise ValueError("Must provide positive int for x and y nodes")
            
            
        ## Determine data_size
        if self.data_size is None:
            self.data_size = 2
        if not newb_som._is_int_type(self.data_size):
            raise ValueError('data_size has to be an integer')
            
        
        ## Determine max_iter
        if self.max_iter is None:
            self.max_iter = 100
        if not newb_som._is_int_type(self.max_iter):
            raise ValueError('max_iter has to be an integer')
            
        
        ## Determine n_epoch
        if self.n_epoch is None:
            self.n_epoch = 1
        if not newb_som._is_int_type(self.n_epoch):
            raise ValueError('n_epoch has to be an integer')
            
        
        ## Determine sigma start and end
        # sigma start
        if self.sigma_start is None:
            self.sigma_start = 1  ### modify this to be rel to som shape
        if not newb_som._is_number_type(self.sigma_start):
            raise ValueError('sigma_start has to be a number')
        # sigma end
        if self.sigma_end is None:
            self.sigma_end = 0.01
        if not newb_som._is_number_type(self.sigma_end):
            raise ValueError('sigma_end has to be a number')
        # confirm sigma_end < sigma_start
        if self.sigma_start < self.sigma_end:
            raise ValueError("sigma_end must be < sigma_start, but have:\n"
                             + "sigma_start="+str(self.sigma_start) + " and "
                             + "sigma_end="+str(self.sigma_end)) 
            
            
        ## Determine learning rate start and end
        # learning rate start
        if self.learning_rate_start is None:
            self.learning_rate_start = 0.5
        if not newb_som._is_number_type(self.learning_rate_start):
            raise ValueError('learning_rate_start has to be a number')
        # learning rate end
        if self.learning_rate_end is None:
            self.learning_rate_end = 0.01
        if not newb_som._is_number_type(self.learning_rate_end):
            raise ValueError('learning_rate_end has to be a number')
        # confirm learning_rate_start < learning_rate_end
        if self.learning_rate_start < self.learning_rate_end:
            raise ValueError(
                "learning_rate_end must be < learning_rate_start, but have:\n"
                + "learning_rate_start="+str(self.learning_rate_start) + " and "
                + "learning_rate_end="+str(self.learning_rate_end)
                            )
    
    
        ## Determine neighborhood function
        if self.neighborhood is None:
            self.neighborhood = _GAUSSIAN
        if not newb_som._is_str_type(self.neighborhood):
            raise ValueError("neighborhood must be of type str")
        if self.neighborhood not in _ALLOWED_NEIGHBORHOOD:
            raise ValueError("neighborhood must be one of "
                             + str(_ALLOWED_NEIGHBORHOOD))
        
        
        ## Determine decay function
        if self.decay is None:
            self.decay = _LINEAR
        if not newb_som._is_str_type(self.decay):
            raise ValueError("decay must be of type str")
        if self.decay not in _ALLOWED_DECAY:
            raise ValueError("decay must be one of " + str(_ALLOWED_DECAY))
            
        
        ## Determine topology
        if self.topology is None:
            self.topology = _RECTANGULAR
        if not newb_som._is_str_type(self.topology):
            raise ValueError("topology must be of type str")
        if self.topology not in _ALLOWED_TOPOLOGY:
            raise ValueError("topology must be one of "
                             + str(_ALLOWED_TOPOLOGY))
            
            
        ## Determine random seed
        # keep random seed as None
        if self.random_seed is not None:
            if not newb_som._is_int_type(self.random_seed):
                raise ValueError("When providing random_seed, it must be int")
                
        
        ## Determine distance metric
        if self.distance is None:
            self.distance = _EUCLIDEAN
        if not newb_som._is_str_type(self.distance):
            raise ValueError("distance must be of type str")
        if self.distance not in _ALLOWED_METRIC:
            raise ValueError("distance must be one of " + str(_ALLOWED_METRIC))
    
    
    
    
        
    def _is_int_type(val):
        """
        Returns true if val is of integer type - val should be NUMBER and
        not TYPE!
        """
        return np.issubdtype( np.integer, type(val) ) \
               or isinstance(val, int)
    
    
    
    
    
    def _is_number_type(val):
        """
        Returns true if val is of number type - val should be NUMBER and
        not TYPE!
        """
        return np.issubdtype( np.number, type(val) ) \
               or isinstance(val, (int,float))
    
    
    
    
    
    def _is_str_type(val):
        """
        Returns true if val is of str type, either raw Python or numpy -
        val should be variable and not TYPE!
        """
        return isinstance(val, (np.str_,str))
  
    
  
    
  
    def _compute_weight_change_from_data_i(self, alpha_t,
                                                 neighborhood_vals,
                                                 data_vec):
        """
        Computes the changes in weights according to the classic neural-net
        formula ...
        delta_weights = learning_rate * (metric for correctness).
        The specific formula here is ...
        delta_weights = learning_rate
                            * distance-proximity to BMU (neighborhood_vals)
                            * difference b/w data and weight vectors
        
        Parameters
        ----------
        alpha_t: number
            Learning rate at iteration t
        neighborhood_vals: 2d numpy array
            Neighborhood values for all neurons relative to winning neuron
        data_vec: 1d numpy array
            Vector of data to measure node distances relative to
            
        Returns
        -------
        None
        """
        # first need to broadcost (e.g. expand dimensionality) of
        # neighborhood values; each weight vector receives a particular
        # scalar neighborhood value, so expand along last axis of
        # the latter to make it 3d-compliant
        neighborhood_vals_3d = neighborhood_vals[:,:,None]  ##
        # can now matrix multiply correctly
        weight_updates = (alpha_t
                            * neighborhood_vals_3d
                            * (data_vec - self._weights))
        return weight_updates
        # then update
        #self._weights = self._weights + weight_updates
        
        
    
  
        
    def pca_weight_init(self):
        """
        Initialize weights using first two dominant eigenvectors
        
        TODO
        """
        return None
    
    
    
    
    
    def train(self, data, plot_every=None, xy=None):
        """
        Batch-trains SOM using data
        No data shuffling
        
        Parameters
        ----------
        data: 2d numpy array or pandas DataFrame
            data to train the SOM; must be at least 2-dimensional
        plot_every: int, optional
            Scatter plot the data and the SOM nodes every (plot_every)
            number of iterations
        xy: 2-element list of str (or int), optional
            List of strs (if pandas dataframe) or ints (if raw numpy array)
            representing column indices of data to plot. SOM nodes will
            be plotted on top using their weight vectors as "positions"

        Returns
        -------
        None.
        """
        
        _is_df = isinstance(data, pd.core.frame.DataFrame)
        
        
        if xy is None:
            if _is_df:
                xy = list(data)[:2]
            else:
                xy = [0,1]
                
        xy_inds = None        
        if _is_df:
            xy_inds = [ list(data).index(elem) for elem in list(data) if elem in xy ]
        else:
            xy_inds = xy
            
        
        ## also account for epoch!
        
        data_arr = data.values if _is_df else data
        
        
        for t in range(self.max_iter):
            
            # Update neighborhood size and learning rate for current iteration
            sigma_t = self._decay_function(self.sigma_start,
                                           self.sigma_end,
                                           t,
                                           self.max_iter)
            alpha_t = self._decay_function(self.learning_rate_start,
                                           self.learning_rate_end,
                                           t,
                                           self.max_iter)
            
            # check if plotting
            if (plot_every is not None) and (t % plot_every == 0):
                
                # plot data
                if _is_df:
                    plt.scatter( *data[xy].values.T, s=1 )
                else:
                    plt.scatter( *data[:,xy].T )
                
                # plot nodes on data
                node_posits = self._weights[:,:,xy_inds].reshape(
                                                    np.prod(self.som_shape),
                                                    2
                                                                )
                plt.scatter(*node_posits.T,
                            s=100,
                            marker='o',
                            facecolors='None',
                            edgecolors='black')
                plt.title('Iteration '+str(t)+' out of '+str(self.max_iter))
                plt.show()
                plt.close()
                    
                
            
            # make array with same shape as weights for storing
            # ongoing weight updates
            new_weights = np.zeros( self._weights.shape )
            
            # compute new weights from data for each data vector
            for q in range(data_arr.shape[0]):
                
                data_vec = data_arr[q]
                bmu = self.get_BMU(data_vec)
                neighborhood_vals = self.compute_neighborhood(bmu,
                                                              sigma_t)
                #print(bmu, neighborhood_vals)
                # compute weight change *due only to current data vector!*
                new_weights = new_weights + \
                       self._compute_weight_change_from_data_i(
                                            alpha_t,
                                            neighborhood_vals,
                                            data_vec
                                                              )
            
            # once all data has been processed for this iteration,
            # update weights
            self._weights = self._weights + new_weights
            
    
    
    
    
    
    
    
    
    
    
    def get_BMU(self, data_vector):
        """
        Return the 2-element index of the best matching unit in the SOM
        
        Parameters
        ----------
        data_vector: 1d numpy array
            Array to compute distances against for all nodes
        
        Returns
        -------
        2-element tuple of ints
        """
        
        distances = self._distance_function(data_vector, self._weights)
        #distances = self._compute_distances(data_vector)
        # numpy min returns index of lowest value in *flattened* array
        # have to use unravel_index along with original shape to
        # get 2d index out
        return np.unravel_index( np.argmin(distances), distances.shape )
    
    
    
    
    
    def compute_neighborhood(self, bmu_index, sigma_t):
        """
        Compute the neighborhood function for all nodes relative to the
        winning neuron given by bmu_index
        
        Parameters
        ----------
        bmu_index: 2-element list / tuple / numpy array
            2d indices of BMU
        
        Returns
        -------
        2d numpy array
            neighborhood values for each neuron
        """
        # distances computed in this way records them into a 1d numpy array
        # from bmu_index to the nodes at (0,0), (0,1), ..., (1,0), (1,1), ... etc
        # need to reshape this to original shape of (x_node,y_node) - a 2d
        # array - and TRANSPOSE it (because of how the axis was collapsed
        # in the distance function)
        node_distances = self._distance_function(bmu_index, self._nodes_xy)
        return self._neighborhood_function(
                        node_distances.reshape( self.som_shape ).T,
                        sigma_t
                                          )
            
        