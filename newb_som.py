#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 10:27:23 2023

@author: jedmond
"""


import numpy as np




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
    
    _RECTANGULAR = 'rectangular'
    _ALLOWED_TOPOLOGY = [ _RECTANGULAR ]
    _GAUSSIAN = 'gaussian'
    _ALLOWED_NEIGHBORHOOD = [ _GAUSSIAN ]
    _LINEAR = 'linear'
    _ALLOWED_DECAY = [ _LINEAR ]
    _EUCLIDEAN = 'euclidean'
    _ALLOWED_METRIC = [ _EUCLIDEAN ]
    
    
    
    
    
    def __init__(self, som_shape             = None,
                       data_size             = None,
                       max_iter              = None,
                       n_epoch               = None,
                       sigma_start           = None,
                       sigma_end             = None,
                       learning_rate_start   = None,
                       learning_rate_end     = None,
                       neighborhood_function = None,
                       decay_function        = None,
                       topology              = None,
                       distance_metric       = None,
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
        self.neighborhood_function = neighborhood_function
        self.decay_function = decay_function
        self.topology = topology
        self.random_seed = random_seed
        self.distance_metric = distance_metric
        
        # init values and check if user gave ones
        self._param_init_and_error_checking()
        
        # prepare other values
        nx, ny = self.some_shape
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
        
        
        
        
        
            
    
    
    
    
    
    
    
    def _param_init_and_error_checking(self):
        
        """
        Initialize any class params given as None and check the various
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
            self.data_size = 3
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
        if self.neighborhood_function is None:
            self.neighborhood_function = newb_som._GAUSSIAN
        if not newb_som._is_str_type(self.neighborhood_function):
            raise ValueError("neighborhood_function must be of type str")
        if self.neighborhood_function not in newb_som._ALLOWED_NEIGHBORHOOD:
            raise ValueError("neighborhood_function must be one of "
                             + str(newb_som._ALLOWED_NEIGHBORHOOD))
        
        
        ## Determine decay function
        if self.decay_function is None:
            self.decay_function = newb_som._EXPONENTIAL
        if not newb_som._is_str_type(self.decay_function):
            raise ValueError("decay_function must be of type str")
        if self.decay_function not in newb_som._ALLOWED_DECAY:
            raise ValueError("decay_function must be one of "
                             + str(newb_som._ALLOWED_DECAY))
            
        
        ## Determine topology
        if self.topology is None:
            self.topology = newb_som._RECTANGULAR
        if not newb_som._is_str_type(self.topology):
            raise ValueError("topology must be of type str")
        if self.topology not in newb_som._ALLOWED_TOPOLOGY:
            raise ValueError("topology must be one of "
                             + str(newb_som._ALLOWED_TOPOLOGY))
            
            
        ## Determine random seed
        # keep random seed as None
        if self.random_seed is not None:
            if not newb_som._is_int_type(self.random_seed):
                raise ValueError("When providing random_seed, it must be int")
                
        
        ## Determine distance metric
        if self.distance_metric is None:
            self.distance_metric = newb_som._EUCLIDEAN
        if not newb_som._is_str_type(self.distance_metric):
            raise ValueError("distance_metric must be of type str")
        if self.distance_metric not in newb_som._ALLOWED_METRIC:
            raise ValueError("distance_metric must be one of "
                             + str(newb_som._ALLOWED_METRIC))
    
    
    
    
    
    
    
    
    
    
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
    
    
    
    
    
    
    
    
    
    
    def _get_neighborhood_func(self):
        """
        Sets neighborhood function; requires neighborhood_function to be set
        """
        neigh_func_dict = {
            newb_som._GAUSSIAN : None
            }
    
    
    
    
    
    
    
    
    
    
    def compute_distances(self, data_vector):
        """
        Compute the distances according to the specified topology between
        all nodes and the given data vector
        
        Parameters
        ----------
        data_vector: 1d numpy array
            Array to compute distances against for all nodes
            
        Returns
        -------
        1d numpy array of distances
        """
        
        if self.distance_metric == newb_som._EUCLIDEAN:
            return np.linalg.norm( self._weights - data_vector, axis=2 )
    
    
    
    
    
    
    
    
    
    
    def nodes_within_neighborhood(self, node_index, sigma_t):
        """
        Finds the 2d indices of the nodes within

        Parameters
        ----------
        node_index : 2-integer list / tuple / 1d numpy array
            2d index of node that we're calculating distance to other
            nodes from.
        
        sigma_t: float
            Value that determines if other nodes are within the
            neighborhood of node_index

        Returns
        -------
        2d numpy array of node indices that are in neighborhood
        Q rows x 2 columns where Q is the number of nodes in neighborhood.
        """
        
        nearby_nodes = None
        
        if self.distance_metric == newb_som._EUCLIDEAN:            
            rel_node_distances = np.linalg.norm( self._nodes_xy - node_index,
                                                 axis=1 )
            nearby_nodes = self._nodes_xy[ 
                    np.where( rel_node_distances <= sigma_t )[0]
                                ]
        
        
        return nearby_nodes
        
        
        
    
    
    
    
    
    
    
    
    
    
    def pca_weight_init(self):
        """
        Initialize weights using first two dominant eigenvectors
        
        TODO
        """
        return None
    
    
    
    
    
    
    
    
    
    
    def train(self, data):
        """
        Trains SOM using data
        
        Parameters
        ----------
        data: 2d numpy array or pandas DataFrame
            data to train the SOM; must be at least 2-dimensional

        Returns
        -------
        None.
        """
    
    
    
    
    
    
    
    
    
    
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
        
        distances = self._compute_distances(data_vector)
        # numpy min returns index of lowest value in *flattened* array
        # have to use unravel_index along with original shape to
        # get 2d index out
        return np.unravel_index( np.argmin(distances), distances.shape )