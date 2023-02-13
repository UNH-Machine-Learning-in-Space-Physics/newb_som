#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 10:27:23 2023

@author: jedmond
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#import cv2
import os
from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle






_RECTANGULAR = 'rectangular'
_ALLOWED_TOPOLOGY = [ _RECTANGULAR ]

_GAUSSIAN = 'gaussian'
_ALLOWED_NEIGHBORHOOD = [ _GAUSSIAN ]

_LINEAR = 'linear'
_EXPONENTIAL = 'exponential'
_ALLOWED_DECAY = [ _LINEAR, _EXPONENTIAL ]

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
    
    
    
    
    
    def _get_exponential_decay(start_val, end_val, current_iter, max_iter):
        """
        Exponential decay function, of form ...
        val(t) = val_0 * exp( -current_iter * c )
        (where c is a constant determined by stipulating the starting and
         ending values)
        
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
        
        if end_val == 0:
            c = -np.log(0.01) / max_iter
        else:
            c = -np.log(end_val / start_val) / max_iter
        return start_val * np.exp(-current_iter * c)
    
    
        
    
    def get_decay_function(decay_type):
        """
        Retrieves the desired decay function 
        """
        func = None
        if decay_type == _LINEAR:
            func = Decay._get_linear_decay
        if decay_type == _EXPONENTIAL:
            func = Decay._get_exponential_decay
            
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
        """
        self._weights = \
            self._random_generator.normal(
                    loc   = 0.5,
                    scale = 0.1,
                    size  = nx*ny*self.data_size
                                         ).reshape( (nx,ny,self.data_size) )
        """
        self._weights = self._random_generator.rand(nx, ny, self.data_size)
            
        # prepare functions needed based on params
        self._decay_function = Decay.get_decay_function(self.decay)
        self._distance_function = Distance.get_distance_function(self.distance)
        self._neighborhood_function = \
                Neighborhood.get_neighborhood_function(self.neighborhood)
                
        # create array connecting adjacent nodes
        #self._node_lines = self._make_vertex_array()
        
        
        
        
        
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
    
    
    
    
    
    def project_nodes(self, ax, xy_inds, sigma_t, show_neighborhood=None):
        """
        Project nodes onto the feature space of data given by the
        column indices xy
        
        Parameters
        ----------
        ax: Matplotlib axis instance
            axis to plot data onto
        xy: 2-element list / arr of integers
            The column indices of data we're project onto
            
        Returns
        -------
        None
        """
        if show_neighborhood is None:
            show_neighborhood = False
        
        node_proj_posits = self._weights[:,:,xy_inds].reshape(
                                            np.prod(self.som_shape),
                                            2
                                                             )
        
        ## numpy is very ... annoying in trying to convert 2d inds
        ## to flattened 1d inds (multi_ravel_index won't do it),
        ## so next best best is to take x and y posits, reshape them
        ## based on node coords, take their transpose, and use that
        ## instead ...
        x = node_proj_posits[:,0].reshape( self.som_shape ).T
        y = node_proj_posits[:,1].reshape( self.som_shape ).T
        
        # get dict showing starting and ending location of
        # lines b/w adjacent nodes
        node_dict = self._make_node_graph()
        
        x_posit, y_posit = [], []
        for starting_node in node_dict:
            for ending_node in node_dict[starting_node]:
                x_posit.append( ( x[starting_node], x[ending_node] ) )
                y_posit.append( ( y[starting_node], y[ending_node] ) )
        
        
        ## finally make plot of nodes ...
        ax.scatter(*node_proj_posits.T,
                    s=100,
                    marker='o',
                    facecolors='None',
                    edgecolors='black')
        # and node-adjacent lines
        ax.plot(np.vstack(x_posit).T,
                np.vstack(y_posit).T,
                'black',
                linewidth=0.5)
        
        
        ## and if specified, draw neighborhood regions
        if show_neighborhood:
            
            # IF Gaussian, neighborhood *technically* spans all space,
            # but we're using isotropic 2d gaussians so we'll just plot
            # the 95% confidence circle (circle since isotropic!)
            # 95% of chi square of degree 2 --> (x/sig_x)**2 + (y/sig_y)**2 = 5.991
            # taking x=y and sig_x = sig_y = sigma_t --> radius = sqrt(5.991/2)*sigma_t
            if self.neighborhood == _GAUSSIAN:
                radius = np.sqrt(5.991/2) * sigma_t
                # create list of individual circle objects with
                # center on nodes
                neigh_patches = []
                for coord in node_proj_posits:
                    neigh_patches.append( 
                        Circle(coord, radius=radius, color='green', fill=True,
                               alpha=0.3)
                                        )
                # turn list into patch collection and plot
                patch_coll = PatchCollection(neigh_patches, match_original=True)
                ax.add_collection(patch_coll)
        
    
    
    
    
    def _make_node_graph(self, xy_array=None):
        """
        Makes list of tuples indicating which nodes are connected adjacently.
        
        Note: Matplotlib plot() function will connect lines between all
              consecutive points in a given array; to only plot the lines
              given (with no extra lines between consecutive points), the
              x, y arrays need to be given as 2 x N arrays (so the transpose
              of what's returned here!)
        """
        
        if xy_array is None:
            xy_array = False
        
        ## Make dict with keys being node coordinates and values being
        ## list of (rectangular) adjacent nodes
        line_dict = {}
        for node in self._nodes_xy:
            tuple_node = tuple(node.tolist())
            line_dict[tuple_node] = []
            # in rectangular, node (x,y) could only be adjacent to node
            # that +/- 1 components to it x or y value
            possible_nodes = [
                ( tuple_node[0]-1, tuple_node[1] ),
                ( tuple_node[0], tuple_node[1]-1 ),
                ( tuple_node[0]+1, tuple_node[1] ),
                ( tuple_node[0], tuple_node[1]+1 )
                              ]
            # eliminate possible nodes that go beyond bounds
            for possible_node in possible_nodes:
                if ((min(possible_node) < 0) 
                       or (possible_node[0] == self.som_shape[0]) 
                       or (possible_node[1] == self.som_shape[1])):
                    continue
                line_dict[tuple_node].append( possible_node )
        
        
        if not xy_array:
            return line_dict
        
        ## If array is desired, convert from node dict to array
        x, y = [], []
        for starting_pt in line_dict:
            for ending_pt in line_dict[starting_pt]:
                x.append( (starting_pt[0], ending_pt[0]) )
                y.append( (starting_pt[1], ending_pt[1]) )
    
                
        return np.vstack(x), np.vstack(y)
        
        
        
    
    
    def train(self, data, plot_every=None, xy=None, save_movie=None):
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
        save_movie: str, optional
            If given folder address as str, will save movie of plots made
            at address

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
        
        
        sigma_vals = []
        alpha_vals = []
        for t in range(self.max_iter):
            
            # Update neighborhood size and learning rate for current iteration
            sigma_t = self._decay_function(self.sigma_start,
                                           self.sigma_end,
                                           t,
                                           self.max_iter)
            sigma_vals.append( sigma_t )
            alpha_t = self._decay_function(self.learning_rate_start,
                                           self.learning_rate_end,
                                           t,
                                           self.max_iter)
            alpha_vals.append( alpha_t )
            
            # check if plotting
            if (plot_every is not None) and (t % plot_every == 0):
                
                fig, axes = plt.subplots(2,2)
                data_ax = axes[0,0]
                decay_ax = axes[1,0]
                neigh_ax = axes[0,1]
                
                
                # plot data
                if _is_df:
                    data_ax.scatter( *data[xy].values.T, s=1 )
                    x_data = data[xy].values[:,0]
                    y_data = data[xy].values[:,1]
                else:
                    data_ax.scatter( *data[:,xy].T, s=1 )
                    x_data = data[:,xy[0]]
                    y_data = data[:,xy[1]]
                
                # sometimes nodes get pushed faaaar away, so need to constrain
                # window view to data
                x_bounds = [x_data.min(),x_data.max()]
                delta_x = np.diff(x_bounds) * 0.05
                x_bounds = [x_bounds[0] - delta_x, x_bounds[1] + delta_x]
                data_ax.set_xlim(x_bounds)
                y_bounds = [y_data.min(),y_data.max()]
                delta_y = np.diff(y_bounds) * 0.05
                y_bounds = [y_bounds[0] - delta_y, y_bounds[1] + delta_y]
                data_ax.set_ylim(y_bounds)
                data_ax.set_title('data')
                
                    
                self.project_nodes(data_ax, xy_inds, sigma_t,
                                   show_neighborhood=True)
                #data_ax.set_title('Iteration '+str(t+1)+' out of '+str(self.max_iter))
                
                
                # plot sigma and learning rate
                decay_ax.plot(np.arange(t+1),sigma_vals,lw=5)
                decay_ax.set_ylim( [ self.sigma_end, self.sigma_start ] )
                decay_ax.set_xlim( [ 0, self.max_iter ] )
                decay_ax.set_title('sigma(t)')
                
                
                # plot neigh func
                xdata_neigh = np.linspace(-2,2)
                neigh_ax.plot(xdata_neigh,
                              self._neighborhood_function(xdata_neigh,
                                                          self.sigma_start),
                              lw=3,
                              linestyle='dashed')
                neigh_ax.plot(xdata_neigh,
                              self._neighborhood_function(xdata_neigh,
                                                          sigma_t),
                              lw=3,
                              linestyle='solid')
                neigh_ax.set_title('Neighborhood function ')
                
                
                # set title
                fig.suptitle('Iteration '+str(t+1)+' out of '+str(self.max_iter))
                plt.subplots_adjust(hspace=0.35)
                
                
                plt.show()
                plt.close()
                
                """# plot nodes on data
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
                plt.close()"""
                    
                
            
            # make array with same shape as weights for storing
            # ongoing weight updates
            new_weights = np.zeros( self._weights.shape )
            
            # compute new weights from data for each data vector
            for q in range(data_arr.shape[0]):
                
                data_vec = data_arr[q]
                bmu = self.get_BMU(data_vec)
                neighborhood_vals = self.compute_neighborhood(bmu,
                                                              sigma_t)
                #print(neighborhood_vals)
                #if q == 5:
                #    raise ValueError
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
            print(t,"\n",new_weights)
            #if t == 5:
            #    raise ValueError
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
            
        