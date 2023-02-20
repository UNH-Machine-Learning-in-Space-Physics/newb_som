# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 10:10:25 2023

@author: edmon
"""

import matplotlib.pyplot as plt
from newb_som import newb_som
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import seaborn as sns
from sklearn.datasets import make_moons
from sklearn.datasets import make_s_curve
from matplotlib import animation




def make_mv_gauss(num_gauss    = None,
                  n_per_gauss  = None,
                  means        = None,
                  covmats      = None,
                  seed         = None,
                  single_array = None):
    if num_gauss is None:
        num_gauss = 1
    if n_per_gauss is None:
        n_per_gauss = 10
    if np.issubdtype(type(n_per_gauss),np.number):
        n_per_gauss = [ int(n_per_gauss) for i in range(num_gauss) ]
    if means is None:
        means = [ i * np.array([1,1]) for i in range(num_gauss) ]
    if not isinstance(means, list):
        means = [ means ]
    if covmats is None:
        covmats = [ (2/(i+2)) * np.identity( means[0].shape[0] )  \
                    for i in range(num_gauss) ]
    if not isinstance(covmats, list):
        covmats = [ covmats ]
    if single_array is None:
        single_array = False
    
    
    rng = np.random.RandomState(seed)
    
    ## generate data
    data = []
    for i in range(num_gauss):
        data.append(
            rng.multivariate_normal(
                    means[i],
                    covmats[i],
                    size=n_per_gauss[i]
                                    )
                    )
    
    ## if not single_array, source gaussian can be found by index in
    ## returned list
    if not single_array:
        return data
    
    ## if compressing all data into one array, need addition 1d array
    ## indicating source gaussian
    else:
        data = np.vstack(data)
        source = np.hstack(
                [ np.full(n_per_gauss[i],i) for i in range(num_gauss) ]
                          )
        return data, source






def make_unif_data(n=None, dim_bounds=None, seed=None, return_src=None):
    if n is None: n = 30
    if dim_bounds is None: dim_bounds = [ [0,1], [0,1] ]
    if return_src is None: return_src = False
    
    rng = np.random.RandomState(seed)
    
    data = np.vstack( [ rng.random(n) for i in range(len(dim_bounds)) ] ).T
    
    for i in range(len(dim_bounds)):
        _min, _max = dim_bounds[i]
        data[:,i] = _min + (_max - _min) * data[:,i]
    
    if not return_src:
        return data
    else:
        return data, np.full(data.shape[0],0)








def prep_data(data, fit=None):
    # if no fit, then do nothing
    if fit is None:
        return data
    
    # otherwise, standardize or normalize
    scaler = None
    if fit == 'standardize':
        scaler = StandardScaler()
    if fit == 'normalize':
        scaler = MinMaxScaler()
    return scaler.fit_transform(data)







def quantization_error_hist(som, data, orig_data, func=None, outliers=None):
    """
    Shows a boxenplot + histogram of the quantization errors
    from the som and the data

    Parameters
    ----------
    som : Instance of newb_som
    data : 2d numpy array / dataframe 
        data used to train the som
    orig_data: 2d numpy array / dataframe
        original data, BEFORE the transformation
    func: function (default = np.mean)
        Function that returns a scalar value
        Data exceeding this value will be indicated in the plot
    outliers: bool (default=False)
        If True, will make scatter plot with inliers / outliers plotted
        separately. If False, colormap of distance will be used.

    Returns
    -------
    None
    """
    if outliers is None:
        outliers = False
    
    ## get quantization data
    quants = som.full_quantization_error(data)
    avg_quant = np.mean(quants)
    if func is None:
        func = np.mean
    func_val = func(quants)
    
    ## make plots
    fig = plt.figure()
    box_ax = plt.subplot2grid((2,4), loc=(0,0), colspan=2)
    hist_ax = plt.subplot2grid((2,4), loc=(1,0), colspan=2)
    scat_ax = plt.subplot2grid((2,4), loc=(0,2), colspan=2, rowspan=2)
    
    mean_kwargs = {'linestyle':'solid',
                   'lw':2.5,
                   'c':'black'}
    func_kwargs = {'linestyle':'dashed',
                   'lw':2.5,
                   'c':'black'}
    
    ## make boxenplot with vert lines
    box_ax.set_title('Boxenplot QE')
    sns.boxenplot(quants, orient='h', ax=box_ax)
    box_ax.axvline(x=avg_quant, **mean_kwargs)
    box_ax.axvline(x=func_val, **func_kwargs)
    
    ## make log-scale hist
    hist_ax.set_title('Log-hist QE')
    hist_ax.hist(quants, bins=50, log=True)
    hist_ax.axvline(x=avg_quant, **mean_kwargs)
    hist_ax.axvline(x=func_val, **func_kwargs)
    
    ## plot data beyond func_value
    if outliers:
        _out = np.where( quants > func_val )[0]
        _in = np.setdiff1d( np.arange(data.shape[0]), _out )
        scat_ax.scatter( orig_data[_in,0], orig_data[_in,1], s=2.5, c='blue')
        scat_ax.scatter( orig_data[_out,0], orig_data[_out,1], s=2.5, c='red')
        scat_ax.set_title('2D data - inliers blue / outliers red')
    
    else:
        res = scat_ax.scatter( orig_data[:,0], orig_data[:,1],
                              cmap='plasma', c=quants, s=2.5)
        scat_ax.set_title('2D data - avg distance ')
        fig.colorbar(res, ax=scat_ax)
    
    fig.subplots_adjust(hspace=0.4, wspace=0.57)
    fig.suptitle('[BMU(data_point) - data_point] for all data (Quantization Error)')
    







def plot_pca(dat):
    """
    Plots the top two principal components and the PCA-rotated data
    
    Parameters
    ----------
    dat: 2d numpy array
        2d array of data (features as columns)
        
    Returns
    -------
    None
    """
    
    # Make subplots and plot raw data with pca directions
    fig, (dat_ax, pca_ax) = plt.subplots(1,2)
    dat_ax.scatter(dat[:,0], dat[:,1], s=1)
    
    # Get eigenvalues/vectors (in order of largest to smallest)
    evals, evecs = np.linalg.eig( np.cov( dat.T ) )
    sort_inds = np.argsort( evals )[::-1]
    sort_evals = evals[sort_inds]
    sort_evecs = evecs.T[sort_inds].T
    
    # scale eigenvalues to some range based on plot limits
    x_range, y_range = np.vstack( [ dat.min(axis=0), dat.max(axis=0) ] ).T
    center_pt = np.mean( [ dat_ax.get_xlim(), dat_ax.get_ylim() ], axis=1 )
    max_arrow_length = np.sqrt(np.sum( 
                            (np.diff( [ x_range, y_range ] ) / 2)**2
                                     )) / 3
    scaled_evals = evals * max_arrow_length / np.max(evals)
    evec_angles = np.array( [ np.arctan(vec[1]/vec[0]) for vec in evecs ] )
    
    # compute and plot each arrow length based on angle of eigvector
    for i in range(evec_angles.shape[0]):
        arrow_end = np.array([
                    center_pt[0] + scaled_evals[i] * np.cos( evec_angles[i] ),
                    center_pt[1] + scaled_evals[i] * np.sin( evec_angles[i] )
                            ])
        dat_ax.arrow(*center_pt, *(arrow_end - center_pt)/(i+1),
                     width = max_arrow_length / 20,
                     facecolor='black',
                     edgecolor='None')
        
    pca_dat = np.matmul(dat,evecs.T)
    pca_ax.scatter( *pca_dat.T, s=1 )
    
    plt.show()
    plt.close()
    
    return evals, evecs, pca_dat
    







def rotate_data(data, angle_degrees=None):
    """
    Rotate the 2d data about an angle

    Parameters
    ----------
    data: 2d numpy array
    angle_degrees: float
        Angle to rotate data about by (in degrees)
        
    Returns
    -------
    2d numpy array (same dims as data)
    """
    
    if angle_degrees is None:
        angle_degrees = 45
    
    ang_rad = angle_degrees * np.pi/180
    rot = np.array([ [np.cos(ang_rad), -np.sin(ang_rad)],
                     [np.sin(ang_rad), np.cos(ang_rad)]  ])
    return np.matmul(data,rot)







def combine_datasets(dat_list, src_list):
    """
    Combine datasets into single 2d numpy array with indication
    of what datapoint belongs to what set
    
    Parameters
    ----------
    dat_list: list of 2d numpy arrays
        list of datasets
    src_list: list of 1d numpy arrays
        list of 1d integer array indicating what dataset a point belongs to
        
    Returns
    -------
    2d numpy arr (combined dataset), 1d numpy arr (combined source arr)
    """
    
    combined_dat = np.vstack( dat_list )
    src_ints = [ np.unique(arr) for arr in src_list ]
    for i in range(1,len(src_ints)):
        prev_max_int = np.max( src_list[i-1] ) 
        src_list[i] = src_list[i] + prev_max_int + 1
    combined_src = np.hstack( src_list )
    
    return combined_dat, combined_src











def make_ugly_dataset(pts_per_dat   = None,
                      num_gauss     = None,
                      n_per_gauss   = None,
                      means         = None,
                      covmats       = None,
                      seed          = None,
                      frac_unif     = None,
                      moon_noise    = None,
                      s_curve_noise = None,
                      angle_rot     = None):
    """
    Combine gaussians, s-curve, moons, and uniform noise into 
    single dataset
    
    Any necessary params will be inferred 
    """
    
    if seed is None:
        seed = 1
    if num_gauss is None:
        num_gauss = 3
    if pts_per_dat is None:
        pts_per_dat = 100
    if moon_noise is None:
        moon_noise = 0.05
    if s_curve_noise is None:
        s_curve_noise = 0.1
    if frac_unif is None:
        frac_unif = 0.05
    if means is None:
        means = [ np.array([0,0]),
                  np.array([10,0]),
                  np.array([0,10]) ]
    if covmats is None:
        scale = 0.9
        corr = 0.999
        covmats = [ np.identity(2) * scale,
                    np.array([[1,corr],[corr,1]]) * scale,
                    np.array([[1,-corr],[-corr,1]]) * scale ]
    if angle_rot is None:
        angle_rot = 45
        
    
    g_dat, g_src = make_mv_gauss(num_gauss    = num_gauss,
                                 n_per_gauss  = int(pts_per_dat)/num_gauss,
                                 means        = means,
                                 covmats      = covmats,
                                 seed         = seed,
                                 single_array = True)
    

    ## make moon data
    m_dat, m_src = make_moons(n_samples    = pts_per_dat,
                              noise        = moon_noise,
                              random_state = seed)
    # shift it
    m_dat[:,1] = 1/(1 + m_dat[:,1] - m_dat[:,1].mean())
    m_dat = m_dat + np.array([10,5])
    
    
    ## make s-curve data (but only in 2d)
    s_dat, _ = make_s_curve(n_samples    = pts_per_dat,
                            noise        = s_curve_noise,
                            random_state = seed)
    s_dat = s_dat[:,[0,2]]
    s_src = np.full( s_dat.shape[0], 0 )
    s_dat = s_dat + np.array([5,5])
    
    
    ## combine datasets
    dat, src = combine_datasets( [g_dat, m_dat, s_dat],
                                 [g_src, m_src, s_src] )
    
    
    ## add in uniform random data
    bounds = np.array( [ dat.min(axis=0), dat.max(axis=0) ] ).T
    u_dat, u_src = make_unif_data(n          = int(dat.shape[0] * frac_unif),
                                  dim_bounds = bounds,
                                  seed       = seed,
                                  return_src = True)
    
    
    ## combine datasets (again, but with unif)
    dat, src = combine_datasets( [dat, u_dat],
                                 [src, u_src] )
    
    
    # rotate data
    dat = rotate_data(dat,
                      angle_degrees = angle_rot)
    
    
    ## shuffle
    inds = np.random.permutation( dat.shape[0] )
    dat = dat[inds]
    src = src[inds]
    
    
    return dat, src
    
        









if __name__ == "__main__":
    
    
    
    rng_seed = 1
    dat, src = make_ugly_dataset(pts_per_dat = 50*1000,
                                 seed        = rng_seed,
                                 angle_rot   = 0)
    
   
    
    
    
    
    
    ## som params
    som_shape = (20,20)
    max_iter = 1000
    sigma_start = 3.0
    sigma_end = 0.1
    learning_rate_start = 0.1
    learning_rate_end = 0.01
    decay_type = 'exponential'
    distance_type = 'euclidean'
    data_fit = 'standardize'
    plot_every = 100
    weight_init = 'linspace'
    animate = False
    
    
    
    model_data = prep_data(dat, fit=data_fit)
    
    plot_pca( model_data ) 
    
    
    
    ## instantiate som
    som = newb_som(som_shape = som_shape,
                   max_iter = max_iter,
                   data_size = len(dat[0]),
                   learning_rate_start = learning_rate_start,
                   learning_rate_end = learning_rate_end,
                   sigma_start = sigma_start,
                   sigma_end = sigma_end,
                   decay = decay_type,
                   distance = distance_type,
                   seed = rng_seed)
    
    
    cam = som.train(model_data,
                     plot_every=plot_every,
                     weight_init=weight_init,
                     all_data_per_iter=False,
                     show_neighborhood=False,
                     animate=animate)
    print('Done training!')
    if animate:
        anim = cam.animate()
        writergif = animation.PillowWriter(fps=30)
        #anim.save('/home/jedmond/Desktop/som.mp4', fps=30)
        loc =  r"C:\Users\edmon\OneDrive\Desktop\som.gif"
        anim.save(loc, writer=writergif)
        print('Made movie!')
    plt.close()
    
    
    # show quant error stats
    quantization_error_hist(som, model_data, dat,
                            outliers=False)
    quantization_error_hist(som, model_data, dat,
                            func= lambda x: np.percentile(x,99),
                            outliers=True)




