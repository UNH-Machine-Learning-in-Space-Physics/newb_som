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
                    size=n_per_gauss
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
                [ np.full(n_per_gauss,i) for i in range(num_gauss) ]
                          )
        return data, source






def make_unif_data(n=None, dim_bounds=None, seed=None):
    if n is None: n = 30
    if dim_bounds is None: dim_bounds = [ [0,1], [0,1] ]
    
    rng = np.random.RandomState(seed)
    
    data = np.vstack( [ rng.random(n) for i in range(len(dim_bounds)) ] ).T
    
    for i in range(len(dim_bounds)):
        _min, _max = dim_bounds[i]
        data[:,i] = _min + (_max - _min) * data[:,i]
        
    return data






def add_unif_data(dat, src, frac_unif=None, seed=None):
    if frac_unif is None:
        frac_unif = 0.2
    
    
    mins = dat.min(axis=0)
    maxs = dat.max(axis=0)
    bounds = [ (mins[i],maxs[i]) for i in range(dat.shape[1]) ]
    
    unif_dat = make_unif_data(n          = int(frac_unif * dat.shape[0]),
                              dim_bounds = bounds,
                              seed       = rng_seed)
    
    unif_src = np.full(unif_dat.shape[0], np.max(src)+1)
    
    return np.vstack( [ dat, unif_dat ] ), np.hstack( [ src, unif_src ] )
    





def make_noisy_gaussians(num_gauss   = None,
                         num_points  = None,
                         means       = None,
                         covmats     = None,
                         seed        = None,
                         unif_frac   = None):
    
    ## make gaussian data
    n_per_gauss = int( total_pts * (1 - frac_unif) / num_gauss )
    dat, g_src = make_mv_gauss(num_gauss=num_gauss,
                               means=means,
                               covmats = covmats,
                               n_per_gauss=n_per_gauss,
                               seed=rng_seed,
                               single_array=True)

    ## add uniform data
    dat, src = add_unif_data(dat, g_src, frac_unif=unif_frac, seed=rng_seed)
    
    return dat, src






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
    
    
    
        









if __name__ == "__main__":
    
    
    ### TO dO
    ### Later, try make_moons sklearn dataset
    
    
    ## params for data generation
    rng_seed = 1
    total_pts = 100*1000
    num_gauss = 3
    frac_unif = 0.3
    means = [ np.array([0,0]),
              np.array([10,0]),
              np.array([0,10]) ]
    scale = 0.9
    corr = 0.999
    covmats = [ np.identity(2) * scale,
                np.array([[1,corr],[corr,1]]) * scale,
                np.array([[1,-corr],[-corr,1]]) * scale ]
    
    """
    dat, src = make_noisy_gaussians(num_gauss  = num_gauss,
                                    num_points = total_pts,
                                    means      = means,
                                    covmats    = covmats,
                                    seed       = rng_seed,
                                    unif_frac  = frac_unif)
    """
    
    dat, src = make_moons(n_samples=total_pts)
    dat, src = add_unif_data(dat,
                             src,
                             frac_unif=frac_unif,
                             seed=rng_seed)
    
    ## showcase
    plt.scatter(dat[:,0], dat[:,1], s=2)
    plt.show()
    plt.close()
    
    ## shuffle
    inds = np.random.permutation( dat.shape[0] )
    dat = dat[inds]
    src = src[inds]
    
    
    
    
    ## som params
    som_shape = (40,40)
    max_iter = 500
    sigma_start = 3.0
    sigma_end = 0.1
    learning_rate_start = 0.1
    learning_rate_end = 0.01
    decay_type = 'exponential'
    distance_type = 'euclidean'
    data_fit = None
    
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
    
    model_data = prep_data(dat,
                           fit=data_fit)
    som.train(model_data,
              plot_every=30,
              weight_init='linspace',
              all_data_per_iter=False,
              show_neighborhood=False)
    
    
    # show quant error stats
    quantization_error_hist(som, model_data, dat,
                            outliers=False)
    quantization_error_hist(som, model_data, dat,
                            func= lambda x: np.percentile(x,99),
                            outliers=True)




