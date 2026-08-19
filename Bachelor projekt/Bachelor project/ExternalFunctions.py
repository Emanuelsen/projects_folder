# coding: utf-8

#Author: Christian Michelsen, NBI, 2018
#        Troels Petersen, NBI, 2019-22
#        Emil Fischer, NBI, 2023
#        Chamilla Terp NBI, 2023
import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit 
from scipy import stats
#from sklearn.mixture import GaussianMixture
#import pandas as pd
from scipy import special                       

def format_value(value, decimals):
    """ 
    Checks the type of a variable and formats it accordingly.
    Floats has 'decimals' number of decimals.
    """
    
    if isinstance(value, (float, np.float)):
        return f'{value:.{decimals}f}'
    elif isinstance(value, (int, np.integer)):
        return f'{value:d}'
    else:
        return f'{value}'


def values_to_string(values, decimals):
    """ 
    Loops over all elements of 'values' and returns list of strings
    with proper formating according to the function 'format_value'. 
    """
    
    res = []
    for value in values:
        if isinstance(value, list):
            tmp = [format_value(val, decimals) for val in value]
            res.append(f'{tmp[0]} +/- {tmp[1]}')
        else:
            res.append(format_value(value, decimals))
    return res


def len_of_longest_string(s):
    """ Returns the length of the longest string in a list of strings """
    return len(max(s, key=len))


def nice_string_output(d, extra_spacing=5, decimals=3):
    """ 
    Takes a dictionary d consisting of names and values to be properly formatted.
    Makes sure that the distance between the names and the values in the printed
    output has a minimum distance of 'extra_spacing'. One can change the number
    of decimals using the 'decimals' keyword.  
    """
    
    names = d.keys()
    max_names = len_of_longest_string(names)
    
    values = values_to_string(d.values(), decimals=decimals)
    max_values = len_of_longest_string(values)
    
    string = ""
    for name, value in zip(names, values):
        spacing = extra_spacing + max_values + max_names - len(name) - 1 
        string += "{name:s} {value:>{spacing}} \n".format(name=name, value=value, spacing=spacing)
    return string[:-2]


def add_text_to_ax(x_coord, y_coord, string, ax, fontsize=12, color='k'):
    """ Shortcut to add text to an ax with proper font. Relative coords."""
    ax.text(x_coord, y_coord, string, family='monospace', fontsize=fontsize,
            transform=ax.transAxes, verticalalignment='top', color=color)
    return None




# =============================================================================
#  Probfit replacement
# =============================================================================

from iminuit.util import make_func_code
from iminuit import describe #, Minuit,

def set_var_if_None(var, x):
    if var is not None:
        return np.array(var)
    else: 
        return np.ones_like(x)
    
def compute_f(f, x, *par):
    
    try:
        return f(x, *par)
    except ValueError:
        return np.array([f(xi, *par) for xi in x])


class Chi2Regression:  # override the class with a better one
        
    def __init__(self, f, x, y, sy=None, weights=None, bound=None):
        
        if bound is not None:
            x = np.array(x)
            y = np.array(y)
            sy = np.array(sy)
            mask = (x >= bound[0]) & (x <= bound[1])
            x  = x[mask]
            y  = y[mask]
            sy = sy[mask]

        self.f = f  # model predicts y for given x
        self.x = np.array(x)
        self.y = np.array(y)
        
        self.sy = set_var_if_None(sy, self.x)
        self.weights = set_var_if_None(weights, self.x)
        self.func_code = make_func_code(describe(self.f)[1:])

    def __call__(self, *par):  # par are a variable number of model parameters
        
        # compute the function value
        f = compute_f(self.f, self.x, *par)
        
        # compute the chi2-value
        chi2 = np.sum(self.weights*(self.y - f)**2/self.sy**2)
        
        return chi2






def simpson38(f, edges, bw, *arg):
    
    yedges = f(edges, *arg)
    left38 = f((2.*edges[1:]+edges[:-1]) / 3., *arg)
    right38 = f((edges[1:]+2.*edges[:-1]) / 3., *arg)
    
    return bw / 8.*( np.sum(yedges)*2.+np.sum(left38+right38)*3. - (yedges[0]+yedges[-1]) ) #simpson3/8


def integrate1d(f, bound, nint, *arg):
    """
    compute 1d integral
    """
    edges = np.linspace(bound[0], bound[1], nint+1)
    bw = edges[1] - edges[0]
    
    return simpson38(f, edges, bw, *arg)



class UnbinnedLH:  # override the class with a better one
    
    def __init__(self, f, data, weights=None, bound=None, badvalue=-100000, extended=False, extended_bound=None, extended_nint=100):
        
        if bound is not None:
            data = np.array(data)
            mask = (data >= bound[0]) & (data <= bound[1])
            data = data[mask]
            if (weights is not None) :
                weights = weights[mask]

        self.f = f  # model predicts PDF for given x
        self.data = np.array(data)
        self.weights = set_var_if_None(weights, self.data)
        self.bad_value = badvalue
        
        self.extended = extended
        self.extended_bound = extended_bound
        self.extended_nint = extended_nint
        if extended and extended_bound is None:
            self.extended_bound = (np.min(data), np.max(data))

        
        self.func_code = make_func_code(describe(self.f)[1:])

    def __call__(self, *par):  # par are a variable number of model parameters
        
        logf = np.zeros_like(self.data)
        
        # compute the function value
        f = compute_f(self.f, self.data, *par)
    
        # find where the PDF is 0 or negative (unphysical)        
        mask_f_positive = (f>0)

        # calculate the log of f everyhere where f is positive
        logf[mask_f_positive] = np.log(f[mask_f_positive]) * self.weights[mask_f_positive] 
        
        # set everywhere else to badvalue
        logf[~mask_f_positive] = self.bad_value
        
        # compute the sum of the log values: the LLH
        llh = -np.sum(logf)
        
        if self.extended:
            extended_term = integrate1d(self.f, self.extended_bound, self.extended_nint, *par)
            llh += extended_term
        
        return llh
    
    def default_errordef(self):
        return 0.5





class BinnedLH:  # override the class with a better one
    
    def __init__(self, f, data, bins=40, weights=None, weighterrors=None, bound=None, badvalue=1000000, extended=False, use_w2=False, nint_subdiv=1):
        
        if bound is not None:
            data = np.array(data)
            mask = (data >= bound[0]) & (data <= bound[1])
            data = data[mask]
            if (weights is not None) :
                weights = weights[mask]
            if (weighterrors is not None) :
                weighterrors = weighterrors[mask]

        self.weights = set_var_if_None(weights, data)

        self.f = f
        self.use_w2 = use_w2
        self.extended = extended

        if bound is None: 
            bound = (np.min(data), np.max(data))

        self.mymin, self.mymax = bound

        h, self.edges = np.histogram(data, bins, range=bound, weights=weights)
        
        self.bins = bins
        self.h = h
        self.N = np.sum(self.h)

        if weights is not None:
            if weighterrors is None:
                self.w2, _ = np.histogram(data, bins, range=bound, weights=weights**2)
            else:
                self.w2, _ = np.histogram(data, bins, range=bound, weights=weighterrors**2)
        else:
            self.w2, _ = np.histogram(data, bins, range=bound, weights=None)


        
        self.badvalue = badvalue
        self.nint_subdiv = nint_subdiv
        
        
        self.func_code = make_func_code(describe(self.f)[1:])
        self.ndof = np.sum(self.h > 0) - (self.func_code.co_argcount - 1)
        

    def __call__(self, *par):  # par are a variable number of model parameters

        # ret = compute_bin_lh_f(self.f, self.edges, self.h, self.w2, self.extended, self.use_w2, self.badvalue, *par)
        ret = compute_bin_lh_f2(self.f, self.edges, self.h, self.w2, self.extended, self.use_w2, self.nint_subdiv, *par)
        
        return ret


    def default_errordef(self):
        return 0.5




import warnings


def xlogyx(x, y):
    
    #compute x*log(y/x) to a good precision especially when y~x
    
    if x<1e-100:
        warnings.warn('x is really small return 0')
        return 0.
    
    if x<y:
        return x*np.log1p( (y-x) / x )
    else:
        return -x*np.log1p( (x-y) / y )


#compute w*log(y/x) where w < x and goes to zero faster than x
def wlogyx(w, y, x):
    if x<1e-100:
        warnings.warn('x is really small return 0')
        return 0.
    if x<y:
        return w*np.log1p( (y-x) / x )
    else:
        return -w*np.log1p( (x-y) / y )


def compute_bin_lh_f2(f, edges, h, w2, extended, use_sumw2, nint_subdiv, *par):
    
    N = np.sum(h)
    n = len(edges)

    ret = 0.
    
    for i in range(n-1):
        th = h[i]
        tm = integrate1d(f, (edges[i], edges[i+1]), nint_subdiv, *par)
        
        if not extended:
            if not use_sumw2:
                ret -= xlogyx(th, tm*N) + (th-tm*N)

            else:
                if w2[i]<1e-200: 
                    continue
                tw = w2[i]
                factor = th/tw
                ret -= factor*(wlogyx(th,tm*N,th)+(th-tm*N))
        else:
            if not use_sumw2:
                ret -= xlogyx(th,tm)+(th-tm)
            else:
                if w2[i]<1e-200: 
                    continue
                tw = w2[i]
                factor = th/tw
                ret -= factor*(wlogyx(th,tm,th)+(th-tm))

    return ret





def compute_bin_lh_f(f, edges, h, w2, extended, use_sumw2, badvalue, *par):
    
    mask_positive = (h>0)
    
    N = np.sum(h)
    midpoints = (edges[:-1] + edges[1:]) / 2
    b = np.diff(edges)
    
    midpoints_pos = midpoints[mask_positive]
    b_pos = b[mask_positive]
    h_pos = h[mask_positive]
    
    if use_sumw2:
        warnings.warn('use_sumw2 = True: is not yet implemented, assume False ')
        s = np.ones_like(midpoints_pos)
        pass
    else: 
        s = np.ones_like(midpoints_pos)

    
    E_pos = f(midpoints_pos, *par) * b_pos
    if not extended:
        E_pos = E_pos * N
        
    E_pos[E_pos<0] = badvalue
    
    ans = -np.sum( s*( h_pos*np.log( E_pos/h_pos ) + (h_pos-E_pos) ) )

    return ans


def fisher_LDA(X, y, picname, spec_name1, spec_name2, spec1_column, spec2_column, Nbins):
    """  
    Makes a Fishers lineaer discriminant between two variables and two species. 
    
    Takes a dataset (X) and a traningsset (y).
    
    """
    
    spec1 = X[y == spec1_column]
    spec2 = X[y == spec2_column]
    
    
    mu_spec1, mu_spec2 = spec1.mean(axis = 0).reshape(-1,1), spec2.mean(axis = 0).reshape(-1,1)
    #print(f"mu_{spec_name1}", mu_spec1)
    #print(f"mu_{spec_name2}", mu_spec2)
    
    cov_comb = np.cov(spec1.T) + np.cov(spec2.T)
    cov_comb_inv = np.linalg.inv(cov_comb)
    
    #print(cov_comb_inv)
    
    omega = cov_comb_inv.dot((mu_spec2 - mu_spec1))
    #print(omega.T)
    
    F_spec1 = np.dot(omega.T, spec1.T)
    F_spec2 = np.dot(omega.T, spec2.T)
    #print(F_spec1)
    #print(F_spec2)
    
    fig = plt.figure(figsize = (10,6))
    spec1_hist = plt.hist(F_spec1[0], bins = Nbins, label = f"{spec_name1}", histtype = "step")
    spec1_hist = plt.hist(F_spec2[0], bins = Nbins, label = f"{spec_name2}", histtype = "step")
    plt.title(f"LDA with all variables for {spec_name1} and {spec_name2}")
    plt.xlabel("x")
    plt.ylabel("Frequency")
    plt.legend()
    plt.savefig(f"{picname}.eps", format = "eps")
    plt.show()
    
    return F_spec1, F_spec2, y




def gaussian_clustering(data, xlabel, ylabel, title, plot = False):
    
    gm = GaussianMixture(n_components=2).fit(data)
    gm.get_params();
    
    pred = gm.predict(data)

    df = pd.DataFrame({'x':data[:,0], 'y':data[:,1], 'label':pred})
    groups = df.groupby('label')
    if plot:
        ig, ax = plt.subplots()
        for name, group in groups:
            ax.scatter(group.x, group.y, label = name)

        ax.legend()
        plt.xlabel(f"{xlabel}")
        plt.ylabel(f"{ylabel}")
        plt.savefig(f"{title}.eps", format = "eps")
        plt.show() 
        
    return pred, df


    







def plot_hist_variables(X, y, var_name, spec_name, Nbins):
      
    
    """ 
    Plot the Histograms for all species and all variables. 
    
    Takes the data X, the labels y, a list of variable names, a list with the names of the species, Number of bins. 
    
    """
    
    nvar  = len(var_name)     # Sepal Length, Sepal Width, Petal Length, Petal Width
    nspec = len(spec_name)    # Setosa, Versicolor, Virginica
    print(nspec)
    
    fig, ax = plt.subplots(nrows = int(nvar/2), ncols  = nspec, figsize = (10,8))
    ax = ax.flatten()
    
    mu = np.zeros((nspec, nvar))
    std = np.zeros((nspec, nvar))

    for ivar in range(nvar): # for each variable plot each species' corresponding histogram of that variable
        for ispec in range(nspec):

            data_spec = X[y == ispec]  # extract the relevant species
            data_spec_var = data_spec[:, ivar]      # extract the relevant variable

            ax[ivar].hist(data_spec_var, Nbins, histtype = 'step', label = spec_name[ispec], lw = 2)

            mu[ispec, ivar] = data_spec_var.mean()
            std[ispec, ivar] = data_spec_var.std(ddof=1)

        ax[ivar].set(title = var_name[ivar], ylabel='Counts')
        ax[ivar].legend()

    fig.tight_layout()



    
def accept_reject(func, picname, N_points, N_bins, x_min, x_max, y_min, y_max, plot = False):
    """
    INPUT: 
    
    func = the function we want to simulate
    
    N_points = amount of numbers we want to create
    
    (x_min, x_max, y_min, y_max) = bounds in both x- and y-direction
    
    N_bins = just for plotting
    
    Remember that the function has to be normalized
    
    Remember that it probably is best if x_max is equal to the normalization
    
    """
   
    k = (x_max - x_min) / N_bins
    N = N_points * k
    
    N_try = 0  # Number of tries
    
    x_generated = np.zeros(N_points)  # Empty array for the number of points I want to generate
    
    for i in range(N_points):
        while True:
            N_try += 1
            x_test = np.random.uniform(x_min, x_max)
            y_test = np.random.uniform(y_min, y_max) 
            
            if (y_test < func(x_test)):  # Test that the point is within the distribution, if yes break loop
                break
    
        x_generated[i] = x_test
    
    eff = N_points / N_try 
    
    eff_err = np.sqrt(eff * (1-eff) / N_try)  # Error on efficiency
    
    integral = eff * eff * (x_max-x_min) * (y_max-y_min)  # Estimate integral, as the efficiency times the area of the bounding box
   
    integral_err = eff_err * (x_max-x_min) * (y_max-y_min)   # Integral error
    
    # Option to plot the generated values in a histogram (this does NOT fit using Minuit, but just shows the generated values)
    if plot: 
        
        # Defining figure
        fig, ax = plt.subplots(figsize = (12,6))
        
        # Plotting
        x_pdf = np.linspace(x_min, x_max, 100)
        y_pdf =  N * func(x_pdf)

        plt.hist(x_generated, bins = N_bins, histtype = 'step', range = (x_min, x_max))
        plt.plot(x_pdf, y_pdf)
        
        plt.savefig(f"{picname}.eps", format = "eps")
        plt.show()
        
 
    
    
    return x_generated, [eff, eff_err], [integral, integral_err]



def bin_and_fit_gauss(data, picname, mu_value, sig_value, N_value, Nbins, xmin, xmax):
    """Fits a gaussian to binned data"""

    counts, bin_edges = np.histogram(data, bins = Nbins, range = (xmin, xmax))


    x = (bin_edges[1:][counts>0] + bin_edges[:-1][counts>0])/2
    y = counts[counts>0]
    sy = np.sqrt(counts[counts>0])   
        
    def gauss_pdf(x, mu, sigma, N):
        return N * (1 / np.sqrt(2 * np.pi) / sigma * np.exp(-0.5 * (x - mu)**2 / sigma**2))
    
    Minuit.print_level = 1    # Print result of fits (generally - can also be moved down to each fit instance)

    def chi2_owncalc(mu, sigma, N) :
        y_fit = gauss_pdf(x, mu, sigma, N)
        chi2 = np.sum(((y - y_fit) / sy)**2)
        return chi2

    minuit_chi2 = Minuit(chi2_owncalc, mu = mu_value, sigma = sig_value, N = N_value)
    minuit_chi2.errordef = 1.0     
    minuit_chi2.migrad()  
    
    fit_mu, fit_sigma, fit_N = minuit_chi2.values[:]
    
    for name in minuit_chi2.parameters :
        value, error = minuit_chi2.values[name], minuit_chi2.errors[name]
        print(f"Fit value: {name} = {value:.5f} +/- {error:.5f}")
    
    
    chi2_value = minuit_chi2.fval 
    
    N_NotEmptyBin = np.sum(y > 0)
    Ndof_value = N_NotEmptyBin - minuit_chi2.nfit
    
    Prob_value = stats.chi2.sf(chi2_value, Ndof_value) 
    print(f"Chi2 value: {chi2_value:.1f}   Ndof = {Ndof_value:.0f}    Prob(Chi2,Ndof) = {Prob_value:5.3f}")
    
    ##########################################################################################
    
    fig, ax = plt.subplots(figsize=(14, 8))  
    ax.errorbar(x, y, yerr=sy, xerr=0.0, label='Data, with Poisson errors', fmt='.k',  ecolor='k', elinewidth=1, capsize=1, capthick=1)


    ax.set(xlabel="Value", 
           ylabel="Relative frequency",  
           #title="Distribution of Gaussian and exponential numbers", 
           ylim=[0.0,None]) # 


    x_axis = np.linspace(xmin, xmax, 1000)
    ax.plot(x_axis, gauss_pdf(x_axis, *minuit_chi2.values[:]), '-r', label='Chi2 fit model result') 

    d = {'N':   [minuit_chi2.values['N'], minuit_chi2.errors['N']],
     'mu':       [minuit_chi2.values['mu'], minuit_chi2.errors['mu']],
     'sigma':       [minuit_chi2.values['sigma'], minuit_chi2.errors['sigma']],
     'Chi2':     chi2_value,
     'ndf':      Ndof_value,
     'Prob':     Prob_value,
        }

    text = nice_string_output(d, extra_spacing=2, decimals=3)
    add_text_to_ax(0.62, 0.95, text, ax, fontsize=15)
   
    ax.legend(loc='lower left', fontsize=18); # could also be # loc = 'upper right' e.g.
    fig.tight_layout()
    
    #fig.savefig(f"{picname}", format = "png")
    
    fig.savefig(f"{picname}.eps", format = "eps")

    
    
def bin_data(data, Nbins, xmin, xmax, plot = False):
    

    counts, bin_edges = np.histogram(data, bins = Nbins, range = (xmin, xmax))
    
    x = (bin_edges[1:][counts>0] + bin_edges[:-1][counts>0])/2
    y = counts[counts>0]
    sy = np.sqrt(counts[counts>0])   
        
    if plot:
        
        fig, ax = plt.subplots(figsize=(14, 6))
        hist = ax.hist(data, bins=Nbins, range=(xmin, xmax), histtype='step', linewidth=2, label='Data, normal histogram')

        ax.errorbar(x, y, yerr=sy, xerr=0.0, label='Data, with Poisson errors', fmt='.k',  ecolor='k', elinewidth=1, capsize=1, capthick=1)

        ax.set(xlabel="Random numbers",         
               ylabel="Frequency")           
               #title ="Distribution of Gaussian and exponential numbers")   
        ax.legend(loc='best', fontsize=20);       
        
    
    return x, y, sy




def minuit_shortcut(x, y, sigma, func, init_vals, names, picname, value_placement = [0.62, 0.95], labels = ["x", "y"], plot = False):
    """
    A shortcut to fitting any function with Minuit. This function does NOT plot, but returns values for plotting.
    INPUT:
    (x,y,sigma) = the data-points we want to fit our model to - array-like
    func = model we want to fit to our data; must only be a function of x and a set of parameters
    init_vals = array of initial values for the input parameters to the model - array-like
    names = array of strings, with the names of each parameter. Only included for readability of the result
    
    """
    
    
    def chi2(x, y, sigma, par):
        y_fit = func(x, *par)
        chi2 = np.sum(((y - y_fit) / sigma)**2)
        return chi2
    
    minimize_object = lambda *par: chi2(x, y, sigma, *par) # This only a function of the ARRAY of parameters
    minimize_object.errordef = Minuit.LEAST_SQUARES

    minuit_fit = Minuit(minimize_object, init_vals)
    minuit_fit.migrad()
    
    y_res = y - func(x, *minuit_fit.values)
    res_mean = np.mean(y_res)
    res_err = np.std(y_res)

    # Extracting and printing fit-parameters
    chi2_val = minuit_fit.fval  # Chi2, the minimized quantity
    ndof = len(x) - len(minuit_fit.values[:]) # Degrees of freedom
    prob_value = stats.chi2.sf(chi2_val, ndof) # p-value
    fit_par, fit_err = minuit_fit.values, minuit_fit.errors # Fit parameters and their errors (might need unpacking)
    
    for i, j in enumerate(names):
        print('Fit', j ,f': {fit_par[i]:.4f} +/- {fit_err[i]:.4f}')
    print(f'\np(chi2 = {chi2_val:.1f}, ndof = {ndof:d}) = {prob_value:6.4f}')
    
    
    xmin, xmax = np.min(x), np.max(x)
    x_fit = np.linspace(xmin, xmax, 1000)
    
    y_fit_new = func(x_fit, *minuit_fit.values[:]) 
    
    
    if plot:
        fig, ax = plt.subplots(figsize=(14, 8))  
        ax.errorbar(x, y, yerr=sigma, xerr=0.0, label= 'Data, with Poisson errors', fmt='.k',  ecolor='k', elinewidth=1, capsize=1, capthick=1)


        ax.set(xlabel = labels[0], 
               ylabel = labels[1],  
               #title="Distribution of Gaussian and exponential numbers", 
               ylim=[None,None]) # 


        x_axis = np.linspace(xmin, xmax, 1000)
        ax.plot(x_axis, func(x_axis, *minuit_fit.values[:]), '-r', label='Chi2 fit model result') 
        
        
        d = {}

        for i in names:
            d[i] = None
 
        for (name, value, error) in zip(names, minuit_fit.values, minuit_fit.errors):
            d[f"{name}"] = [value, error]
        
        d["Chi2"] = chi2_val
        d["ndf"] = ndof
        d["Prob"] = prob_value
        
        text = nice_string_output(d, extra_spacing=2, decimals=3)
        add_text_to_ax(value_placement[0], value_placement[1], text, ax, fontsize=15)

        ax.legend(loc = 'best', fontsize=12); 
        fig.tight_layout()

        fig.savefig(f"{picname}.eps", format = "eps")
    
    
    return fit_par, fit_err, chi2_val, ndof, prob_value, y_res, x_fit, y_fit_new

        
    
def c_criterion(reduction_list, threshold):
    """ n = sample size, reduction_list = list with outliers"""
    
    deleted_list = [] # for saving the deleted values
    
    reduction_list_update = reduction_list
    
    boolean = 0
    while boolean < 1:     
        mean = np.mean(reduction_list_update)
        std = np.std(reduction_list_update)
        n = len(reduction_list_update)
        
        erfc_list = n * special.erfc(np.absolute(reduction_list_update - mean) / std) 
        
        delete_index = []        
        
        for i, value in enumerate(erfc_list):
            if value < threshold:
                delete_index.append(i)
        
        
        for i in delete_index: # saving the deleted values
            deleted_list.append(reduction_list_update[i])
                
        reduction_list_update = np.delete(reduction_list_update, delete_index)
        
        
        if not delete_index: # stop the loop
            boolean = 1
            
    
    return reduction_list_update, deleted_list
    

def calc_ROC(hist1, hist2) :

    # First we extract the entries (y values) and the edges of the histograms:
    # Note how the "_" is simply used for the rest of what e.g. "hist1" returns (not really of our interest)
    y_sig, x_sig_edges, _ = hist1 
    y_bkg, x_bkg_edges, _ = hist2
    
    # Check that the two histograms have the same x edges:
    if np.array_equal(x_sig_edges, x_bkg_edges) :
        
        # Extract the center positions (x values) of the bins (both signal or background works - equal binning)
        x_centers = 0.5*(x_sig_edges[1:] + x_sig_edges[:-1])
        
        # Calculate the integral (sum) of the signal and background:
        integral_sig = y_sig.sum()
        integral_bkg = y_bkg.sum()
    
        # Initialize empty arrays for the True Positive Rate (TPR) and the False Positive Rate (FPR):
        TPR = np.zeros_like(y_sig) # True positive rate (sensitivity)
        FPR = np.zeros_like(y_sig) # False positive rate ()
        
        # Loop over all bins (x_centers) of the histograms and calculate TN, FP, FN, TP, FPR, and TPR for each bin:
        for i, x in enumerate(x_centers): 
            
            # The cut mask
            cut = (x_centers < x)
            
            # True positive
            TP = np.sum(y_sig[~cut]) / integral_sig    # True positives
            FN = np.sum(y_sig[cut]) / integral_sig     # False negatives
            TPR[i] = TP / (TP + FN)                    # True positive rate
            
            # True negative
            TN = np.sum(y_bkg[cut]) / integral_bkg      # True negatives (background)
            FP = np.sum(y_bkg[~cut]) / integral_bkg     # False positives
            FPR[i] = FP / (FP + TN)                     # False positive rate            
            
        return FPR, TPR
    
    else:
        AssertionError("Signal and Background histograms have different bins and/or ranges")
    

def runs_test(data, fit, threshold = 2.33):
    N, n_p, n_m = 0,0,0

    for i, i_fit in zip(data, fit):
        if i > i_fit:
            n_p += 1
        elif i < i_fit:
            n_m += 1
    N = n_p + n_m 
    
    
    runs_obs = 0
    for i in range(N-1):
        if ((data[i] < fit[i] and data[i+1] > fit[i+1]) or (data[i] > fit[i] and data[i+1] < fit[i+1])):
            runs_obs += 1
    print("runs observed", runs_obs)
    
    mean_runs = (2 * n_p * n_m / N) + 1

    variance_runs = ((mecalan_runs - 1) * (mean_runs - 2)) / (N - 1)

    std_runs = np.sqrt(variance_runs)

    print('mean', mean_runs)
    print('variance', variance_runs)
    print("std", std_runs)
    
    z = (mean_runs - runs_obs)/std_runs
    p = stats.norm.sf(abs(z))  

    print("z", z)
    threshold = 2.33 # 90 percent confidence.
    alpha = stats.norm.sf(threshold) 

    print("alpha", alpha)
    print("p", p)
    
    
    
    
    
def calibration(x, y, sy, a):
    
    def linear(x, a):
        return x * a 
    
    def chi2_owncalc(a) :
        y_fit = linear(x, a)
        chi2 = np.sum(((y - y_fit) / sy)**2)
        return chi2
    
    minuit_chi2 = Minuit(chi2_owncalc, a) 
    minuit_chi2.errordef = 1.0     
    minuit_chi2.migrad()
    
    fit_a = minuit_chi2.values[:]
    
    for name in minuit_chi2.parameters :
        value, error = minuit_chi2.values[name], minuit_chi2.errors[name]
        print(f"Fit value: {name} = {value:.5f} +/- {error:.5f}")
    
    chi2_value = minuit_chi2.fval 

    N_NotEmptyBin = np.sum(y > 0)
    Ndof_value = N_NotEmptyBin - minuit_chi2.nfit

    Prob_value = stats.chi2.sf(chi2_value, Ndof_value) 
    print(f"Chi2 value: {chi2_value:.1f}   Ndof = {Ndof_value:.0f}    Prob(Chi2,Ndof) = {Prob_value:5.3f}")
    

    
def weighted_mean(val, err, picname, labels, plot = False, title = None):
    """
    INPUT:
    val = values, array_like
    err = erros, array_like
    plot = option to plot or not
    
    """
   
    # Calculate the avg according to Barlow (4.6)
    avg = np.sum( (val / err**2) / np.sum( 1 / err**2 ) )
    
    # Calculate the error
    avg_sig = np.sqrt( 1 / np.sum(1 / err**2) ) 
    
    # Find degrees of freedom (-1)
    N_dof = len(val) - 1
    
    # Calculate chi_square
    chi2 = np.sum( (val - avg)**2 / err**2 )
    
    # Calculate p-value (the integral of the chi2-distribution from chi2 to infinity)
    p = stats.chi2.sf(chi2, N_dof)
    
    # Option to plot the fitted line
    if plot:
        
        # Create figure
        fig, ax = plt.subplots(figsize = (12,6))
        
        # X values are measurement number
        x = np.arange(len(val))+1
        
        # Plot values and errorbars
        ax.scatter(x, val)
        ax.errorbar(x, val, err, fmt = 'ro', ecolor = 'k', elinewidth = 1, capsize = 2, capthick = 1)
        
        # Plot the weighted average line
        ax.hlines(avg, 0, len(val)+0.5, colors = 'r', linestyle = 'dashed')
        
        # Nice text
        d = {'mu':   avg,
             'sigma_mu': avg_sig,
             'Chi2':     chi2,
             'ndf':      N_dof,
             'Prob':     p,
            }
        
        
        ax.set(xlabel = labels[0], 
               ylabel = labels[1],  
               #title="Distribution of Gaussian and exponential numbers", 
               ylim=[None,None]) # 
        
        text = nice_string_output(d, extra_spacing = 2, decimals = 3)
        add_text_to_ax(0.02, 0.95, text, ax, fontsize = 14)
        ax.set_title(title, fontsize = 18)
        ax.set(xlabel = labels[0],
               ylabel = labels[1])
        fig.tight_layout()
        
        fig.savefig(f"{picname}.eps", format = "eps")
        
        
    return avg, avg_sig, chi2, p



def t_test(group_1, group_2, hyp, equal_var = True):
    """
    Calculates the test statistics and p-value using the scipy function 'ttest_ind'.
    
    INPUT:
    (group_1, group_2) = the two data-sets we want to perfom the test on
    hyp = hypothesis (can be: 'two-sided', 'less', 'greater')
    equal_var = If True (default), perform a standard independent 2 sample test that assumes equal population variances. If False,     perform Welch’s t-test, which does not assume equal population variance. 
    
    """
    
    # Scipy function
    t_statistics, p_value = stats.ttest_ind(group_1, group_2, equal_var, alternative = hyp)
    print('t-statistics:', t_statistics, '(Scipy function)')
    print('p-value:', p_value, '(Scipy function)')
       
    return t_statistics, p_value
    
def transformation_method(func, func_transform, Npoints, Nbins, xmin, xmax, norm):

    
    r = np.random
    r.seed(42)
    
    r_uniform = r.uniform(size = Npoints)
    
    k = (xmax - xmin) / Nbins
    N = Npoints * k
    
    x_transformation = func_transform(r_uniform)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(x_transformation, bins = Nbins, range = (xmin, xmax), histtype='step', label = 'histogram' )
    ax.set(xlabel = "x", ylabel="Frequency", xlim=(xmin-0.1, xmax+0.1));
    
    
    x_axis1 = np.linspace(xmin, xmax, 1000)
    y_axis1 = norm * N * func(x_axis1)
    ax.plot(x_axis1, y_axis1, 'r-', label='function (not fitted)')


    d = {'Entries': len(x_transformation),
         'Mean': x_transformation.mean(),
         'Std': x_transformation.std(ddof=1),
        }

    text = nice_string_output(d, extra_spacing=2, decimals=3)
    add_text_to_ax(0.05, 0.75, text, ax, fontsize=14)

    ax.legend(loc='best')
    fig.tight_layout()

    fig
    
    
    
    
    
    