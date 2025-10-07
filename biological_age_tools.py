
import numpy as np
import sklearn
import sklearn.linear_model
import matplotlib
import matplotlib.pyplot as plt

###
### weighted moving average filters
###

def weighted_rectangular_filtering(xs, loc, std, normalize=True):
    #w = np.exp(-(xs - loc)**2 / (2*std**2))
    w = (np.abs((xs-loc)) < std).astype(np.float32)
    
    if normalize:
        w = w / np.sum(w)
    return w

def weighted_gaussian_filtering(xs, loc, std, normalize=True):
    w = np.exp(-(xs - loc)**2 / (2*std**2))
    if normalize:
        w = w / np.sum(w)
    return w

def weighted_laplace_filtering(xs, loc, std, normalize=True):
    w = np.exp(-np.abs(xs - loc) / std)
    if normalize:
        w = w / np.sum(w)
    return w

###
### Bootstrap toops for weighted moving averages with CI:s
###

def bootstrap_weighted_moving_average(xs, ys, loc, std, n, filtering_fun):
    x_means = []
    y_means = []
    all_inds = np.array(list(range(xs.size)))
    for i in range(n):
        inds_i = np.random.choice(all_inds, size=all_inds.size, replace=True)
        xs_i = xs[inds_i]#np.random.choice(xs, n=xs.size, replace=True)
        ys_i = ys[inds_i]
        w_i = filtering_fun(xs_i, loc, std, normalize=False)
        x_mean_i = np.sum(w_i*xs_i/np.sum(w_i))
        y_mean_i = np.sum(w_i*ys_i/np.sum(w_i))
        x_means.append(x_mean_i)
        y_means.append(y_mean_i)
    return np.mean(x_means), np.percentile(x_means, q=2.5), np.percentile(x_means, q=97.5), np.mean(y_means), np.percentile(y_means, q=2.5), np.percentile(y_means, q=97.5)

def weighted_average_with_ci_bootstrap(xs, ys, loc, std, filtering_fun=weighted_laplace_filtering, n=1000):
    return bootstrap_weighted_moving_average(xs, loc, std, n=n, filtering_fun=filtering_fun)

def weighted_percentile(xs, w, p):
    s = np.argsort(xs)
    xs = xs[s]
    w = w[s]
    wcum = np.cumsum(w)
    pos = wcum >= p
    ind = np.argmax(pos)
    return xs[ind]

def remove_bias_bspline_order1(pa, ca, STEPS=100, W=1.0, filt=None, enable_filtering=True, weighted_filtering_fun=weighted_laplace_filtering, verbose=False):
    # pa: predicted ages
    # ca: chronological ages
    # STEPS: number of grid points
    # W: The width of the weighted average filter
    # FILT: boolean or integer indices of elements to keep in the analysis
    # weighted_filtering_fun: the weighted filtering function
    # verbose: Flag controlling if to print the centers and shifts

    mn = np.amin(ca)
    mx = np.amax(ca)
    center = np.zeros((1, STEPS))
    shift = np.zeros((1, STEPS))
    
    for i in range(STEPS):
        age_i = mn + (mx-mn) * i / (STEPS-1)
        w_i = weighted_filtering_fun(ca, age_i, W, normalize=False)
        if filt is not None and enable_filtering:
            w_i = w_i[filt]
            ca_i = ca[filt]
            pa_i = pa[filt]
        else:
            ca_i = ca
            pa_i = pa

        p10 = weighted_percentile(pa_i, w_i/np.sum(w_i), 0.005)
        p90 = weighted_percentile(pa_i, w_i/np.sum(w_i), 0.995)
        filt_outliers = np.logical_or(pa_i < p10, pa_i > p90)
        w_i[filt_outliers] = 0.0

        w_i_sum = np.sum(w_i)

        ca_i_mean = np.sum(w_i * ca_i / w_i_sum)
        pa_i_mean = np.sum(w_i * pa_i / w_i_sum)

        diff = ca_i_mean - pa_i_mean

        center[0, i] = ca_i_mean
        shift[0, i] = diff

    if verbose:
        print('---Unbiasing weights---')
        print('Centers: ', center, np.diff(center[0, :]))
        print('Shifts: ', shift, np.diff(shift[0, :])/np.diff(center[0, :]))
    
    ca_ = ca.reshape(-1, 1)
    dist = np.abs(ca_ - center)
    total_shift = np.zeros(ca.shape[0])
    for i in range(ca.shape[0]):
        #assert(np.abs(w1[i]+w2[i]-1.0) < 1e-5)
        #total_shift[i] = w1[i] * shift[0, dist_ind1[i]] + w2[i] * shift[0, dist_ind2[i]]
        dist_i = dist[i, :]
        dist_ind = np.argsort(dist_i)
        dist1 = dist_i[dist_ind[0]]
        dist2 = dist_i[dist_ind[1]]
        tdist = dist1 + dist2
        w1 = 1.0 - dist1 / tdist
        w2 = 1.0 - dist2 / tdist
        total_shift[i] = w1 * shift[0, dist_ind[0]] + w2 * shift[0, dist_ind[1]]
        
    print(total_shift[:10])

    return pa + total_shift, center, shift

def apply_shift(pa, ca, center, shift):
    ca = np.clip(ca, np.amin(center), np.amax(center))
    ca_ = ca.reshape(-1, 1)
    dist = np.abs(ca_ - center)

    total_shift = np.zeros(ca.shape[0])
    for i in range(ca.shape[0]):
        dist_i = dist[i, :]
        dist_ind = np.argsort(dist_i)
        dist1 = dist_i[dist_ind[0]]
        dist2 = dist_i[dist_ind[1]]
        tdist = dist1 + dist2
        w1 = 1.0 - dist1 / tdist
        w2 = 1.0 - dist2 / tdist
        total_shift[i] = w1 * shift[0, dist_ind[0]] + w2 * shift[0, dist_ind[1]]
    
    return pa + total_shift
    
def wstd(xs, w):
    # assume normalized weights
    assert(np.abs(np.sum(w)-1.0) < 1e-5)
        
    wmean = np.sum(w*xs)
    return np.sqrt(np.sum(w*(xs-wmean)**2))

class DebiaserLinear:
    def __init__(self, steps, bw=None, weighted_filtering_fun=weighted_laplace_filtering):
        self.steps = steps
        self.bw = bw
        self.wff = weighted_filtering_fun
        self.model = None
        
    def fit(self, ca, pa, filt=None):
        mn = np.amin(ca)
        mx = np.amax(ca)
        
        if self.bw is None:
            # default
            local_bw = ((mx-mn)/self.steps)/2
        else:
            local_bw = self.bw
        
        c = []
        p = []
        for i in range(self.steps):
            age_i = mn + (mx-mn) * i / (self.steps-1)
            
            w_i = self.wff(ca, age_i, local_bw, normalize=False)
            if filt is not None:
                w_i = w_i[filt]
                ca_i = ca[filt]
                pa_i = pa[filt]
            else:
                ca_i = ca
                pa_i = pa
                
            w_i_sum = np.sum(w_i)

            ca_i_mean = np.sum(w_i * ca_i / w_i_sum)
            pa_i_mean = np.sum(w_i * pa_i / w_i_sum)
            
            c.append(ca_i_mean)
            p.append(pa_i_mean)

        c = np.array(c)
        p = np.array(p)
    
        model = sklearn.linear_model.LinearRegression()
        model.fit(c.reshape(-1, 1), c.reshape(-1, 1)-p.reshape(-1, 1), sample_weight=None)
        
        print(model.coef_, model.intercept_)
        self.model = model

    def predict(self, ca, pa):
        shift = self.model.predict(ca.reshape(-1, 1)).reshape(-1)
        pa = pa + shift
        
        return pa

class DebiaserBSpline1:
    def __init__(self, steps, bw=None, weighted_filtering_fun=weighted_laplace_filtering):
        self.steps = steps
        self.bw = bw
        self.wff = weighted_filtering_fun
        self.model = None
        
    def fit(self, ca, pa, filt=None):
        mn = np.amin(ca)
        mx = np.amax(ca)
        center = np.zeros((1, self.steps))
        shift = np.zeros((1, self.steps))
        
        for i in range(self.steps):
            age_i = mn + (mx-mn) * i / (self.steps-1)
            w_i = self.wff(ca, age_i, self.bw, normalize=False)
            if filt is not None:
                w_i = w_i[filt]
                ca_i = ca[filt]
                pa_i = pa[filt]
            else:
                ca_i = ca
                pa_i = pa

            w_i_sum = np.sum(w_i)

            ca_i_mean = np.sum(w_i * ca_i / w_i_sum)
            pa_i_mean = np.sum(w_i * pa_i / w_i_sum)

            diff = ca_i_mean - pa_i_mean

            center[0, i] = ca_i_mean
            shift[0, i] = diff
       
        self.center = center
        self.shift = shift           

    def predict(self, ca, pa):
        ca = np.clip(ca, np.amin(self.center), np.amax(self.center))
        ca_ = ca.reshape(-1, 1)
        dist = np.abs(ca_ - self.center)

        total_shift = np.zeros(ca.shape[0])
        for i in range(ca.shape[0]):
            dist_i = dist[i, :]
            dist_ind = np.argsort(dist_i)
            dist1 = dist_i[dist_ind[0]]
            dist2 = dist_i[dist_ind[1]]
            tdist = dist1 + dist2
            w1 = 1.0 - dist1 / tdist
            w2 = 1.0 - dist2 / tdist
            total_shift[i] = w1 * self.shift[0, dist_ind[0]] + w2 * self.shift[0, dist_ind[1]]
    
        return pa + total_shift

def remove_bias_linear(pa, ca, STEPS=100, W=1.0, filt=None):
    cmean = np.mean(ca)
    pmean = np.mean(pa)
    
    mn, mx = np.percentile(ca, q=(2.5, 97.5))
    
    c = []
    p = []
    for i in range(STEPS):
        age_i = mn + (mx-mn) * i / (STEPS-1)
        w_i = weighted_laplace_filtering(ca, age_i, W, normalize=False)
        if filt is not None:
            w_i = w_i[filt]
            ca_i = ca[filt]
            pa_i = pa[filt]
        else:
            ca_i = ca
            pa_i = pa
            
        p10 = weighted_percentile(pa_i, w_i/np.sum(w_i), 0.005)
        p90 = weighted_percentile(pa_i, w_i/np.sum(w_i), 0.995)
        filt_outliers = np.logical_or(pa_i < p10, pa_i > p90)
        w_i[filt_outliers] = 0.0
            
        w_i_sum = np.sum(w_i)

        ca_i_mean = np.sum(w_i * ca_i / w_i_sum)
        pa_i_mean = np.sum(w_i * pa_i / w_i_sum)
        
        c.append(ca_i_mean)
        p.append(pa_i_mean)

    c = np.array(c)
    p = np.array(p)
   
    model = sklearn.linear_model.LinearRegression()
    model.fit(c.reshape(-1, 1), c.reshape(-1, 1)-p.reshape(-1, 1), sample_weight=None)
    print(f'Bias removal intercept: {model.intercept_}, and coef: {model.coef_[0, 0]}')
    shift = model.predict(ca.reshape(-1, 1)).reshape(-1)
    pa = pa + shift
    return pa

def remove_bias_linear_with_scale(pa, ca, STEPS=100, W=1.0, filt=None):
    cmean = np.mean(ca)
    pmean = np.mean(pa)
    
    mn, mx = np.percentile(ca, q=(2.5, 97.5))
    
    c = []
    p = []
    sc = []
    for i in range(STEPS):
        age_i = mn + (mx-mn) * i / (STEPS-1)
        w_i = weighted_laplace_filtering(ca, age_i, W, normalize=False)
        if filt is not None:
            w_i = w_i[filt]
            ca_i = ca[filt]
            pa_i = pa[filt]
        else:
            ca_i = ca
            pa_i = pa
            
        p10 = weighted_percentile(pa_i, w_i/np.sum(w_i), 0.005)
        p90 = weighted_percentile(pa_i, w_i/np.sum(w_i), 0.995)
        filt_outliers = np.logical_or(pa_i < p10, pa_i > p90)
        w_i[filt_outliers] = 0.0
            
        w_i_sum = np.sum(w_i)

        ca_i_mean = np.sum(w_i * ca_i / w_i_sum)
        pa_i_mean = np.sum(w_i * pa_i / w_i_sum)
        ca_i_std = wstd(ca_i, w_i / w_i_sum)
        pa_i_std = wstd(pa_i, w_i / w_i_sum)
        
        c.append(ca_i_mean)
        p.append(pa_i_mean)
        sc.append(ca_i_std / np.clip(pa_i_std, 1e-3, None))

    c = np.array(c)
    p = np.array(p)
    sc = np.array(sc)
    
    model_shift = sklearn.linear_model.LinearRegression()
    model_shift.fit(c.reshape(-1, 1), c.reshape(-1, 1)-p.reshape(-1, 1), sample_weight=None)

    model_scale = sklearn.linear_model.LinearRegression()
    model_scale.fit(c.reshape(-1, 1), sc.reshape(-1, 1), sample_weight=None)
    
    print(f'Shift bias removal intercept: {model_shift.intercept_}, and coef: {model_shift.coef_[0, 0]}')
    print(f'Scale bias removal intercept: {model_scale.intercept_}, and coef: {model_scale.coef_[0, 0]}')
    
    shift = model_shift.predict(ca.reshape(-1, 1)).reshape(-1)
    scale = model_scale.predict(ca.reshape(-1, 1)).reshape(-1)
    
    # apply scale
    pa = ((pa + shift) - ca) * scale + ca
        
    return pa

def age_gap(pa, ca, FILT=None):
    if FILT is not None:
        pa = pa[FILT]
        ca = ca[FILT]
    
    return pa-ca

def filter_outlier_age_gaps(pa, ca, extra, p):
    ag = age_gap(pa, ca)
    c = np.maximum(np.abs(np.percentile(ag, q=p)), np.abs(np.percentile(ag, q=100-p)))
    filt = np.logical_and(ag > -c, ag < c)
    return pa[filt, ...], ca[filt, ...], extra[filt, ...]

def print_distribution(xs, title):
    print(f'{title}: {np.mean(xs)} +- {np.std(xs, ddof=1)}, min: {np.amin(xs)}, p25: {np.percentile(xs, q=25)}, median: {np.median(xs)}, p75: {np.percentile(xs, q=75)}, max: {np.amax(xs)}')
# plotting

def plot_age_gap(ca, pa, cls, name, PLOT_TITLE, out_path, STEPS=50, SPAN = 3.0, BOOTSTRAP_N=500, ymin=-1.0, ymax=3.0):
        x_pos = []
        y_pos = []
        x_control = []
        y_control = []
        y_min_pos = []
        y_max_pos = []
        y_min_control = []
        y_max_control = []

        n_pos = np.sum(cls>0)
        n_control = cls.size - n_pos

        min_age = np.amin(ca)
        max_age = np.amax(ca)
        for i in range(STEPS):
            age_i = min_age + (max_age-min_age) * (i / (STEPS-1))
            #print(age_i)

            age_gap = pa-ca

            pos = cls > 0
            control = cls == 0
            
            xmean_control, xmin_control, xmax_control, ymean_control, ymin_control, ymax_control = bootstrap_weighted_moving_average(ca[control], age_gap[control], age_i, SPAN, BOOTSTRAP_N, weighted_laplace_filtering)
            xmean_pos, xmin_pos, xmax_pos, ymean_pos, ymin_pos, ymax_pos = bootstrap_weighted_moving_average(ca[pos], age_gap[pos], age_i, SPAN, BOOTSTRAP_N, weighted_laplace_filtering)

            x_pos.append(xmean_pos)
            y_pos.append(ymean_pos)
            y_min_pos.append(ymin_pos)
            y_max_pos.append(ymax_pos)

            x_control.append(xmean_control)
            y_control.append(ymean_control)
            y_min_control.append(ymin_control)
            y_max_control.append(ymax_control)

        plt.plot(x_pos, y_pos, c='r', label=f'{name} ($n={n_pos}$)')
        plt.gca().fill_between(x_pos, y_min_pos, y_max_pos, color='r', alpha=.1)
        plt.plot(x_control, y_control, c='b', label=f'Control ($n={n_control}$)')
        plt.gca().fill_between(x_control, y_min_control, y_max_control, color='b', alpha=.1)
        plt.xlim((np.maximum(np.amin(x_pos), np.amin(x_control)), np.minimum(np.amax(x_pos), np.amax(x_control))))
        
        if ymin is None:
            ymin = np.minimum(np.amin(y_min_pos), np.amin(y_min_control))
        if ymax is None:
            ymax = np.maximum(np.amax(y_max_pos), np.amax(y_max_control))
        
        plt.ylim((ymin, ymax))
        plt.grid(True, axis='y')
        plt.legend()
        plt.xlabel(f'Chronological age (years)')
        plt.ylabel(f'Age gap (years)')
        print(f'Title: {PLOT_TITLE}')
        plt.title(PLOT_TITLE)
        print(f'Saving figure to {out_path}')
        plt.savefig(out_path, bbox_inches='tight')

        plt.close()
