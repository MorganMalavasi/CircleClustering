import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.decomposition import PCA


def doPCA(X, labels, n_dataset):
    pca = PCA(n_components=2)
    components = pca.fit_transform(X)
    fig = px.scatter(components, x = 0, y = 1, title = 'Blobs', color=labels, labels={'0': 'PC 1', '1': 'PC 2'})
    fig.update_layout(
        width = 600,
        height = 600,
        title = 'Dataset nr {0} - samples = {1} - features = {2} - classes = {3}'.format(n_dataset, X.shape[0], X.shape[1], np.max(labels) + 1))
    fig.update_yaxes(
        scaleanchor = "x",
        scaleratio = 1)
    fig.show()

def plot_blobs(X, labels=None, threeD=False, doPCA=True, sizex=1):

    """
    Plot the dataset on screen

    Parameters
    ----------
    X       : numpy.ndarray
    labels  : numpy.ndarray
    threeD  : boolean
    doPCA   : boolean
    sizex   : integer

    Returns
    ----------
    nothing
    
    """

    if threeD:
        if PCA:
          pca = PCA(n_components=3)
          components = pca.fit_transform(X)
        else:
            components = X[:,0:3]  
        if labels is None:
            fig = px.scatter_3d(components, x=0, y=1, z=2, title='Blobs 3D', labels={'0': 'PC 1', '1': 'PC 2', '2': 'PC 3'})
        else:
            fig = px.scatter_3d(components, x=0, y=1, z=2, color=labels, title='Blobs 3D', labels={'0': 'PC 1', '1': 'PC 2', '2': 'PC 3'})
    else:
        if doPCA:
            pca = PCA(n_components=2)
            components = pca.fit_transform(X)
        else:
            components = X[:,0:2]  
        if labels is None:
            fig = px.scatter(components, x=0, y=1, title='Blobs 2D', labels={'0': 'PC 1', '1': 'PC 2'})
        else:
            fig = px.scatter(components, x=0, y=1, title='Blobs 2D', color=labels, labels={'0': 'PC 1', '1': 'PC 2'})
    
    fig.update_layout(
        width = 800*sizex,
        height = 800*sizex,
        title = "fixed-ratio axes")
    fig.update_yaxes(
        scaleanchor = "x",
        scaleratio = 1)
    fig.show()  

# plotting the data
def plot_circle(theta, label=None, radius=500):
    """
    Plot the results of the CircleClustering algorithm on the circle

    Parameters
    ----------
    theta   : numpy.ndarray
    labels  : numpy.ndarray
    radius  : integer

    Returns
    ----------
    nothing
    
    """

    x = np.cos(theta)
    y = np.sin(theta)

    fig = go.Figure()
    fig.add_shape(type="circle", xref="x", yref="y", x0=-1, y0=-1, x1=1, y1=1, line=dict(color="black", width=1))
    
    if label is None:
        fig.add_trace(go.Scatter(x=x, y=y,
            mode='markers',
            marker_symbol='circle',
            marker_size=10))
    else:
        ul = np.unique(label)
        cols = list(range(len(ul)))
        for c,u in zip(cols,ul):
            idx = np.where(u == label)
            fig.add_trace(go.Scatter(x=x[idx], y=y[idx],
                mode='markers',
                marker_symbol='circle',
                marker_color=cols[c], 
                marker_line_color=cols[c],
                marker_line_width=0, 
                marker_size=10))
    
    M = 1.05
    fig.update_xaxes(title='', range=[-M, M])
    fig.update_yaxes(title='', range=[-M, M])
    fig.update_layout(title='clusters', width=radius, height=radius)
    fig.show()

def smooth(x, window_len=11, window='hanning'):

    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len<3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s = np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w = np.ones(window_len,'d')
    else:
        w = eval('np.'+window+'(window_len)')

    y = np.convolve(w/w.sum(),s,mode='valid')
    return y

def plot_scatter(hist, bins, mode=0, smooth_wlen=None):

    if mode==0:
        mode_line = 'lines'
    elif mode == 1:
        mode_line = 'markers'
    else:
        mode_line = 'lines+markers'
    
    if smooth_wlen is not None:
        hist = smooth(hist, window_len=smooth_wlen, window='hanning')

    figh = go.Figure(data=go.Scatter(x=bins, y=hist, mode=mode_line))
    figh.show()

def plot_hist(hist, bins):

    size = hist.shape[0]
    samplesInHistogram = np.empty([size, 2])
    for i in range(size):
        samplesInHistogram[i] = [bins[i], hist[i]]

    df = pd.DataFrame(samplesInHistogram, columns=['bin', 'height'])
    fig = px.histogram(df, x="bin", y="height", nbins=bins.shape[0])
    fig.show()
    # print 
    
    # print(samplesInHistogram)
    # figh = px.histogram(samplesInHistogram)
    # figh.show()