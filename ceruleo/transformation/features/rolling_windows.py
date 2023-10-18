import numpy as np 


def _strided_app(a: np.array, L: int, S: int):  # Window len = L, Stride len/stepsize = S
    """
    Returns an array that is strided

    Parameters:
        a: Array to be strided
        L: Length of the window
        S: Stride (S=L/stepsize)
    """
    nrows = ((a.size-L)//S)+1
    n = a.strides[0]
    r =  np.lib.stride_tricks.as_strided(
        a, 
        shape=(nrows, L), 
        strides=(S*n, n),
        writeable=False)
    last = r.shape[0]*r.shape[1]
    l = r.tolist()
    if last < len(a):
        l.append(a[last:])
    return l
    

def apply_rolling_data(values : np.ndarray, function, window: int, step: int =1):
    """
    Perform a rolling window analysis at the column `col` from `data`

    Given a dataframe `data` with time series, call `function` at sections of length `window` at the data of column `col`. Append the results to `data` at a new columns with name `label`.

    Parameters:
        data: 1-D Time series of data
        function: Function to be called to calculate the rolling window analysis, the function must receive as input an array or pandas series. Its output must be either a number or a pandas series
        window: Length of the window to perform the analysis
        step: Step to take between two consecutive windows, by default 1

    Returns:
        data: Columns generated by the function applied
    """

    x = _strided_app(values, window, step)

    return np.vstack([function(np.array(b)) for b in x])
    
