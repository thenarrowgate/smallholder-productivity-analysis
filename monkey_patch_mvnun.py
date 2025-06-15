# monkey_patch_mvnun.py

import numpy as np
import scipy.stats as stats

# semopy does: from scipy.stats import mvn
# so mvn is stats.mvn, which is scipy.stats._multivariate
mvn = stats.mvn if hasattr(stats, "mvn") else stats._multivariate  

# Grab the modern multivariate_normal
from scipy.stats import multivariate_normal

def mvnun(lower, upper, mean, cov):
    """
    Replacement for the old mvn.mvnun(lower, upper, mean, cov):
    returns (probability_between(lower,upper), None).
    """
    lower = np.atleast_1d(lower)
    upper = np.atleast_1d(upper)
    mean  = np.atleast_1d(mean).ravel()
    # CDF difference gives the probability mass in the hyper-rectangle
    p_upper = multivariate_normal.cdf(upper, mean=mean, cov=cov)
    p_lower = multivariate_normal.cdf(lower, mean=mean, cov=cov)
    return (p_upper - p_lower, None)

# Patch it in
setattr(mvn, "mvnun", mvnun)
