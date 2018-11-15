"""Linear regysression"""

from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import random
#from Cython.Utility.MemoryView import step
from _tkinter import create


style.use('fivethirtyeight')

xs = np.array([1,2,3,4,5,6], dtype = np.float64)

ys = np.array([5,4,6,5,6,7], dtype = np.float64)



def create_dataset(hm, variance, step=2,correlation = False):
    
    
    val = 1
    ys = []
    for i in range(hm):
        y = val + random.randrange(-variance, variance)
        ys.append(y)
        if correlation and correlation=='pos':
            val+=step
        elif correlation and correlation == 'neg':
            val-+step
    xs = [i for i in range(len(ys))]

    return np.array(xs,dtype = np.float64),np.array(ys, dtype = np.float64)




def best_fit_slope(xs,ys):
    
    m = (mean(xs)*mean(ys)-mean(xs*ys))/((mean(xs)**2) - mean(xs**2))
    b = mean(ys)-m*mean(xs)
    
    return m, b


def square_error(ys_orig, ys_line):
    
    
    
    
    
    return sum((ys_line-ys_orig)**2)




def coefficient_of_determination(ys_orig,ys_line):
    
    y_mean_line = [mean(ys_orig)for y in ys_orig]
    sq_err_regr = square_error(ys_orig, ys_line)
    sq_err_y_mean = square_error(ys_orig, y_mean_line)
    return 1-(sq_err_regr/sq_err_y_mean)
    



xs, ys = create_dataset(40,40,2,correlation = 'pos')




m, b = best_fit_slope(xs,ys)


regression_line = [(m*x)+b for x in xs]


r_squared = coefficient_of_determination(ys,regression_line)
print r_squared
pre_x= 8
pre_y = (m*pre_x)+b

plt.scatter(xs,ys)
plt.scatter(pre_x, pre_y, color = 'g')
plt.plot(xs, regression_line)
plt.show()














