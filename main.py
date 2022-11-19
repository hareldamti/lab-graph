import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd

table = pd.read_excel('data.xlsx', sheet_name='Sheet1')

x, y, dx, dy = table.iloc[14:, :4].values.T
config = {row.iloc[0]: row.iloc[1] for idx, row in table.iloc[3:11, :2].iterrows()}

linear_fit = lambda x, a, b: a * x + b
quad_fit = lambda x, a, b, c: a * x ** 2 + b * x + c
qubic_fit = lambda x, a, b, c, d: a * x ** 3 + b * x ** 2 + c * x + d
exp_fit = lambda x, a, b: a * np.e ** (b * x)
log_fit = lambda x, a, b: a * np.log(b * x)
inv_fit = lambda x, a, b: a / (x - b)
inv_sq_fit = lambda x, a, b: a / (x - b) ** 2

function_dict = {
    'linear': linear_fit,
    'quadratic': quad_fit,
    'qubic': qubic_fit,
    'exponential': exp_fit,
    'logarithmic': log_fit,
    'inverse': inv_fit,
    'inverse square': inv_sq_fit
}

f = function_dict[config['fit type']]


popt, pcov = curve_fit(f, x, y)

plt.errorbar(x, y, xerr=dx, yerr=dy, fmt='.', color=config['data color'],label='data')

x_range = np.linspace(min(x), max(x), 200)
plt.plot(x_range, f(x_range, *popt), color=config['fit color'],
         label=f'{config["fit type"]} fit')

plt.xlabel(table.iloc[13, 0])
plt.ylabel(table.iloc[13, 1])

if config['x_start'] != 'auto' and config['x_end'] != 'auto':
    plt.xlim(float(config['x_start']),float(config['x_end']))
if config['y_start'] != 'auto' and config['y_end'] != 'auto':
    plt.ylim(float(config['y_start']), float(config['y_end']))

chi2 = sum((f(x, *popt)-y)**2 / dy**2) / (len(x) - len(popt))
textbox = '\n'.join([f'{chr(ord("a")+i)}: {popt[i]:.3f}' for i in range(len(popt))])
textbox += f'\nchi2: {chi2:.3f}'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
plt.text(0.55, 0.97, textbox, transform=plt.gca().transAxes, fontsize=9,
        verticalalignment='top', bbox=props)

plt.title(config['title'])
plt.grid()
plt.legend()
plt.plot()

plt.save_fig('graph.png')