"""

This file contains some follow-up analysis I do to explore the patterns of the
cluster statistics
"""
# %% ########## Import and Set Path ##########
import geopandas as gpd
import pandas as pd
import numpy as np
import tqdm
from os import walk
from shapely.geometry import Point, Polygon, MultiPolygon, LineString
import matplotlib.pyplot as plt
import random
import seaborn as sns
import scipy

source_path = 'Export/3-7_Cluster_Statistics/'


# %% ########### PLOTTING ########## Define plotting functions: Gini, Plot_Ideal, plot_log_with_stat
def gini(x):
	# Mean absolute difference
	mad = np.abs(np.subtract.outer(x, x)).mean()
	# Relative mean absolute difference
	rmad = mad / np.mean(x)
	# Gini coefficient
	g = 0.5 * rmad
	return g


def plt_ideal(a, b):
	"""
	draw an ideal reference curve for graph of mean area vs. Gini
	:param a:
	:param b:
	:return:
	"""
	f = lambda X, a, b: (abs(a - b) * X * (1 - X)) / ((1 - X) * a + X * b)
	X = np.arange(0, 1, 0.0001)
	Y = f(X, a, b)
	plt.scatter(X * (b - a) + a, Y, s=1, alpha=.8, c='orange')


def plot_log_with_stat_dense(x, y, file_list, xlabel='x', ylabel='y', title='title',
                             show_plot=True, gini_ideal=False, source_path='Export/3-7_Cluster_Statistics/',
                             save_path=None, do_linear_regres=True):
	"""
	plot on single graph, using scipy and seaborn, with 95% CI and relevant statistics
	:param show_plot:
	:param source_path:
	:param gini_ideal:
	:param x:
	:param y:
	:param xlabel:
	:param ylabel:
	:return: the object of the graph
	"""
	# dont_log = ['label', 'Area_gini', 'building_density', 'high', 'mid', 'low', 'status_Gini']
	dont_log = dont_log_cols
	assert len(file_list) == 10
	fig = plt.figure(figsize=(25, 30), dpi=300)
	for i in range(len(file_list)):
		dtf = pd.read_csv(source_path + file_list[i])[[x, y]]
		fig.add_subplot(4, 3, i + 1)
		remove_notice = ''
		if x in dont_log:
			data_x = dtf[x]
			x_label = x
		else:
			data_x = np.log(dtf[x])
			x_label = 'log ' + x
		if y in dont_log:
			data_y = dtf[y]
			y_label = y
		else:
			data_y = np.log(dtf[y])
			y_label = 'log ' + y

		if do_linear_regres:
			sns.regplot(x=data_x, y=data_y, ci=95, fit_reg=True)
			slope, intercept, corr_coeff, p_value, std_err = scipy.stats.linregress(data_x, data_y)
			info = list(map(lambda k: round(k, 3), [slope, intercept, corr_coeff ** 2, p_value, std_err]))
			title_str = file_list[i][:-4] + remove_notice + \
			            '\nslope: ' + str(info[0]) + ', intercept: ' + str(info[1]) + \
			            ',r squared:' + str(info[2])
		else:
			sns.scatterplot(x=data_x, y=data_y)
			title_str = file_list[i][:-4] + remove_notice
		if gini_ideal:
			a, b, c = 25, 326.29, 2015 # these are min, mean, and max household area
			plt_ideal(a, b)
			plt_ideal(b, c)
		plt.xlabel(x_label)
		plt.ylabel(y_label)
		plt.title(title_str)
	# ax.plot()
	if show_plot:
		plt.show()
	else:
		if save_path:
			plt.savefig(save_path + x + '_|_' + y + '.png')
			print('saved to new path!')
		else:
			plt.savefig('Export/3-7_Cluster_Statistics/relatin_graphs/' + x + '_|_' + y + '.png')
			print('saved to deafult path!')
	return


# %% =========plot cluster characterstics: define the parametres and names=========
x_y_set = [
	('building_area', 'Area_gini'),
	('building_area', 'building_density'),
	('building_area', 'high'),
	# ('building_area', 'avg_status'),
	('total_households', 'cluster_area'),
	('total_households', 'Area_gini'),
	('total_households', 'high'),
	('total_households', 'avg_status'),
	('Area_mean', 'Area_gini'),
	('Area_gini', 'building_density'),
	('Area_gini', 'high'),
	('Area_gini', 'low'),
	('Area_gini', 'avg_status'),
	('high', 'low'),
	('building_density', 'high')
]

dont_log_cols = ['Unnamed: 0', 'label', 'building_area', 'total_households',
            'total_status', 'num_buildings', 'cluster_area',
            'Area_gini', 'Area_mean', 'building_density', 'high', 'mid', 'low',
            'status_Gini', 'avg_status']

ms_s = ['MeanShift-bandwidth=560.csv',
        'MeanShift-bandwidth=350.csv',
        'MeanShift-bandwidth=265.csv']

km_s = ['Kmeans-k=20.csv',
        'Kmeans-k=50.csv',
        'Kmeans-k=100.csv']

DB_s = ['DBSCAN-eps=184.csv',
        'DBSCAN-eps=141.csv',
        'DBSCAN-eps=111.5.csv']

plz = ['Plazas_Over10K_Minus_Sun_Thiessen.csv']

file_list = ms_s + km_s + DB_s + plz

# %% ########## plot HISTOGRAM of household area ##########
households = gpd.read_file('Export/Total_Households_info.geojson')
households.Area.value_counts()

# %% pltting using matplotlib
plt.hist(households[households.Status == 2]['Area'], log=True, bins=50)
plt.hist(households[households.Status == 1]['Area'], log=True, bins=200, color='red')

plt.hist(households[households.Status == 3]['Area'], log=True, bins=1, color='black')
plt.title('Counting numbers of households\nRed: high status, blue')
plt.xlabel('Household Area (sq meter)')
plt.ylabel('Log Count')
# plt.savefig('Export/3-7_Cluster_Statistics/relatin_graphs/Histogram-household_area-rich.png')
plt.savefig('/Users/qitianhu/Desktop/histogram-2.png')
# plt.show()

# %% ########## Household Area Basic analysis -- graph with ideal Gini line ##########
hh = gpd.read_file('Export/Total_Households_info.geojson')
# hh.Area.value_counts()
# we can reduce the thing to 330, 25, 1200
dtf = pd.read_csv('Export/3-7_Cluster_Statistics/Kmeans-k=100.csv')[['Area_mean', 'Area_gini']]
dtf.plot.scatter('Area_mean', 'Area_gini')
a, b, c = hh.Area.min(), hh.Area.mean(), 1250#hh.Area.max()
plt_ideal(a, b)
plt_ideal(b, c)
plt.show()
#%% drawing a randomized case
import scipy.stats

def my_distribution(min_val, max_val, mean, std):
    scale = max_val - min_val
    location = min_val
    # Mean and standard deviation of the unscaled beta distribution
    unscaled_mean = (mean - min_val) / scale
    unscaled_var = (std / scale) ** 2
    # Computation of alpha and beta can be derived from mean and variance formulas
    t = unscaled_mean / (1 - unscaled_mean)
    beta = ((t / unscaled_var) - (t * t) - (2 * t) - 1) / ((t * t * t) + (3 * t * t) + (3 * t) + 1)
    alpha = beta * t
    # Not all parameters may produce a valid distribution
    if alpha <= 0 or beta <= 0:
        raise ValueError('Cannot create distribution for the given parameters.')
    # Make scaled beta distribution with computed parameters
    return scipy.stats.beta(alpha, beta, scale=scale, loc=location)
#%%

md = my_distribution(25, 2015, hh.Area.mean(), 100)
# x = md.rvs(5000)
x = np.array([])
base = 3000
x = np.append(x, np.array([25 for i in range(base*30)] +
                          [326 for i in range(base*50)] +
                          [340 for i in range(base*25)] +
                          [310 for i in range(base*25)] +
                          [2000 for i in range(int(base*2.5))] +
                          [2030 for i in range(int(base*2.5))] +
                          [2015 for i in range(base*10)]))
np.random.shuffle(x)
xs = np.split(x, 5000)
a, b, c = hh.Area.min(), hh.Area.mean(), hh.Area.max()
plt_ideal(a, b)
plt_ideal(b, c)
plt_ideal(a, c)
sns.scatterplot(map(lambda x: sum(x)/len(x), xs), map(gini, xs))
plt.show()

#%% plot Area_mean vs Area_gini with ideal lines
plot_log_with_stat_dense(x='Area_mean', y='Area_gini', gini_ideal=True, file_list=file_list,
                         save_path='/Users/qitianhu/Desktop/',
                         show_plot=False, do_linear_regres=False)













