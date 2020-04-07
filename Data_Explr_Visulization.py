"""
This file contains all analysis related to the Gini vs. average Status graph and related analysis
Exports like "Add Gini and avg_status Column to Cluster_contour geoJSO"
is in New_Data_Processing.py

Feb 24, 2020 update:

This file also contains some follow-up analysis I do to explore the patterns of the
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

# file_list = [i for i in walk(source_path)][0][2]
# if '.DS_Store' in file_list:
# 	file_list.remove('.DS_Store')

file_list = \
	[
		'DBSCAN-eps=184.csv',
		'DBSCAN-eps=141.csv',
		'DBSCAN-eps=111.5.csv',
		'Kmeans-k=20.csv',
		'Kmeans-k=50.csv',
		'Kmeans-k=100.csv',
		'MeanShift-bandwidth=560.csv',
		'MeanShift-bandwidth=350.csv',
		'MeanShift-bandwidth=265.csv',
		'Plazas_Over10K_Minus_Sun_Thiessen.csv']

dont_log_cols = ['Unnamed: 0', 'label', 'building_area', 'total_households',
                 'total_status', 'num_buildings', 'cluster_area',
                 'Area_gini', 'Area_mean', 'building_density', 'high', 'mid', 'low',
                 'status_Gini', 'avg_status']


# %% ########### PLOTTING ########## Define plotting functions: Gini, Plot_Ideal, plot_log_with_stat
def gini(x):
	# Mean absolute difference
	mad = np.abs(np.subtract.outer(x, x)).mean()
	# Relative mean absolute difference
	rmad = mad / np.mean(x)
	# Gini coefficient
	g = 0.5 * rmad
	return g


def plt_ideal(a, b, c='orange'):
	"""
	draw an ideal reference curve for graph of mean area vs. Gini
	:param a:
	:param b:
	:return:
	"""
	f = lambda X, a, b: (abs(a - b) * X * (1 - X)) / ((1 - X) * a + X * b)
	X = np.arange(0, 1, 0.0001)
	Y = f(X, a, b)
	plt.scatter(X * (b - a) + a, Y, s=1, alpha=.8, c=c)


def plot_log_with_stat(x, y, files, xlabel='x', ylabel='y', title='title', plotit=True):
	"""
	plot on single graph, using scipy and seaborn, with 95% CI and relevant statistics
	:param x:
	:param y:
	:param xlabel:
	:param ylabel:
	:return: the object of the graph
	"""
	num = len(files)
	fig = plt.figure(figsize=(6 * num, 6), dpi=1000)
	for i in range(num):
		dtf = pd.read_csv(source_path + files[i])
		fig.add_subplot(1, num, i + 1)
		sns.regplot(x=np.log(dtf[x]), y=np.log(dtf[y]), ci=95, fit_reg=True)
		plt.xlabel(x)
		plt.ylabel(y)
		slope, intercept, corr_coeff, p_value, std_err = scipy.stats.linregress(np.log(dtf[x]), np.log(dtf[y]))
		info = list(map(lambda k: round(k, 3), [slope, intercept, corr_coeff ** 2, p_value, std_err]))
		title_str = files[i][:-4] + '\nslope: ' + str(info[0]) + ', intercept: ' + str(info[1]) + \
		            ', \nr squared:' + str(info[2])
		plt.title(title_str)
	# ax.plot()
	if plotit:
		plt.show()
	else:
		plt.savefig('Export/3-7_Cluster_Statistics/relatin_graphs/'
		            + x + '_vs_' + y + '_' + files[0].split('=')[0] + '.png')
	return


def plot_log_with_stat_dense(x, y, file_list, xlabel='x', ylabel='y', title='title',
                             show_plot=True, gini_ideal=False, source_path='Export/3-7_Cluster_Statistics/',
                             save_path=None, do_linear_regres=True, random_comp=False):
	"""
	plot on single graph, using scipy and seaborn, with 95% CI and relevant statistics
	:param random_comp: to use the num_building info to produce a randomized comparison to check
		the real effect of clustering algorithm.
	:param do_linear_regres:
	:param save_path: the path to save the graph to
	:param show_plot: if True then show the plot; if False save it as file
	:param source_path: directory of the data sources
	:param gini_ideal: whether to plot the ideal gini curve
	:param x: name of variable to plot on x-axis
	:param y: name of variable to plot on y-axis
	:param xlabel:
	:param ylabel:
	:return: the object of the graph
	"""
	# dont_log = ['label', 'Area_gini', 'building_density', 'high', 'mid', 'low', 'status_Gini']
	dont_log = dont_log_cols
	assert len(file_list) == 10
	fig = plt.figure(figsize=(25, 30), dpi=300)
	if random_comp:
		areas = list(gpd.read_file('Export/Total_Households_info.geojson').Area)
	for i in range(len(file_list)):
		dtf = pd.read_csv(source_path + file_list[i])
		fig.add_subplot(4, 3, i + 1)
		remove_notice = ''
		# log data if necessary
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

		if random_comp: # random comparison for the gini vs. mean graph
			assert y == 'Area_gini' and x == 'Area_mean'
			clusters = []
			for n in dtf.num_buildings:
				clusters.append(areas[:n])
				areas = areas[n:]
			data_x = [i for i in map(lambda _: sum(_) / len(_), clusters)]
			data_y = [i for i in map(gini, clusters)]

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
			a, b, c = 25, 326.29, 2015  # these are min, mean, and max household area
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
			if random_comp:
				plt.savefig(save_path + x + '_|_' + y + '_Random_Comparison' + '.png')
			else:
				plt.savefig(save_path + x + '_|_' + y + '.png')
			print('saved to new path!')
		else:
			plt.savefig('Export/3-7_Cluster_Statistics/relatin_graphs/' + x + '_|_' + y + '.png')
			print('saved to deafult path!')
	return

#%%
if __name__ == "__main__":
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

	# %% Do PLOTTING in congreged manner
	for pair in x_y_set:
		var_x = pair[0]
		var_y = pair[1]
		plot_log_with_stat_dense(x=var_x, y=var_y, file_list=file_list, show_plot=False)
		plt.close('all')

	# %% ########## plot HISTOGRAM of household area ##########
	households = gpd.read_file('Export/Total_Households_info.geojson')
	households.Area.value_counts()

	# %% pltting using matplotlib
	plt.hist(households[households.Status == 2]['Area'], log=True, bins=50)
	plt.hist(households[households.Status == 1]['Area'], log=True, bins=200, color='red')
	plt.hist(households[households.Status == 3]['Area'], log=True, bins=1, color='black')
	plt.title('Counting numbers of households\nRed: high status, blue: mid status. Low status ignored (=25)')
	plt.xlabel('Household Area (sq meter)')
	plt.ylabel('Log Count')
	# plt.savefig('Export/3-7_Cluster_Statistics/relatin_graphs/Histogram-household_area-rich.png')
	plt.savefig('/Users/qitianhu/Desktop/histogram-2.png')
plt.show()
