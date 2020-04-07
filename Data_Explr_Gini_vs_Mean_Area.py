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
from Data_Explr_Visulization import gini, plot_log_with_stat_dense, plt_ideal
source_path = 'Export/3-7_Cluster_Statistics/'
# %% Define the parametres and names=========


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

hh = gpd.read_file('Export/Total_Households_info.geojson')
# %% ########## Plot graph with ideal Gini line ##########
dtf = pd.read_csv('Export/3-7_Cluster_Statistics/Kmeans-k=100.csv')
dtf.plot.scatter('Area_mean', 'Area_gini')
a, b, c = hh.Area.min(), hh.Area.mean(), hh.Area.max()
plt_ideal(a, b)
plt_ideal(b, c)
plt.show()
# %% 在用 COMPARISON -- real hh data, real cluster size, random grouping by household
areas = list(hh.Area)
title = 'Kmeans-k=50'
dtf = pd.read_csv('Export/3-7_Cluster_Statistics/' + title + '.csv')
random.shuffle(areas)
clusters = []
for n in dtf.num_buildings:
	clusters.append(areas[:n])
	areas = areas[n:]
a, b, c = hh.Area.min(), hh.Area.mean(), hh.Area.max()
plt_ideal(a, b)
plt_ideal(b, c)
# plt_ideal(a, c)
sns.scatterplot(map(lambda _: sum(_)/len(_), clusters),
				map(gini, clusters))
plt.title(title + ' Random Clustering')
plt.show()
# %% 不用 COMPARISON -- real hh data, real cluster size, random grouping by building
"""
note that one building could be split into several clusters here
"""
areas = list(hh.Area)
num_cluster = 50
# group areas according to buildings, to avoid households in the same building
# cluster into different clusters
i, hh_grouped, tmp = 0, [], []
while i < len(hh.Area):
	tmp.append(areas[i])
	# print(i, end='-')
	while i < 15686 and areas[i] == areas[i+1]:
		# print(i, end='-')
		tmp.append(areas[i+1])
		i += 1
	hh_grouped.append(tmp)
	tmp = []
	i += 1
# random.shuffle(hh_grouped)
clusters = []
building_per_cluster = int(len(hh_grouped) / num_cluster)
while len(hh_grouped) > building_per_cluster:
	tmp = hh_grouped[:building_per_cluster]
	clusters.append([j for i in tmp for j in i]) # 这个太他妈骚了，牛逼
	hh_grouped = hh_grouped[building_per_cluster:]
clusters.append(hh_grouped)


# %% COMPARISON -- fake data, random grouping
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
# %% create my own household data

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

#%% 总的 plotPlot Dense Area_mean vs Area_gini with ideal lines
plot_log_with_stat_dense(x='Area_mean', y='Area_gini', gini_ideal=True, file_list=file_list,
						 save_path='/Users/qitianhu/Desktop/',
						 show_plot=False, do_linear_regres=False,
						 random_comp=False)

plot_log_with_stat_dense(x='Area_mean', y='Area_gini', gini_ideal=True, file_list=file_list,
						 save_path='/Users/qitianhu/Desktop/',
						 show_plot=False, do_linear_regres=False)


# %% Create dense 10-subplot graph for Random Comparison
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

# dont_log = ['label', 'Area_gini', 'building_density', 'high', 'mid', 'low', 'status_Gini']

# %% Draw 10-subplot Random Comparison plot -- splitting buildings
fig = plt.figure(figsize=(25, 30), dpi=300)
for i in tqdm.tqdm(range(len(file_list))):
	dtf = pd.read_csv(source_path + file_list[i])
	areas = list(gpd.read_file('Export/Total_Households_info.geojson').Area)
	fig.add_subplot(4, 3, i + 1)
	y = 'Area_gini'
	x = 'Area_mean'
	clusters = []
	for n in dtf.num_buildings:
		if len(areas[:n]) == 0:
			continue
		clusters.append(areas[:n])
		areas = areas[n:]
	data_x = [i for i in map(lambda _: sum(_) / len(_), clusters)]
	data_y = [i for i in map(gini, clusters)]

	sns.scatterplot(x=data_x, y=data_y)
	title_str = file_list[i][:-4]
	a, b, c = 25, 326.29, 2015  # these are min, mean, and max household area
	plt_ideal(a, b)
	plt_ideal(b, c)
	a, b, c = 25, 326.29, 1500  # these are min, mean, and max household area
	plt_ideal(a, b, c='green')
	plt_ideal(b, c, c='green')
	plt.xlabel(x)
	plt.ylabel(y)
	plt.title(title_str)
	# plt.show()
	plt.savefig('/Users/qitianhu/Desktop/Random_Comparison_plot.png')

# %% Draw 10-subplot Random Comparison plot -- Don't Split buildings
res = gpd.read_file('Data/Angela_Oct2019/MergedResidences.shp')
res['Area2'] = res['geometry'].to_crs({'init': 'epsg:3395'}) \
	.map(lambda p: p.area)
res.rename(columns={'Household1': 'Households'}, inplace=True)
small = gpd.read_file('Data/Angela_Oct2019/Small_FeaturesCopy.shp')
small['Area'] = 25

# buildings is a geoDataFrame that should have each entry representing a HOUSEHOLD
buildings = pd.concat([res, small], sort=True)
buildings = buildings[['Area', 'Households', 'Status', 'geometry']]
buildings.index = [i for i in range(buildings.shape[0])]
buildings['num_buildings'] = 1
buildings = buildings.sample(frac=1) # shuffle the rows
# buildings_total_status = buildings.copy()
# buildings_total_status['Status'] = buildings['Status'] * buildings['Households']


#%%
fig = plt.figure(figsize=(25, 30), dpi=300)
for i in tqdm.tqdm(range(len(file_list))):
	dtf = pd.read_csv(source_path + file_list[i])
	# do the random custering work: produce a clusters list, each entry is
	# the households areas in a cluster
	done, clusters = 0, [] # done keeps track of where we are in the buildings.Areas
	for n in dtf['num_buildings']:
		# each cycle add a nother cluster in clusters list
		print(n)
		if n + done > buildings.shape[0]:
			clusters.append(list(buildings.Area[done:]))
			print('break')
			break
		else:
			# building_nums = dtf['num_buildings'][done:done + n]
			tmp = [] # tmp is one cluster
			for j in range(done, done + n): # for each building, split it into households
				tmp += [buildings.Area[j]/buildings.Households[j]
				        for k in range(buildings.Households[j])]
			clusters.append(tmp)
			done += n
	for c in range(len(clusters)):
		if clusters[c] == []:
			clusters = clusters[:c] + clusters[c+1:]
	print('real num of cluster', dtf.shape[0], ' here: ', len(clusters))
	# areas = list(gpd.read_file('Export/Total_Households_info.geojson').Area)
	fig.add_subplot(4, 3, i + 1)
	y = 'Area_gini'
	x = 'Area_mean'
	data_x = [i for i in map(lambda _: sum(_) / len(_), clusters)]
	data_y = [i for i in map(gini, clusters)]

	sns.scatterplot(x=data_x, y=data_y)
	title_str = file_list[i][:-4]
	a, b, c = 25, 326.29, 2015  # these are min, mean, and max household area
	plt_ideal(a, b)
	plt_ideal(b, c)
	a, b, c = 25, 326.29, 1500  # these are min, mean, and max household area
	plt_ideal(a, b, c='green')
	plt_ideal(b, c, c='green')
	plt.xlabel(x)
	plt.ylabel(y)
	plt.title(title_str)
	# plt.show()
	plt.savefig('/Users/qitianhu/Desktop/Random_Comparison_plot-dont_split_building.png')

