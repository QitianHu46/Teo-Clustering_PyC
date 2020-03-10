"""
Serves as replacement of New_Data_Processing.py = more efficient categorize and analhysis
to cluster statistics
this file is made in Feb 2020 to aggregate all the statistics there are
to each clusters.
Main Goal: Add columns Total area and Building area to each file of Status_Gini
and export them to Export/2-20_Cluster_Statistics

the old gini analysis is falty, so I will redo it here

Notes: avg_status has been changed, status_Gini

Mar 7 Note: not use cluster label to calculate Gini, but use areas! --
	now exporting to Export/3-7_Cluster_Statistics
"""
# %% Import and Path
import geopandas as gpd
import pandas as pd
import numpy as np
import tqdm
from os import walk
from shapely.geometry import Point, Polygon, MultiPolygon, LineString
import matplotlib.pyplot as plt
import random

source_path = 'Export/Hidden/'


# %% Define gini Functions

def gini(x):
	# Mean absolute difference
	mad = np.abs(np.subtract.outer(x, x)).mean()
	# Relative mean absolute difference
	rmad = mad / np.mean(x)
	# Gini coefficient
	g = 0.5 * rmad
	return g

# def gini(x, w=None):
# 	# The rest of the code requires numpy arrays.
# 	x = np.asarray(x)
# 	if w is not None:
# 		w = np.asarray(w)
# 		sorted_indices = np.argsort(x)
# 		sorted_x = x[sorted_indices]
# 		sorted_w = w[sorted_indices]
# 		# Force float dtype to avoid overflows
# 		cumw = np.cumsum(sorted_w, dtype=float)
# 		cumxw = np.cumsum(sorted_x * sorted_w, dtype=float)
# 		return (np.sum(cumxw[1:] * cumw[:-1] - cumxw[:-1] * cumw[1:]) /
# 				(cumxw[-1] * cumw[-1]))
# 	else:
# 		sorted_x = np.sort(x)
# 		n = len(x)
# 		cumx = np.cumsum(sorted_x, dtype=float)
# 		# The above formula, with all weights equal to 1 simplifies to:
# 		return (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n


def num_to_Gini(low, mid, high):
	# This function converts the number of low, mid and high status in each cluster
	# to a Gini coefficient
	# IMPORTANT: 3 is used in shapefile to denote LOW status, but I use it in calculate to
	# calcualte HIGH ststus
	# input must be 3 arrays
	r = []
	for i in range(len(low)):
		r.append(gini(
			np.array([[1] * int(low[i]) + [2] * int(mid[i]) + [3] * int(high[i])])
		))
	return r


# %% Initialization

file_list = [i for i in walk('Export/Hidden')][0][2]
# extract all files that contain list of cluster info
SG_list = list(filter(lambda x: (len(x) > 10) and
								(x[-4:] == '.csv') and
								("_Status_Gini.csv" in x),
					  file_list))
# SG_list[2][:-16] is the part that is informative

# list of
geo_list = list(filter(lambda x: (len(x) > 10) and
								 ("geojson" in x),
					   file_list))
# [:-24] is the informative part

res = gpd.read_file('Data/Angela_Oct2019/MergedResidences.shp')
res['Area'] = res['geometry'].to_crs({'init': 'epsg:3395'}) \
	.map(lambda p: p.area)
res.rename(columns={'Household1': 'Households'}, inplace=True)
small = gpd.read_file('Data/Angela_Oct2019/Small_FeaturesCopy.shp')
small['Area'] = 10

buildings = pd.concat([res, small], sort=True)
buildings = buildings[['Area', 'Households', 'Status', 'geometry']]
buildings.index = [i for i in range(buildings.shape[0])]
buildings['num_buildings'] = 1

buildings_total_status = buildings.copy()
buildings_total_status['Status'] = buildings['Status'] * buildings['Households']
#%% Main Data Processing
for SG_name in tqdm.tqdm(SG_list):
	for geo_name in geo_list:
		if geo_name[:-24] == SG_name[:-16]:
			break
	# add cluster area column
	clusters = gpd.read_file(source_path + geo_name)
	# info = pd.read_csv(source_path + SG_name)
	# info.rename(columns={'Unnamed: 0': 'label'}, inplace=True)
	clusters.crs = {'init': 'epsg:3857'}
	clusters['cluster_area'] = clusters['geometry'].to_crs({'init': 'epsg:3395'}).map(lambda p: p.area)

	joined = gpd.sjoin(buildings, clusters, op='intersects', how='left')

	# produce Status_gini for each cluster
	status_count = {}
	unique_labels = list(clusters.label.unique())
	if -1 in unique_labels:
		unique_labels.remove(-1)
	for i in unique_labels:
		if i == -1:
			break
		# for each cluster, calculate the number of each status
		status_in_cluster = joined.groupby('label').get_group(i).Status.unique()
		single_status_count = {1: 0, 2: 0, 3: 0}
		for status_num in range(1, 4):
			# if a certain status is present in this cluster, do analysis
			if status_num in status_in_cluster:
				single_status_count[status_num] = \
					joined.groupby('label').get_group(i).groupby('Status').get_group(status_num).Households.sum()
		status_count[i] = single_status_count
	# here it's still the original labels, so 3 is low and 1 is high

	# add building area column and producing the final output
	joined = gpd.sjoin(buildings_total_status, clusters, op='intersects', how='left')
	# info2 = joined.groupby('label').sum()

	# to_export = pd.merge(info, info2, on='label')
	to_export = joined.groupby('label').sum()
	# to_export['label'] = to_export.index
	to_export.rename(columns={
		"Area": "building_area", 'Status': 'total_status', 'Households': 'total_households'
	}, inplace=True)
	# del to_export['index_right'], to_export['label']
	to_export['building_density'] = to_export['building_area'] / to_export['cluster_area']

	# add back the low, mid, high info
	status_info = pd.DataFrame()
	status_info['label'] = [i for i in unique_labels]
	status_info['high'] = [status_count[i][1] for i in unique_labels]
	status_info['mid'] = [status_count[i][2] for i in unique_labels]
	status_info['low'] = [status_count[i][3] for i in unique_labels]

	status_info['status_Gini'] = num_to_Gini(low=status_info['low'],
											 mid=status_info['mid'],
											 high=status_info['high'])

	to_export = pd.merge(to_export, status_info, on='label')
	# to_export.drop(to_export[to_export['label'] == -1].index, inplace=True)
	# to_export.to_csv('/Users/qitianhu/Desktop/' + SG_name[12:-16] + '.csv')
	# 锦上添花
	to_export['avg_status'] = \
		(to_export['mid'] * 2 + to_export['low'] * 1 + to_export['high'] * 3) / to_export['total_households']

	assert sum(to_export['high'] + to_export['mid'] + to_export['low'] == \
			   to_export['total_households']) == to_export.shape[0]
	to_export.to_csv('Export/2-20_Cluster_Statistics/' + SG_name[12:-16] + '.csv')


#%% Tim's Thessian Polygon Analysis
geo_list = ['Data/Tim-Thessian_Poly-Feb2020/Plazas_Over10K_Minus_Sun_Thiessen.shp']
geo_name = 'Plazas_Over10K_Minus_Sun_Thiessen'
# add cluster area column
clusters = gpd.read_file('Data/Tim-Thessian_Poly-Feb2020/Plazas_Over10K_Minus_Sun_Thiessen.shp')
clusters['label'] = [i for i in range(clusters.shape[0])]
clusters.crs = {'init': 'epsg:3857'}
clusters['cluster_area'] = clusters['geometry'].to_crs({'init': 'epsg:3395'}).map(lambda p: p.area)

joined = gpd.sjoin(buildings, clusters, op='intersects', how='left')

# produce Status_gini for each cluster
status_count = {}
unique_labels = list(clusters.label.unique())
if -1 in unique_labels:
	unique_labels.remove(-1)
for i in unique_labels:
	if i == -1:
		break
	# for each cluster, calculate the number of each status
	status_in_cluster = joined.groupby('label').get_group(i).Status.unique()
	single_status_count = {1: 0, 2: 0, 3: 0}
	for status_num in range(1, 4):
		# if a certain status is present in this cluster, do analysis
		if status_num in status_in_cluster:
			single_status_count[status_num] = \
				joined.groupby('label').get_group(i).groupby('Status').get_group(status_num).Households.sum()
	status_count[i] = single_status_count
# here it's still the original labels, so 3 is low and 1 is high

# add building area column and producing the final output
joined = gpd.sjoin(buildings_total_status, clusters, op='intersects', how='left')
to_export = joined.groupby('label').sum()
to_export.rename(columns={
	"Area": "building_area", 'Status': 'total_status', 'Households': 'total_households'
}, inplace=True)
to_export['building_density'] = to_export['building_area'] / to_export['cluster_area']

# add back the low, mid, high info
status_info = pd.DataFrame()
status_info['label'] = [i for i in unique_labels]
status_info['high'] = [status_count[i][1] for i in unique_labels]
status_info['mid'] = [status_count[i][2] for i in unique_labels]
status_info['low'] = [status_count[i][3] for i in unique_labels]

status_info['status_Gini'] = num_to_Gini(low=status_info['low'],
										 mid=status_info['mid'],
										 high=status_info['high'])

to_export = pd.merge(to_export, status_info, on='label')
# to_export.drop(to_export[to_export['label'] == -1].index, inplace=True)
# to_export.to_csv('/Users/qitianhu/Desktop/' + SG_name[12:-16] + '.csv')
# 锦上添花
to_export['avg_status'] = \
	(to_export['mid'] * 2 + to_export['low'] * 1 + to_export['high'] * 3) / to_export['total_households']

assert sum(to_export['high'] + to_export['mid'] + to_export['low'] == \
		   to_export['total_households']) == to_export.shape[0]
to_export.to_csv('Export/2-20_Cluster_Statistics/' + 'Plazas_Over10K_Minus_Sun_Thiessen' + '.csv')
