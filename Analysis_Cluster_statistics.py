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

# source_path = 'Export/Hidden/'
source_path = 'Export/Cluster_contours/Raw/'

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


# %% Load file lists


file_list = [i for i in walk(source_path)][0][2]
# extract all files that contain list of cluster info
# list of cluster contours
geo_list = list(filter(lambda x: (len(x) > 10) and ((".geojson" in x) or (".shp" in x)), file_list))
# the code to process Thessian polygon into geojson
# thess_plg = gpd.read_file('Data/Tim-Thessian_Poly-Feb2020/Plazas_Over10K_Minus_Sun_Thiessen.shp')
# thess_plg.to_file('Export/Hidden/Plazas_Over10K_Minus_Sun_Thiessen.shp')

# %% Read building map

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

buildings_total_status = buildings.copy()
buildings_total_status['Status'] = buildings['Status'] * buildings['Households']
# %% Make the Households 耗时一分钟
re_calculate_households = False
if re_calculate_households:
	households = pd.DataFrame()
	for row in tqdm.tqdm(buildings.iterrows()):
		# print("f")
		for i in range(row[1]['Households']):
			households = households.append(row[1])
	# buildings = pd.concat(buildings, pd.DataFrame(row[1]))
	households['Area'] = households['Area'] / households['Households']
	households["Num"] = [i for i in range(households.shape[0])]
	households.set_index('Num', inplace=True)
	# households['num_households'] = households['num_buildings']
	# del households['num_buildings']
	households = gpd.GeoDataFrame(households)
	households.crs = buildings.crs
	households.to_file('Export/Total_Households_info.geojson', driver='GeoJSON')
else:
	households = gpd.read_file('Export/Total_Households_info.geojson')


# %% Main Data Processing
export_statistics = True
for geo_name in tqdm.tqdm(geo_list):
	# add cluster area column
	clusters = gpd.read_file(source_path + geo_name)
	clusters.crs = {'init': 'epsg:3857'}
	clusters['cluster_area'] = clusters['geometry'].to_crs({'init': 'epsg:3395'}).map(lambda p: p.area)
	if not ('label' in clusters.columns):
		clusters['label'] = [i for i in range(clusters.shape[0])]
	# SETUP add building area column and producing the final output
	joined = gpd.sjoin(buildings_total_status, clusters, op='intersects', how='left')
	to_export = joined.groupby('label').sum()

	# GINI INDEX BY AREA
	joined = gpd.sjoin(households, clusters, op='intersects', how='left')
	# produce area_gini for each cluster
	area_gini_count = {}
	area_mean_count = {}
	unique_labels = list(clusters.label.unique())
	if -1 in unique_labels:
		unique_labels.remove(-1)
	for i in unique_labels:
		if i == -1:
			print("-1 encounters")
			continue
		# for each cluster, add the area of each household into the list
		area_gini_count[i] = gini(np.array(joined.groupby('label').get_group(i).Area))
		area_mean_count[i] = np.mean(np.array(joined.groupby('label').get_group(i).Area))
	to_export['Area_gini'] = to_export.index.map(area_gini_count)
	to_export['Area_mean'] = to_export.index.map(area_mean_count)

	# STATUS LABEL GINI
	joined = gpd.sjoin(buildings, clusters, op='intersects', how='left')
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

	# TODO: setup distance to avenue of death

	# add back the low, mid, high info
	status_info = pd.DataFrame()
	status_info['label'] = [i for i in unique_labels]
	status_info['high'] = [status_count[i][1] for i in unique_labels]
	status_info['mid'] = [status_count[i][2] for i in unique_labels]
	status_info['low'] = [status_count[i][3] for i in unique_labels]
	status_info['status_Gini'] = \
		num_to_Gini(low=status_info['low'], mid=status_info['mid'], high=status_info['high'])

	to_export.rename(columns={"Area": "building_area", 'Status': 'total_status', 'Households': 'total_households'},
	                 inplace=True)
	# del to_export['index_right'], to_export['label']
	to_export['building_density'] = to_export['building_area'] / to_export['cluster_area']

	to_export = pd.merge(to_export, status_info, on='label')
	# to_export.drop(to_export[to_export['label'] == -1].index, inplace=True)
	# to_export.to_csv('/Users/qitianhu/Desktop/' + SG_name[12:-16] + '.csv')
	# 锦上添花
	to_export['avg_status'] = \
		(to_export['mid'] * 2 + to_export['low'] * 1 + to_export['high'] * 3) / to_export['total_households']

	assert sum(to_export['high'] + to_export['mid'] + to_export['low'] == \
	           to_export['total_households']) == to_export.shape[0]
	if export_statistics:
		if 'Plazas' in geo_name:
			to_export.to_csv('Export/3-7_Cluster_Statistics/' + geo_name[:-8] + '.csv')
		else:
			to_export.to_csv('Export/3-7_Cluster_Statistics/' + geo_name[12:-24] + '.csv')




#  现在不用！Tim's Thessian Polygon Analysis
# geo_list = ['Data/Tim-Thessian_Poly-Feb2020/Plazas_Over10K_Minus_Sun_Thiessen.shp']
# geo_name = 'Plazas_Over10K_Minus_Sun_Thiessen'
# # add cluster area column
# clusters = gpd.read_file('Data/Tim-Thessian_Poly-Feb2020/Plazas_Over10K_Minus_Sun_Thiessen.shp')
# clusters['label'] = [i for i in range(clusters.shape[0])]
# clusters.crs = {'init': 'epsg:3857'}
# clusters['cluster_area'] = clusters['geometry'].to_crs({'init': 'epsg:3395'}).map(lambda p: p.area)
#
# joined = gpd.sjoin(buildings, clusters, op='intersects', how='left')
#
# # produce Status_gini for each cluster
# status_count = {}
# unique_labels = list(clusters.label.unique())
# if -1 in unique_labels:
# 	unique_labels.remove(-1)
# for i in unique_labels:
# 	if i == -1:
# 		break
# 	# for each cluster, calculate the number of each status
# 	status_in_cluster = joined.groupby('label').get_group(i).Status.unique()
# 	single_status_count = {1: 0, 2: 0, 3: 0}
# 	for status_num in range(1, 4):
# 		# if a certain status is present in this cluster, do analysis
# 		if status_num in status_in_cluster:
# 			single_status_count[status_num] = \
# 				joined.groupby('label').get_group(i).groupby('Status').get_group(status_num).Households.sum()
# 	status_count[i] = single_status_count
# # here it's still the original labels, so 3 is low and 1 is high
#
# # add building area column and producing the final output
# joined = gpd.sjoin(buildings_total_status, clusters, op='intersects', how='left')
# to_export = joined.groupby('label').sum()
# to_export.rename(columns={
# 	"Area": "building_area", 'Status': 'total_status', 'Households': 'total_households'
# }, inplace=True)
# to_export['building_density'] = to_export['building_area'] / to_export['cluster_area']
#
# # add back the low, mid, high info
# status_info = pd.DataFrame()
# status_info['label'] = [i for i in unique_labels]
# status_info['high'] = [status_count[i][1] for i in unique_labels]
# status_info['mid'] = [status_count[i][2] for i in unique_labels]
# status_info['low'] = [status_count[i][3] for i in unique_labels]
#
# status_info['status_Gini'] = num_to_Gini(low=status_info['low'],
#                                          mid=status_info['mid'],
#                                          high=status_info['high'])
#
# to_export = pd.merge(to_export, status_info, on='label')
# # to_export.drop(to_export[to_export['label'] == -1].index, inplace=True)
# # to_export.to_csv('/Users/qitianhu/Desktop/' + SG_name[12:-16] + '.csv')
# # 锦上添花
# to_export['avg_status'] = \
# 	(to_export['mid'] * 2 + to_export['low'] * 1 + to_export['high'] * 3) / to_export['total_households']
#
# assert sum(to_export['high'] + to_export['mid'] + to_export['low'] == \
#            to_export['total_households']) == to_export.shape[0]
# to_export.to_csv('Export/3-7_Cluster_Statistics' + 'Plazas_Over10K_Minus_Sun_Thiessen' + '.csv')
