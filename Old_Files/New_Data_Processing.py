"""
LATEST!
This file deals with all data processing -- which will lead to data export in
/Data/Export
"""
#%%
import geopandas as gpd
import pandas as pd
import numpy as np
import tqdm
from os import walk
from shapely.geometry import Point, Polygon, MultiPolygon, LineString
import matplotlib.pyplot as plt
import random
import matplotlib.pyplot as plt

#%% ########## Generating the list of centroids ##########
res = gpd.read_file('Data/Angela_Oct2019/MergedResidences.shp')
small = gpd.read_file('Data/Angela_Oct2019/Small_FeaturesCopy.shp')

small_centers = np.vstack(small.centroid.apply(lambda x: np.array([x.x, x.y])))
res['centers'] = res.centroid
res_new = pd.DataFrame([])
for i in tqdm.tqdm((range(res.shape[0]))):
	res_new = res_new.append(
		pd.concat([res.iloc[i].to_frame().transpose()] * res.iloc[i]['Household1'])
	)
res_centers = np.vstack(res_new.centers.apply(lambda x: np.array([x.x, x.y])))
np.savetxt('Data/Total_Centroids-added_households.txt',
		   np.vstack([res_centers, small_centers]))

#%% ############# to draw the weighted household graph ##############

# total = res.append(small)
# g = plt.figure()
ax = small.plot(figsize=(20, 20), c='orange', markersize=10,
				marker="x")
res.plot(ax=ax, column="Households", legend=True)
# total.plot(figsize=(20, 20), column="Households", legend=True)
plt.title("Visulizing the graph: color of building is density, "
		  "little corsses are small features")
plt.savefig('Data/NewData-Visulization2')
plt.show()
#%% ############# Export GeoJSON from list of labels ##############
res = gpd.read_file('Data/Angela_Oct2019/MergedResidences.shp')
small = gpd.read_file('Data/Angela_Oct2019/Small_FeaturesCopy.shp')
small['Area'] = 0
res = res.rename(columns={"Household1": "Households"})


def export_geojeon(path="Export/15687labels", method="Kmeans",
				   parameter="k=20"):
	source_path = path + "-" + method + "-" + parameter + ".csv"

	buildings_export = pd.DataFrame([])
	for j in tqdm.tqdm((range(res.shape[0]))):
		buildings_export = buildings_export.append(
			pd.concat([res.iloc[j].to_frame().transpose()] * res.iloc[j]['Households'])
		)
	buildings_export = buildings_export.append(small)

	buildings_export['label'] = np.genfromtxt(source_path)
	buildings_export['label'] = buildings_export['label'].apply(lambda x: int(x))

	gpd.GeoDataFrame(buildings_export).apply(pd.to_numeric, errors='ignore') \
		.to_file(path + "-" + method + "-" + parameter + ".geojson", driver='GeoJSON')

	return 'done'

export_geojeon(path="Export/15687labels",
			   method="DBSCAN",
			   parameter="eps=184")
#%% ################## [OLD] Status Count for each cluster ##################
# Count number of households of each social classs in each neighborhood
# 2 - calcualte Gini coefficient on status

# get a list of status information
res = gpd.read_file('Data/Angela_Oct2019/MergedResidences.shp')
res_new = pd.DataFrame([])
# construct the res with household separated
for i in tqdm.tqdm((range(res.shape[0]))):
	res_new = res_new.append(
		pd.concat([res.iloc[i].to_frame().transpose()] * res.iloc[i]['Household1'])
	)
small = gpd.read_file('Data/Angela_Oct2019/Small_FeaturesCopy.shp')
status_list = pd.DataFrame(pd.concat([res_new['Status'], small['Status']]))
# status_list.set_index(np.arange(len(status_info)))
status_list.set_index(np.arange(len(status_list)), inplace=True)
# list of Status information DONE

# def gini(x):
#     # Mean absolute difference
#     mad = np.abs(np.subtract.outer(x, x)).mean()
#     # Relative mean absolute difference
#     rmad = mad / np.mean(x)
#     # Gini coefficient
#     g = 0.5 * rmad
#     return g


def gini(x, w=None):
	# The rest of the code requires numpy arrays.
	x = np.asarray(x)
	if w is not None:
		w = np.asarray(w)
		sorted_indices = np.argsort(x)
		sorted_x = x[sorted_indices]
		sorted_w = w[sorted_indices]
		# Force float dtype to avoid overflows
		cumw = np.cumsum(sorted_w, dtype=float)
		cumxw = np.cumsum(sorted_x * sorted_w, dtype=float)
		return (np.sum(cumxw[1:] * cumw[:-1] - cumxw[:-1] * cumw[1:]) /
				(cumxw[-1] * cumw[-1]))
	else:
		sorted_x = np.sort(x)
		n = len(x)
		cumx = np.cumsum(sorted_x, dtype=float)
		# The above formula, with all weights equal to 1 simplifies to:
		return (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n

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

def export_status_statistics(status_info, cluster_info_path):
	# cluster_info_path should be a path that ends in xx/xxx/xx.csv
	cluster_info = pd.read_csv(cluster_info_path, header=None)
	assert status_info.shape[0] == cluster_info.shape[0]
	status_info['cluster'] = cluster_info

	status_info['c'] = np.ones(len(status_info))
	# create a column with 1 to do summation

	# aggregate information
	result = status_info.groupby(['cluster', 'Status']).c.count().unstack()

	# create a total column at the end
	# result = result.append(result.sum(numeric_only=True), ignore_index=True)

	# Do Gini analysis
	result.fillna(0, inplace=True)
	result["status_Gini"] = \
		num_to_Gini(low=np.array(result[3]),
					mid=np.array(result[2]),
					high=np.array(result[1]))
	result.to_csv(cluster_info_path[:-4] + "_Status_Gini.csv")


# get a list of filename of labels___.csv, use the cluster label information
# to calculate how many status are there in each cluster
file_list = [i for i in walk('Export/Hidden')][0][2]
# extract all files that contain list of cluster info
file_list = list(filter(lambda x: (len(x) > 10) and
								  (x[-4:] == '.csv') and
								  ("Status_" not in x),
						file_list))

for i in tqdm.tqdm(range(len(file_list))):
	export_status_statistics(
		status_info=status_list,
		cluster_info_path='Export/Hidden/' + file_list[i] # incorporate result from clustering
	)

#%%############### add a column of average status for all the Status_Gini files ####################
# also, rename the columns from 1, 2, 3 to high, mid low
if True:
	file_list = [i for i in walk('Export/Hidden')][0][2]
	# extract all files that contain list of cluster info
	file_list = list(filter(lambda x: (len(x) > 10) and
									  (x[-4:] == '.csv') and
									  ("_Status_Gini.csv" in x),
							file_list))

	for path in file_list:
		df = pd.read_csv('Export/Hidden/' + path)
		df = df.rename(columns = {'1': 'high', '2': 'mid', '3': 'low'})
		df['avg_status'] = (df['high']*3 + df['mid']*2 + df['low']*1)/(df['high']+df['mid']+df['low'])
		df.to_csv('Export/Hidden/' + path)


#%%  CONCAVE HULL -- initilization
import alphashape
# for testing
m = gpd.read_file('Export/15687labels-DBSCAN-eps=111.5.geojson')
c = 1
geojson_path = 'Export/15687labels-Kmeans-k=50.geojson'

paths = ['Export/15687labels-DBSCAN-eps=111.5.geojson',
		 'Export/15687labels-DBSCAN-eps=141.geojson',
		 'Export/15687labels-DBSCAN-eps=184.geojson',
		 'Export/15687labels-Kmeans-k=20.geojson',
		 'Export/15687labels-Kmeans-k=100.geojson',
		 'Export/15687labels-MeanShift-bandwidth=265.geojson',
		 'Export/15687labels-MeanShift-bandwidth=350.geojson',
		 'Export/15687labels-MeanShift-bandwidth=560.geojson']



def export_cluster_contour(geojson_path, alpha=None):
	m = gpd.read_file(geojson_path)
	cluster_label = m.label.unique()
	if -1 in cluster_label:
		cluster_label = np.delete(cluster_label, np.where(cluster_label == -1))

	result = gpd.GeoDataFrame([])

	for i in tqdm.tqdm(cluster_label):
		print(i)
		points = m[m.label == i].geometry.centroid
		if len(points.unique()) < 4:
			# print('cluster: ' + str(i) + 'doesn\'t work')
			continue
		result = result.append({
			'label': i,
			'geometry': alphashape.alphashape(points=points, alpha=alpha)
		}, ignore_index=True)

	result.plot()
	plt.show()

	result.to_file(geojson_path[:-8] + "_cluster_contour.geojson",
				   driver='GeoJSON')
#%%
for path in paths:
	print(c)
	c += 1
	export_cluster_contour(path)

export_cluster_contour('Export/15687labels-DBSCAN-eps=111.5.geojson',
					   alpha=None)

export_cluster_contour('Export/15687labels-Kmeans-k=50.geojson')



#%% #################### Add Gini and avg_status Column to Cluster_contour geoJSON #####################
file_list = [i for i in walk('Export/')][0][2]
hidden_file_list = [i for i in walk('Export/Hidden')][0][2]
# read Ststus_Gini.csv files
Gini_info_list = list(filter(lambda x: (len(x) > 10) and
								  (x[-4:] == '.csv') and
								  ("Status_Gini" in x),
						hidden_file_list))

# read cluster_contour geoJSON files
cluster_contor_list = \
	list(filter(
		lambda x: (len(x) > 10) and ('.geojson' in x) and ("cluster_contour" in x) and ("plsGini" not in x),
		hidden_file_list))

## %% Add column of Gini coefficient to the cluster countour files
# for g in Gini_info_list:
# 	for c in cluster_contor_list:
# 		if c[:-24] == g[:-16]:
# 			print(c)
# 			cluster_contor = gpd.read_file('Export/Hidden/' + c)
# 			Gini_info = pd.read_csv('Export/Hidden/' + g)[['cluster', 'status_Gini', 'avg_status']]
# 			a = cluster_contor.merge(Gini_info, left_on='label', right_on='cluster')
# 			a = a[['label', 'status_Gini', 'avg_status', 'geometry']][a.label != -1]
# 			a.to_file("Export/" + c[:-24] + "_cluster_contour_plsGini.geojson", driver='GeoJSON')


#%% Add column of area Gini onto the cluster_countour files



for g in Gini_info_list:
	for c in cluster_contor_list:
		if c[:-24] == g[:-16]:
			print(c)
			cluster_contor = gpd.read_file('Export/Hidden/' + c)
			Gini_info = pd.read_csv('Export/Hidden/' + g)[['cluster', 'status_Gini', 'avg_status']]
			a = cluster_contor.merge(Gini_info, left_on='label', right_on='cluster')
			a = a[['label', 'status_Gini', 'avg_status', 'geometry']][a.label != -1]
			a.to_file("Export/" + c[:-24] + "_cluster_contour_plsGini.geojson", driver='GeoJSON')

