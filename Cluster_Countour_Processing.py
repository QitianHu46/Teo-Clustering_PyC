"""
"""
# %%
import geopandas as gpd
import pandas as pd
import numpy as np
import tqdm
from os import walk
from shapely.geometry import Point, Polygon, MultiPolygon, LineString
import matplotlib.pyplot as plt
import random
import matplotlib.pyplot as plt

# %%  CONCAVE HULL -- define functions
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


# %%  CONCAVE HULL -- run functions
for path in paths:
	print(c)
	c += 1
	export_cluster_contour(path)

# export_cluster_contour('Export/15687labels-DBSCAN-eps=111.5.geojson', alpha=None)
#
# export_cluster_contour('Export/15687labels-Kmeans-k=50.geojson')


# %% prossing Thessian polygon into geojson formet
thess_poly = gpd.read_file('Data/Tim-Thessian_Poly-Feb2020/Plazas_Over10K_Minus_Sun_Thiessen.shp')
thess_poly['label'] = [i for i in range(thess_poly.shape[0])]
thess_poly.to_file('Export/Cluster_contours/Raw/Plazas_Over10K_Minus_Sun_Thiessen.geojson', driver='GeoJSON')

# %% Add info column to cluster_countour -- initilization
info_path = 'Export/3-7_Cluster_Statistics/'
info_list = [i for i in walk(info_path)][0][2]
info_list.remove('.DS_Store')

cluster_contour_path = 'Export/Cluster_contours/Raw/'
cluster_contour_list = [i for i in walk(cluster_contour_path)][0][2]

info_needed = ['building_area', 'total_households', 'total_status', 'num_buildings',
               'cluster_area', 'Area_gini', 'Area_mean', 'building_density',
               'status_Gini', 'avg_status', 'label']

# %% Add info column to cluster_countour -- running

for g in info_list:
	for c in cluster_contour_list:
		if g[:-4] in c:
			print(g)
			cluster_contor = gpd.read_file(cluster_contour_path + c)

			# if 'Plaza' in g:
			# 	if cluster_contor.crs ==
			# 	cluster_contor.crs = {'init': 'epsg:4326'}
			# 	cluster_contor.to_crs({'init': 'epsg:3857'}, inplace=True)
			# else:
			# 	cluster_contor.crs = {'init': 'epsg:3857'}

			if cluster_contor.crs == {'init': 'epsg:3857'}:
				print('ere')
				cluster_contor.to_crs({'init': 'epsg:4326'}, inplace=True)
			# if 'Plaza' not in g:


			Gini_info = pd.read_csv(info_path + g)[info_needed]

			to_export = cluster_contor.merge(Gini_info, left_on='label', right_on='label')
			# to_export = to_export[['label', 'status_Gini', 'avg_status', 'geometry']][to_export.label != -1]
			to_export.to_file("Export/Cluster_contours/" + g[:-4] + "_contour_info.geojson", driver='GeoJSON')
