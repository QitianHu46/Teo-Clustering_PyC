"""
This file contains all analysis related to the Gini vs. average Status graph and related analysis
Exports like "Add Gini and avg_status Column to Cluster_contour geoJSO"
is in New_Data_Processing.py

Feb 24, 2020 update:
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


# %% ########## Define Gini and Plot_Ideal ##########
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


def plt_ideal(a, b):
	f = lambda X, a, b: (abs(a - b) * X * (1 - X)) / ((1 - X) * a + X * b)
	X = np.arange(0, 1, 0.0001)
	Y = f(X, a, b)
	plt.scatter(X * (b - a) + a, Y,
				s=1, alpha=.8, c='orange')

def plot_log_with_stat(x, y, files, xlabel='x', ylabel='y' ,title='title', plotit=True):
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
		# plt.set(xlabel=xlabel, ylabel=ylabel)
		# line_x = ax.get_lines()[0].get_xdata()
		# line_y = ax.get_lines()[0].get_ydata()
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
			 + x + '_vs_' + y + '_'+ files[0].split('=')[0] + '.png')
	return


def plot_log_with_stat_dense(x, y, file_list, xlabel='x', ylabel='y' ,title='title', plotit=True):
	"""
	plot on single graph, using scipy and seaborn, with 95% CI and relevant statistics
	:param x:
	:param y:
	:param xlabel:
	:param ylabel:
	:return: the object of the graph
	"""
	dont_log = ['label', 'Area_gini', 'building_density', 'high', 'mid', 'low', 'status_Gini']
	dont_log = all_cols
	assert len(file_list) == 10
	num = len(file_list)
	fig = plt.figure(figsize=(25, 30), dpi=300)
	for i in range(10):
		dtf = pd.read_csv(source_path + file_list[i])[[x, y]]
		fig.add_subplot(4, 3, i + 1)
		remove_notice = ''
		if x in dont_log:
			data_x = dtf[x]
			x_label = x
		else:
			# if (0 in dtf[x].values) or (0 in dtf[y].values):
			# 	dtf = dtf.replace(0, np.nan)
			# 	dtf = dtf.dropna()
			# 	remove_notice = '  !! Zero Term Removed!!  '
			data_x = np.log(dtf[x])
			x_label = 'log ' + x
		if y in dont_log:
			data_y = dtf[y]
			y_label = y
		else:
			# if (0 in dtf[x].values) or (0 in dtf[y].values):
			# 	dtf = dtf.replace(0, np.nan)
			# 	dtf = dtf.dropna()
			# 	remove_notice = '  !! Zero Term Removed!!  '
			# 	if ('gini' in x) or ('density' in x):
			# 		data_x = dtf[x]
			# 	else:
			# 		data_x = np.log(dtf[x])
			data_y = np.log(dtf[y])
			y_label = 'log ' + y

		sns.regplot(x=data_x, y=data_y, ci=95, fit_reg=True)
		plt.xlabel(x_label)
		plt.ylabel(y_label)
		# plt.set(xlabel=xlabel, ylabel=ylabel)
		# line_x = ax.get_lines()[0].get_xdata()
		# line_y = ax.get_lines()[0].get_ydata()
		slope, intercept, corr_coeff, p_value, std_err = scipy.stats.linregress(data_x, data_y)
		info = list(map(lambda k: round(k, 3), [slope, intercept, corr_coeff ** 2, p_value, std_err]))
		title_str = file_list[i][:-4] + remove_notice + \
		            '\nslope: ' + str(info[0]) + ', intercept: ' + str(info[1]) + \
		            ',r squared:' + str(info[2])
		plt.title(title_str)
	# ax.plot()
	if plotit:
		plt.show()
	else:
		plt.savefig('Export/3-7_Cluster_Statistics/relatin_graphs/'
			 + x + '_|_' + y + '.png')
	return
#%% DEBUGGING
i = 7
x = 'Area_gini'
y = 'Area_mean'

plt.savefig('/Users/qitianhu/Desktop/a.png')

from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(np.array(dtf[x]).reshape(-1,1), np.array(dtf[y]).reshape(-1,1))


#%% initialize path
file_list = [i for i in walk(source_path)][0][2]
if '.DS_Store' in file_list:
	file_list.remove('.DS_Store')

# %% ############ PLOT Gini vs. Avg_status -- Separate #############
exporting = False
use_svg = False

# Plot graphs in file_list
for path in tqdm.tqdm(file_list):
	df = pd.read_csv(source_path + path)
	ax = df.plot(x='avg_status', y='status_Gini', style='o')
	plt.ylim((0, .25))
	plt.xlim((1, 3))
	plt.title(path[:-4])
	plt.ylabel('Gini Coefficient')
	plt.xlabel('Average Status of a Cluster')
	ax.get_legend().remove()

	plt_ideal(1, 2)
	plt_ideal(2, 3)
	# plt_ideal(1, 3)
	if exporting and use_svg:
		plt.savefig('Export/Gini_status/' + path[:-4] +
					'_Gini_vs_AvgStatus.svg', format='svg')
	if exporting and (not use_svg):
		plt.savefig('Export/Gini_status/' + path[:-4] +
					'_Gini_vs_AvgStatus.png')
	else:
		plt.show()


# %%Create 3 subplots for KMeans
km_list = ['Kmeans-k=20.csv',
		   'Kmeans-k=50.csv',
		   'Kmeans-k=100.csv']
fig = plt.figure(figsize=(10, 4), dpi=1000)

for i in range(3):
	df = pd.read_csv(source_path + km_list[i])
	# plt.subplot(1, 3, i+1, figsize=(15,15))
	fig.add_subplot(1, 3, i + 1)
	plt_ideal(1, 2)
	plt_ideal(2, 3)
	plt.scatter(x=df['avg_status'], y=df['status_Gini'],
				c=None, s=20)
	plt.ylim((0, .2))
	plt.xlim((1, 3))
	# plt.title(path[12:-16])
	plt.text(1.65, 0.17, km_list[i][:-4])
	plt.ylabel('Gini Coefficient')
	plt.xlabel('Average Status of a Cluster')
fig.suptitle(t='Gini-Avgerage Status Graph for K-Means')
plt.subplots_adjust(wspace=.5, hspace=1)  # 调整子图间距
# fig.tight_layout(rect=[.5, .5, .5, .5], h_pad=.5)
# plt.show()
if exporting:
	plt.savefig('Export/Gini_status/Combined-Kmeans.jpg', format='jpg')
else:
	plt.show()

# %% PLOT different attributes of clusters against each other

"""
Luis on Mar 1, 2020
- the labels in some are either unnecessary or don’t look good (for a paper):
the Gini by average status plots look good for now. Note that these labels are nicely readable.
- In the figures for the clusters I think you need to use some transparency or different colors, as  in many cases the green dots are overwhelming the colors of the clusters
also we will need a spacial scale, the other labels (# of clusters) are not essential and don’t look good at the moment (weird font …)

-For the plots of quantities versus total households: 
They look a bit rough. i recommend that you use logarithmic axes on both axes, which will make the numbers smaller. 
Then we also need a linear fit (in log log) for each plot, with a 95% Confidence Interval on the two parameters 
of the fit (intercept and slope) and an R^2 for the fit  … 
also show the best fit line in the plot… can you do that ?
"""
figsize = (9, 3.7)
dpi = 200
fig, axs = plt.subplots(1, 3, figsize=figsize, sharey=False, dpi=dpi)
export_path = 'Export/Result_graph/'

# exporting = False
col_x = 'total_households'
col_y = 'cluster_area'
seq = '1-'

fig, axs = plt.subplots(1, 3, figsize=figsize, sharey=False, dpi=dpi)
# fig.suptitle('Cluster Area Against Number of Households')
for i in range(0, 3):
	df = pd.read_csv(source_path + file_list[i])
	axs[i].scatter(df[col_x], df[col_y])
	axs[i].title.set_text(file_list[i][:-4])
	if i == 1:
		axs[i].set_xlabel(col_x)
	if i == 0:
		axs[i].set_ylabel(col_y)

plt.savefig(export_path + seq + 'Kmeans-' + col_y + '-vs-' + col_x)

fig, axs = plt.subplots(1, 3, figsize=figsize, sharey=False, dpi=dpi)
for i in range(3, 6):
	df = pd.read_csv(source_path + file_list[i])
	axs[i - 3].scatter(df[col_x], df[col_y])
	axs[i - 3].title.set_text(file_list[i][:-4])
	if i == 4:
		axs[i - 3].set_xlabel(col_x)
	if i == 3:
		axs[i - 3].set_ylabel(col_y)
plt.savefig(export_path + seq + 'MeanShift-' + col_y + '-vs-' + col_x)

fig, axs = plt.subplots(1, 3, figsize=figsize, sharey=False, dpi=dpi)
for i in range(6, 9):
	df = pd.read_csv(source_path + file_list[i])
	axs[i - 6].scatter(df[col_x], df[col_y])
	axs[i - 6].title.set_text(file_list[i][:-4])
	if i == 7:
		axs[i - 6].set_xlabel(col_x)
	if i == 6:
		axs[i - 6].set_ylabel(col_y)
plt.savefig(export_path + seq + 'DBSCAN-' + col_y + '-vs-' + col_x)

#%% Plot other
col_x = 'total_households'
col_y = 'building_area'
seq = '2-'
if True:
	fig, axs = plt.subplots(1, 3, figsize=figsize, sharey=False, dpi=dpi)
	for i in range(0, 3):
		df = pd.read_csv(source_path + file_list[i])
		axs[i].scatter(df[col_x], df[col_y])
		axs[i].title.set_text(file_list[i][:-4])
		if i == 1:
			axs[i].set_xlabel(col_x)
		if i == 0:
			axs[i].set_ylabel(col_y)
	plt.savefig(export_path + seq + 'Kmeans-' + col_y + '-vs-' + col_x)

	fig, axs = plt.subplots(1, 3, figsize=figsize, sharey=False, dpi=dpi)
	for i in range(3, 6):
		df = pd.read_csv(source_path + file_list[i])
		axs[i - 3].scatter(df[col_x], df[col_y])
		axs[i - 3].title.set_text(file_list[i][:-4])
		if i == 4:
			axs[i - 3].set_xlabel(col_x)
		if i == 3:
			axs[i - 3].set_ylabel(col_y)
	plt.savefig(export_path + seq + 'MeanShift-' + col_y + '-vs-' + col_x)

	fig, axs = plt.subplots(1, 3, figsize=figsize, sharey=False, dpi=dpi)
	for i in range(6, 9):
		df = pd.read_csv(source_path + file_list[i])
		axs[i - 6].scatter(df[col_x], df[col_y])
		axs[i - 6].title.set_text(file_list[i][:-4])
		if i == 7:
			axs[i - 6].set_xlabel(col_x)
		if i == 6:
			axs[i - 6].set_ylabel(col_y)
	plt.savefig(export_path + seq + 'DBSCAN-' + col_y + '-vs-' + col_x)

col_x = 'num_buildings'
col_y = 'building_density'
seq = '3-'
if True:
	fig, axs = plt.subplots(1, 3, figsize=figsize, sharey=False, dpi=dpi)
	for i in range(0, 3):
		df = pd.read_csv(source_path + file_list[i])
		axs[i].scatter(df[col_x], df[col_y])
		axs[i].title.set_text(file_list[i][:-4])
		if i == 1:
			axs[i].set_xlabel(col_x)
		if i == 0:
			axs[i].set_ylabel(col_y)
	plt.savefig(export_path + seq + 'Kmeans-' + col_y + '-vs-' + col_x)

	fig, axs = plt.subplots(1, 3, figsize=figsize, sharey=False, dpi=dpi)
	for i in range(3, 6):
		df = pd.read_csv(source_path + file_list[i])
		axs[i - 3].scatter(df[col_x], df[col_y])
		axs[i - 3].title.set_text(file_list[i][:-4])
		if i == 4:
			axs[i - 3].set_xlabel(col_x)
		if i == 3:
			axs[i - 3].set_ylabel(col_y)
	plt.savefig(export_path + seq + 'MeanShift-' + col_y + '-vs-' + col_x)

	fig, axs = plt.subplots(1, 3, figsize=figsize, sharey=False, dpi=dpi)
	for i in range(6, 9):
		df = pd.read_csv(source_path + file_list[i])
		axs[i - 6].scatter(df[col_x], df[col_y])
		axs[i - 6].title.set_text(file_list[i][:-4])
		if i == 7:
			axs[i - 6].set_xlabel(col_x)
		if i == 6:
			axs[i - 6].set_ylabel(col_y)
	plt.savefig(export_path + seq + 'DBSCAN-' + col_y + '-vs-' + col_x)

col_x = 'avg_status'
col_y = 'building_density'
seq = '4-'
if True:
	fig, axs = plt.subplots(1, 3, figsize=figsize, sharey=False, dpi=dpi)
	for i in range(0, 3):
		df = pd.read_csv(source_path + file_list[i])
		axs[i].scatter(df[col_x], df[col_y])
		axs[i].title.set_text(file_list[i][:-4])
		if i == 1:
			axs[i].set_xlabel(col_x)
		if i == 0:
			axs[i].set_ylabel(col_y)
	plt.savefig(export_path + seq + 'Kmeans-' + col_y + '-vs-' + col_x)

	fig, axs = plt.subplots(1, 3, figsize=figsize, sharey=False, dpi=dpi)
	for i in range(3, 6):
		df = pd.read_csv(source_path + file_list[i])
		axs[i - 3].scatter(df[col_x], df[col_y])
		axs[i - 3].title.set_text(file_list[i][:-4])
		if i == 4:
			axs[i - 3].set_xlabel(col_x)
		if i == 3:
			axs[i - 3].set_ylabel(col_y)
	plt.savefig(export_path + seq + 'MeanShift-' + col_y + '-vs-' + col_x)

	fig, axs = plt.subplots(1, 3, figsize=figsize, sharey=False, dpi=dpi)
	for i in range(6, 9):
		df = pd.read_csv(source_path + file_list[i])
		axs[i - 6].scatter(df[col_x], df[col_y])
		axs[i - 6].title.set_text(file_list[i][:-4])
		if i == 7:
			axs[i - 6].set_xlabel(col_x)
		if i == 6:
			axs[i - 6].set_ylabel(col_y)
	plt.savefig(export_path + seq + 'DBSCAN-' + col_y + '-vs-' + col_x)

#%% Texting

x = np.linspace(-np.pi, np.pi, 256)
y = []
for i in range(0, 7):
 y += [np.cos(x + i)]

plt.plot(x, y[0], color='red', linewidth=2.5, linestyle='-', label='linestyle="_"')
plt.plot(x, y[1], color='blue', linewidth=5, alpha=0.5, linestyle='-', label='lines tyle="-"')
plt.plot(x, y[2], color='#aa0000', linewidth=1, linestyle='--', label='linestyle="--"')
plt.plot(x, y[3], color='black', linestyle=':', label='linestyle=":"')
plt.plot(x, y[4], color='black', linewidth=2, linestyle='-.', label='linestyle="-."')

plt.legend()

plt.show()
# # %% ############ Do a comparison: what if randomly assign cluster? ##########
# res = gpd.read_file('Data/Angela_Oct2019/MergedResidences.shp')
# res_new = pd.DataFrame([])
# # as a comparison -- randomly assign labels and draw the graph
# # get a list of status information
# # construct the res with household separated
# for i in tqdm.tqdm((range(res.shape[0]))):
# 	res_new = res_new.append(
# 		pd.concat([res.iloc[i].to_frame().transpose()] * res.iloc[i]['Household1'])
# 	)
# small = gpd.read_file('Data/Angela_Oct2019/Small_FeaturesCopy.shp')
# status_list = pd.DataFrame(pd.concat([res_new['Status'], small['Status']]))
# # status_list.set_index(np.arange(len(status_info)))
# status_list.set_index(np.arange(len(status_list)), inplace=True)
# status_list.Status = status_list.Status.apply(lambda x: int(x))
#
# status_list_save = status_list
# status_list = status_list_save
#
# # what if the status is also randomly assigned?
# if True:
# 	status_list = pd.DataFrame({'Status': np.random.randint(1, 4, size=15687)})
#
# 	side_num = 100
# 	status_list = pd.DataFrame({'Status': [1 for _ in range(386)] +
# 										  [2 for _ in range(14587)] +
# 										  [3 for _ in range(714)]})
#
# num_cluster = 1000
# gini_and_avgstatus = pd.DataFrame([], columns=['gini', 'avg_status'])
# status_list['label'] = np.random.randint(0, num_cluster, status_list.shape[0])
# avg_status_list = status_list.groupby('label').Status.mean()
#
# for i in range(0, num_cluster):
# 	gini_and_avgstatus = gini_and_avgstatus.append({
# 		'gini': (gini(status_list[status_list['label'] == i]['Status'])),
# 		'avg_status': avg_status_list[i]
# 	}, ignore_index=True)
# gini_and_avgstatus.plot(x='avg_status', y='gini', style='o')
# plt.title('Randomly assign: ' + str(num_cluster) + ' clusters')
# plt.show()
#%% PLOTTING WITH NEW FUNCTION--iterate thru pairs of variables
for pair in x_y_set:
	var_x = pair[0]
	var_y = pair[1]
	plot_log_with_stat(x=var_x, y=var_y, files=ms_s, plotit=False)
	plot_log_with_stat(x=var_x, y=var_y, files=km_s, plotit=False)
	plot_log_with_stat(x=var_x, y=var_y, files=DB_s, plotit=False)
	plot_log_with_stat(x=var_x, y=var_y, files=plz, plotit=False)
	plt.close('all')

#%% PLOTTING in congreged manner
for pair in x_y_set:
	var_x = pair[0]
	var_y = pair[1]
	plot_log_with_stat_dense(x=var_x, y=var_y, file_list=file_list, plotit=False)
	plt.close('all' )


 #%%
x_y_set = [
('building_area', 'Area_gini'),
('building_area', 'building_density'),
('building_area', 'high'),
# ('building_area', 'avg_status'),
('total_households', 'cluster_area'),
('total_households', 'Area_gini'),
('total_households', 'high'),
('total_households', 'avg_status'),
('Area_gini', 'Area_mean'),
('Area_gini', 'building_density'),
('Area_gini', 'high'),
('Area_gini', 'low'),
('Area_gini', 'avg_status'),
('high', 'low'),
('building_density', 'high')
           ]

all_cols = ['Unnamed: 0', 'label', 'building_area', 'total_households',
       'total_status', 'num_buildings', 'cluster_area',
       'Area_gini', 'Area_mean', 'building_density', 'high', 'mid', 'low',
       'status_Gini', 'avg_status']

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


