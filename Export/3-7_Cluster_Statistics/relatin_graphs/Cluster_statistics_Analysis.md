### Area_gini -- Area_mean 

* the two orange lines in each subgraph are ideal gini lines of minimum household area (25, the value we set for insubstantial structures), mean household area (326, calculated). max household area (2015, calculated). 
* It's surprising to see that in general, the scattered points fit the shape of the ideal gini curves, and this means that in general, each cluster is composed of a number of high, middle, and poor households, and the area of each group is distinctive to others.
* This conclusion is supported by the histogram of households. If we consider bars with more than 10 entries, the whole graph could be characterized by 3 groups, corresponding to high, low, and middle status. 
* For all clustering algorithms (and different parameters), the majority of datapoints are mainly in the "bottom of the valley". Thie corresponds to the fact that the majority of datapoints are 
* [One remaining question]: why is the second ideal curve characterized by maximum household area rather than the mean area for the rich people. 
* Comparing with the purely random case 
  * For the case of purely random cluster structures, the green line (characterized by ideal gini curve of mean household area and the mode of high-status area). So, the reason why it's characterized by the ideal_gini(mean, max) rather than ideal_gini(mean, mode of the rich) max must be the spatial arrangement. 
  * i.e. spatial arrangement of the city consistently rises intra-cluster inequality. 
  * Another point is that for the random case, most of the clusters fall into the "valley," meaning that the spatial arrangement of the city tend to merge poor and rich with the middle class(?) 





### Area_Gini vs. Building_density

* For the nine top methods, the graph shows a L-shape, meaning that most of the building densities are low, and those high-density clusters tend to be more equal than others. This might be due to the fact that the dense clusters in the city center are more equal. 