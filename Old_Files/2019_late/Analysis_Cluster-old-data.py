import numpy as np
from Analysis_Area import AreaAnalysis
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd


class ClusterAnalysis:
    # def get_spaced_colors(n):
    #     max_value = 16581375  # 255**3
    #     interval = int(max_value / n)
    #     colors = [hex(I)[2:].zfill(6) for I in range(0, max_value, interval)]
    #     return tuple([(int(i[:2], 16), int(i[2:4], 16), int(i[4:], 16)) for i in colors])

    def __init__(self,
                 ctr_points=np.asarray(list(np.genfromtxt('Data/rsd_array_GeometricCentroids.csv', delimiter=',')) +
                                       list(np.genfromtxt('Data/Insubstantial_structures.csv', delimiter=',')))
                 ):
        assert len(ctr_points) == 2454 + 824
        self.ctr = ctr_points
        self.km_labels, self.km_error, self.km_centers = 0, 0, 0

    def kmeans(self, kvalue=10, draw=False, export=False,
               analyze_area=False, SSE_graph=False, compare_with_random=0):

        from sklearn.cluster import KMeans

        km = KMeans(n_clusters=kvalue, init="k-means++", n_init=100,
                    max_iter=3000, tol=1e-7, random_state=None)

        self.km_labels = km.fit_predict(self.ctr)
        self.km_error = km.inertia_
        self.km_centers = km.cluster_centers_

        # For drawing graphs
        lower_lim = 1
        upper_lim = 25
        size = len(self.ctr)

        if draw:
            # import matplotlib.pyplot as plt

            plt.figure(figsize=(15, 15), facecolor='0.6')

            # Draw households
            plt.scatter(self.ctr[:, 0], self.ctr[:, 1], c=self.km_labels, s=5)  # c = sequence of color
            # Draw centers
            plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
                        c='r', s=100)
            plt.title("K-Means Clustering of Residences in Points \n num of Clusters: 10")
            plt.show()

        if export:
            import json
            import fiona

            resMap = fiona.open('Data/Original_Data/Residences.shp')
            insubMap = fiona.open('Data/Original_Data/insubs_Teo/Small_Features.shp')
            assert len(resMap) == 2456
            assert len(insubMap) == 824
            totalMap = []
            for i in resMap:
                totalMap.append(i)
            for i in insubMap:
                totalMap.append(i)
            for i in range(len(totalMap)):
                totalMap[i]['properties']['Id'] = i

            labels = [int(i) for i in self.km_labels]
            kmeans_num_list = labels[:2447] + [-1, -1] + labels[2447:]
            assert len(kmeans_num_list) == len(totalMap)

            # kmeans_num_list = np.genfromtxt("Export/kmeans-total-list-plus2error.csv", delimiter=",")
            # meanshift_num_list = np.genfromtxt("Export/meanshift-total-list-plus2error.csv", delimiter=",")
            # DBSCAN_num_list = np.genfromtxt("Export/DBSCAN-total-list-plus2error.csv", delimiter=",")
            # kmeans_num_list = [int(i) for i in kmeans_num_list]
            # meanshift_num_list = [int(i) for i in meanshift_num_list]
            # DBSCAN_num_list = [int(i) for i in DBSCAN_num_list]
            # assert len(kmeans_num_list) == len(totalMap) == len(DBSCAN_num_list) == len(meanshift_num_list)

            for i in range(len(totalMap)):
                if totalMap[i]['properties']['Id'] == i:
                    totalMap[i]["properties"]['kmeans_num'] = kmeans_num_list[i]
                    totalMap[i]["properties"].pop('Apt_TMP_ID', None)
                    totalMap[i]["properties"].pop('Apt_label', None)
                    totalMap[i]["properties"].pop('Apt_ID', None)
                    totalMap[i]["properties"].pop('Apt_sub_ID', None)
                else:
                    print(i, 'error')

            # Method II: My own method
            # method 2: USE DUMPED FILE
            for i in range(len(totalMap)):
                path = 'Export/individual_shapes/' + str(i) + '.json'
                with open(path, 'w') as file:
                    json.dump(totalMap[i], file)
            # write the files independently
            shape_total = ''

            # combine them mechanically
            for i in range(len(totalMap)):
                path = 'Export/individual_shapes/' + str(i) + '.json'
                with open(path, 'r') as file:
                    shape = str(json.load(file))
                shape_total += shape
            shape_total = '''{ "type": "FeatureCollection","features": [''' + shape_total + ''']}'''

            # Replace all the errors
            double_quote = """ " """[1]
            with open('Export/total_shape_' + 'kmeans-k=' + str(kvalue) + '.json', 'w') as file:
                # with open('Export/total_shape1.json', 'w') as file:
                shape_total = shape_total.replace("None", "null")
                shape_total = shape_total.replace("""'""", double_quote)
                file.write(shape_total)

            # np.savetxt("Export/kmeans-total-list-plus2error.csv",
            #            np.asarray(list(self.km_labels[:2447]) + [-1, -1] +
            #                       list(self.km_labels[2447:])),
            #            delimiter=",")

            print('Export Finishes')

        if analyze_area:
            self.km_area_result = AreaAnalysis(labels=self.km_labels,
                                               ctr_points=self.ctr,
                                               areas=np.genfromtxt('Area_3278.csv', delimiter=','))

        if SSE_graph:
            print('Analyzing SEE')
            distortions = []
            upper_lim = 25
            for i in range(lower_lim, upper_lim):
                km = KMeans(n_clusters=i, init="k-means++", n_init=10,
                            max_iter=3000, tol=1e-7, random_state=None)
                # n_init-跑多少次随机初始值，从中间取最好的；max_iter:设置最大迭代次数；
                # tol:设置算法的容错范围SSE(簇内误平方差);init:random表示使用Kmeans算法，默认是k-means++

                km.fit(self.ctr)
                # get Sum of Squared Errors
                distortions.append(km.inertia_)
                # show
            plt.figure(facecolor='.6')
            plt.plot(range(lower_lim, upper_lim), distortions, marker="o")
            plt.xlabel("No. of Clusters")
            plt.ylabel("Sum of Squared Error (SSE)")
            plt.title('SEE Graph for K-Means')
            plt.show()

        if bool(compare_with_random):
            print('Analyzing SEE and compare with', compare_with_random, 'random sets')

            # Draw authentic line to compare
            print('Evaluating normal SEE graph')
            distortions = []
            for i in range(lower_lim, upper_lim):
                km = KMeans(n_clusters=i, init="k-means++", n_init=100,
                            max_iter=3000, tol=1e-7, random_state=None)
                km.fit(self.ctr)
                # get Sum of Squared Errors
                distortions.append(km.inertia_)

            plt.plot(range(lower_lim, upper_lim), distortions,
                     marker='o', color='blue')

            print('Generating and evaluating random sets')
            # Draw randomly generated stuff
            m = np.mean(self.ctr, axis=0)
            mean_x, mean_y = m[0], m[1]
            s = np.std(self.ctr, axis=0)
            std_x, std_y = s[0], s[1]
            distortions_rand = []
            for i in tqdm(range(compare_with_random)):
                distortions_rand.append([])
                ctr_random = np.transpose(
                    np.array([np.random.normal(mean_x, std_x, size),
                              np.random.normal(mean_y, std_y, size)])
                )

                for j in range(lower_lim, upper_lim):
                    km = KMeans(n_clusters=j, init="k-means++", n_init=20,
                                max_iter=3000, tol=1e-7, random_state=None)
                    km.fit(ctr_random)
                    distortions_rand[-1].append(km.inertia_)

            for i in range(len(distortions_rand)):
                plt.plot(range(lower_lim, upper_lim), distortions_rand[i],
                         marker="o", color='indianred')

            plt.figure(facecolor='.6')
            plt.xlabel("No. of Clusters")
            plt.ylabel("Sum of Squared Error (SSE)")
            plt.title('SEE Graph for K-Means \ncompared with ' + str(compare_with_random) + 'random sets')
            plt.show()

    def meanshift(self, bandwidth=700, draw=False,
                  export=False, analyze_area=False):
        start_time = time.time()
        from sklearn.cluster import MeanShift

        ms = MeanShift(bandwidth=bandwidth)  # bandwidth is radius

        # bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=500)

        ms.fit(self.ctr)
        self.ms_labels = ms.labels_
        num_clusters = len(np.unique(self.ms_labels))
        print('num of clusters:', len(np.unique(self.ms_labels)))
        self.ms_centers = ms.cluster_centers_

        if draw:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(15, 15), facecolor='.6')
            colors = ['r.', 'c.', 'b.', 'k.', 'y.', 'm.', 'g.', 'b.', 'k.', 'y.', 'm.'] * 10

            for i in range(len(self.ctr)):
                plt.plot(self.ctr[i][0], self.ctr[i][1],
                         colors[self.ms_labels[i]], markersize=3)

            # paint center points
            plt.scatter(self.ms_centers[:, 0], self.ms_centers[:, 1],
                        marker='x', s=100, linewidths=0.3, zorder=10)

            plt.title("MeanShift Clustering\nCenters are crosses \n Num of Clusters: %d"
                      % len(np.unique(self.ms_labels)))
            plt.show()

        if export:
            np.savetxt("Export/meanshift-total-list-plus2error.csv",
                       np.asarray(list(self.ms_labels[:2447]) + [-1, -1] +
                                  list(self.ms_labels[2447:])),
                       delimiter=",")

        if analyze_area:
            self.ms_area_result = AreaAnalysis(labels=self.ms_labels,
                                               ctr_points=self.ctr,
                                               areas=np.genfromtxt('Area_3278.csv', delimiter=','))
        print("MeanShift Run Time:" % (time.time() - start_time))

    def DBSCAN(self, eps_value=100, min_samples=4,
               draw=False, export=False, analyze_area=False,
               n_dist_analysis=False):
        start_time = time.time()

        from sklearn.cluster import DBSCAN
        from sklearn import metrics
        from sklearn.preprocessing import StandardScaler

        # X = StandardScaler().fit_transform(self.ctr)
        X = self.ctr

        db = DBSCAN(eps=eps_value, min_samples=min_samples).fit(X)

        # Mark core samples
        self.db_core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        self.db_core_samples_mask[db.core_sample_indices_] = True

        self.db_labels = db.labels_

        self.db_n_clusters = len(set(self.db_labels)) - (1 if -1 in self.db_labels else 0)
        n_noise_ = list(self.db_labels).count(-1)
        print('DBSCAN: Estimated number of clusters: %d' % self.db_n_clusters)
        print('DBSCAN: Estimated number of noise points: %d' % n_noise_)
        print("DBSCAN: Silhouette Coefficient: %0.3f"
              % metrics.silhouette_score(self.ctr, self.db_labels))

        if draw:
            plt.figure(figsize=(15, 15), facecolor='.6')

            # Black removed and is used for noise instead.
            unique_labels = set(self.db_labels)
            colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

            for k, col in zip(unique_labels, colors):
                if k == -1:
                    # Black used for noise.
                    col = [0, 0, 0, 1]
                class_member_mask = (self.db_labels == k)

                xy = X[class_member_mask & self.db_core_samples_mask]
                plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                         markeredgecolor=None, markersize=5)

                xy = X[class_member_mask & ~self.db_core_samples_mask]
                plt.plot(xy[:, 0], xy[:, 1], 'x', markerfacecolor=tuple(col),
                         markeredgecolor=None, markersize=1)

            plt.title(
                'DBSCAN - num of clusters: %d \npoints with cover are core point'
                ', while others are phriphery points' % len(unique_labels) +
                '\neps_vaue = ' + str(eps_value) + ' ;min_samples = ' + str(min_samples)
            )

            plt.show()

        if export:
            np.savetxt("Export/DBSCAN-total-list-plus2error.csv",
                       np.asarray(list(self.db_labels[:2447]) + [-1, -1] +
                                  list(self.db_labels[2447:])),
                       delimiter=",")

        if analyze_area:
            self.db_area_result = AreaAnalysis(labels=self.db_labels,
                                               ctr_points=self.ctr,
                                               areas=np.genfromtxt('Area_3278.csv', delimiter=','))

        if n_dist_analysis:
            from scipy.spatial import distance_matrix
            from scipy.spatial import minkowski_distance

            dm = pd.read_csv('Data/Distance_Matrix_3278.csv')

            # StandardScaler().fit_transform
            # Plot the kth distance graph
            def nth_smallest(pd_series, n):
                return max(pd_series.nsmallest(n))

            def plot_for_n(dataframe, n):
                nth_dists = []
                for i in range(len(dataframe)):
                    nth_dists.append(
                        nth_smallest(dataframe[i], n)
                    )

                nth_dists = sorted(nth_dists)
                plt.plot(nth_dists)
                plt.title('Arrangement for ' + str(n) + 'th Distance')
                plt.xlabel('number of points')
                plt.ylabel(str(n) + 'th Distance')
                plt.show()

            if isinstance(n_dist_analysis, int):
                plot_for_n(dm, n_dist_analysis)
            elif isinstance(n_dist_analysis, list):
                for i in n_dist_analysis:
                    plot_for_n(dm, i)
            else:
                print('Variable \'n_dist_analysis\' type error')

        print("DBSCAN Run Time: ", str(time.time() - start_time))


if __name__ == "__main__":
    a = ClusterAnalysis()

    a.kmeans(kvalue=13, draw=True, SSE_graph=True, compare_with_random=1,
             export=False)
    #
    # a.kmeans(kvalue=15, draw=True)
    # a.meanshift(bandwidth=800, draw=True,
    #               export=False, analyze_area=False)

    # a.DBSCAN(eps_value=100, min_samples=10, draw=True, analyze_area=False)
    # a.DBSCAN(eps_value=150, min_samples=4,

    # a.DBSCAN(eps_value=100, min_samples=4, draw=True,
    #          n_dist_analysis=[i for i in range(5, 15)])

    # a.DBSCAN(draw=True)
    # 800 - 9
    # 700 - 14
    # 600 - 17
    # 500 - 27
    # 400 - 40

    # a.DBSCAN(draw=True)

    # import numpy as np
    # from Analysis_Area import AreaAnalysis
    # import time
    # import matplotlib.pyplot as plt
    # from sklearn.cluster import KMeans

    # Intra-class debugging

    if False:
        import numpy as np
        from Analysis_Area import AreaAnalysis
        import time
        import matplotlib.pyplot as plt
        from sklearn.cluster import KMeans

        ctr_points = np.asarray(list(np.genfromtxt('rsd_array_GeometricCentroids.csv', delimiter=',')) +
                                list(np.genfromtxt('Insubstantial_structures.csv', delimiter=',')))

        m = np.mean(ctr_points, axis=0)
        mean_x, mean_y = m[0], m[1]

        s = np.std(ctr_points, axis=0)
        std_x, std_y = s[0], s[1]

        compare_with_random = 10
        size = 1000
        lower_lim = 1
        upper_lim = 25
        size = 1000

        for i in range(compare_with_random):

            distortions = []
            ctr_random = np.array([
                np.random.normal(mean_x, std_x, size),
                np.random.normal(mean_y, std_y, size)])

            # draw an SSE graph for this
            for j in range(lower_lim, upper_lim):
                km = KMeans(n_clusters=j, init="k-means++", n_init=20,
                            max_iter=3000, tol=1e-7, random_state=None)
                km.fit(np.transpose(ctr_random))
                distortions.append(km.inertia_)
            print(i)
        plt.plot(range(lower_lim, upper_lim), distortions,
                 marker="o", color='indianred')
        plt.show()
        #
        #

        # print('Analyzing SEE')
        # distortions = []
        # upper_lim = 25
        # for i in range(5, upper_lim):
        #     km = KMeans(n_clusters=i, init="k-means++", n_init=10,
        #                 max_iter=3000, tol=1e-7, random_state=None)
        #     # n_init-跑多少次随机初始值，从中间取最好的；max_iter:设置最大迭代次数；
        #     # tol:设置算法的容错范围SSE(簇内误平方差);init:random表示使用Kmeans算法，默认是k-means++
        #
        #     km.fit(ctr_points)
        #     # get Sum of Squared Errors
        #     distortions.append(km.inertia_)
        #     # show
        #
        # plt.figure(facecolor='.6')
        # plt.plot(range(lower_lim, upper_lim), distortions, marker="o")
        # plt.xlabel("No. of Clusters")
        # plt.ylabel("Sum of Squared Error (SSE)")
        # plt.show()

    # Export to geoJSON
    if False:
        import json
        import fiona
        import os

        os.environ['GDAL_DATA'] = \
            '/Library/Frameworks/Python.framework/Versions/3.6/lib/python3.6/site-packages/fiona/gdal_data/'

        # import stat

        # gdal_data = os.environ['GDAL_DATA']
        # print('is dir: ' + str(os.path.isdir(gdal_data)))
        # gcs_csv = os.path.join(gdal_data, 'gcs.csv')
        # print('is file: ' + str(os.path.isfile(gcs_csv)))
        # st = os.stat(gcs_csv)
        # print('is readable: ' + str(bool(st.st_mode & stat.S_IRGRP)))

        resMap = fiona.open('Data/Original_Data/Residences.shp')
        # resMap = fiona.open('/Users/qitianhu/Desktop/a.shp')
        insubMap = fiona.open('Data/Original_Data/insubs_Teo/Small_Features.shp')
        assert len(resMap) == 2456
        assert len(insubMap) == 824
        totalMap = []
        for i in resMap:
            totalMap.append(i)
        for i in insubMap:
            totalMap.append(i)
        for i in range(len(totalMap)):
            totalMap[i]['properties']['Id'] = i

        labels = [int(i) for i in self.km_labels]
        kmeans_num_list = labels[:2447] + [-1, -1] + labels[2447:]
        assert len(kmeans_num_list) == len(totalMap)

        # kmeans_num_list = np.genfromtxt("Export/kmeans-total-list-plus2error.csv", delimiter=",")
        # meanshift_num_list = np.genfromtxt("Export/meanshift-total-list-plus2error.csv", delimiter=",")
        # DBSCAN_num_list = np.genfromtxt("Export/DBSCAN-total-list-plus2error.csv", delimiter=",")
        # kmeans_num_list = [int(i) for i in kmeans_num_list]
        # meanshift_num_list = [int(i) for i in meanshift_num_list]
        # DBSCAN_num_list = [int(i) for i in DBSCAN_num_list]
        # assert len(kmeans_num_list) == len(totalMap) == len(DBSCAN_num_list) == len(meanshift_num_list)

        for i in range(len(totalMap)):
            if totalMap[i]['properties']['Id'] == i:
                totalMap[i]["properties"]['kmeans_num'] = kmeans_num_list[i]
                totalMap[i]["properties"].pop('Apt_TMP_ID', None)
                totalMap[i]["properties"].pop('Apt_label', None)
                totalMap[i]["properties"].pop('Apt_ID', None)
                totalMap[i]["properties"].pop('Apt_sub_ID', None)
            else:
                print(i, 'error')

        # Method II: My own method
        # method 2: USE DUMPED FILE
        for i in range(len(totalMap)):
            path = 'Export/individual_shapes/' + str(i) + '.json'
            with open(path, 'w') as file:
                json.dump(totalMap[i], file)
        # write the files independently
        shape_total = ''

        # combine them mechanically
        for i in range(len(totalMap)):
            path = 'Export/individual_shapes/' + str(i) + '.json'
            with open(path, 'r') as file:
                shape = str(json.load(file))
            shape_total += shape
        shape_total = '''{ "type": "FeatureCollection","features": [''' + shape_total + ''']}'''

        # Replace all the errors
        double_quote = """ " """[1]
        with open('Export/total_shape_' + 'kmeans-k=' + str(kvalue) + '.json', 'w') as file:
            # with open('Export/total_shape1.json', 'w') as file:
            shape_total = shape_total.replace("None", "null")
            shape_total = shape_total.replace("""'""", double_quote)
            file.write(shape_total)
