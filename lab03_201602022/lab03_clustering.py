import matplotlib.pyplot as plt
import numpy as np
import random
import math
import copy
from collections import OrderedDict
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering


class KMean:
	def __init__(self, data, n):
		self.data = data
		self.n = n
		self.cluster = OrderedDict()
	

	def init_center(self):
		index = random.randint(0, self.n)
		index_list = []
		for i in range(self.n):
			while index in index_list:
				index = random.randint(0, self.n)
			index_list.append(index)
			self.cluster[i] = {'center' : self.data[index], 'data' : []}


	def clustering(self, cluster):
		i = 0
		centroids = np.empty((0, 2), dtype=float)
		for key, value in cluster.items():				 
			centroids = np.append(centroids, np.reshape(value['center'], (1, 2)), axis=0)
		
		distance_matrix = euclidean_distance(centroids, self.data)
		cluster_number = np.argmin(distance_matrix, axis=1)
		
		for j in range(self.n):
			self.cluster[j]['data'] = []
		
		for number in cluster_number:
			self.cluster[number]['data'].append(self.data[i])
			i = i + 1
		
		return self.cluster	
	

	def update_center(self):
		for i in range(self.n):
			mean_point = np.mean(self.cluster[i]['data'], axis=0)
			self.cluster[i]['center'] = mean_point

	
	def update(self):
		prev_cluster = OrderedDict()
		update_continue = True
		iteration = 0
		while update_continue:
			prev_cluster = copy.deepcopy(self.cluster)
			self.update_center()
			self.cluster = self.clustering(self.cluster)
			update_continue = self.continue_test(prev_cluster)
			print(iteration)
			iteration = iteration + 1


	def fit(self):
		self.init_center()
		self.cluster = self.clustering(self.cluster)
		self.update()
		result, labels = self.get_result(self.cluster)
		draw_graph(result, labels, 'result_kmean.png')


	def get_result(self, cluster):
		result = []
		labels = []
		for key, value in cluster.items():
			for item in value['data']:
				labels.append(key)
				result.append(item)
		return np.array(result), labels
	
	
	def continue_test(self, prev_cluster):
		counter = 0
		for i in range(self.n):
			prev_centroid = np.array(prev_cluster[i]['center'])
			curr_centroid = np.array(self.cluster[i]['center'])
			print(prev_centroid, end='\t')
			print(curr_centroid, end='\t result : ')
			if np.array_equal(prev_centroid, curr_centroid):
				print('matched!')
				counter = counter + 1
			else :
				print('not matched!')
		return counter != self.n


def euclidean_distance(centroid, data) :
	dh, dw = data.shape
	ch, cw = centroid.shape
	distance_matrix = np.empty((0, ch), dtype=float)
	for i in range(dh):
		subtracted = centroid - data[i]
		distance_list = []
		for j in range(ch):
			distance = math.sqrt(np.sum(subtracted[j] * subtracted[j]))
			distance_list.append(distance)
		distance_array = np.array(distance_list)
		distance_matrix = np.append(distance_matrix, np.reshape(distance_array, (1, ch)), axis=0)
	return distance_matrix


def draw_graph(data, labels, fig_name):
	plt.figure(figsize=(10, 10))
	plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='rainbow')
	plt.savefig(fig_name)


def main():
	data = np.loadtxt(
		fname = 'covid-19.txt',
		encoding = 'utf-8',
		delimiter = '\t',
		skiprows = 1,
		usecols = [5, 6]
	)
	
	scaler = MinMaxScaler()
	normalized_data = scaler.fit_transform(data[:])

	dbscan = DBSCAN(eps=0.1, min_samples=2)
	dbscan_clusters = dbscan.fit_predict(normalized_data)
	draw_graph(normalized_data, dbscan_clusters, 'result_dbscan.png')

	aggcls = AgglomerativeClustering(n_clusters=8, affinity='Euclidean', linkage='complete')
	aggcls_clusters = aggcls.fit_predict(normalized_data)
	draw_graph(normalized_data, aggcls_clusters, 'result_aggcls.png')
	
	kmean = KMean(data=normalized_data, n=8)
	kmean.fit()

if __name__ == '__main__' :
	main()
