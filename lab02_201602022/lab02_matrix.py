import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn.metrics.pairwise import cosine_distances
from sklearn.preprocessing import MinMaxScaler

def distanceCalc(data_matrix) :
	h, w = data_matrix.shape
	euclidean_result = np.zeros((h, h), dtype = float)
	manhatan_result = np.zeros((h, h), dtype = float)
	cosine_result = cosine_distances(data_matrix, data_matrix)
	for i in range(h) :
		for j in range(i, h) :
			subtracted = data_matrix[i] - data_matrix[j]
	
			euclidean = math.sqrt(np.sum(subtracted * subtracted))
			euclidean_result[i, j] = euclidean
			euclidean_result[j, i] = euclidean
			
			manhatan = np.sum(np.abs(subtracted))
			manhatan_result[i, j] = manhatan
			manhatan_result[j, i] = manhatan
	
	return euclidean_result, manhatan_result, cosine_result

def main() :
	data = np.loadtxt(
		fname = 'seoul_tax.txt',
		encoding = 'utf-8',
		delimiter = '\t',
		skiprows = 1,
		usecols = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
	)
	
	euclidean, manhatan, cosine = distanceCalc(data)

	plt.figure(figsize = (20, 5))
	plt.subplot(1, 3, 1)
	plt.title('Euclidean')
	plt.pcolor(euclidean)
	plt.colorbar()
	plt.subplot(1, 3, 2)
	plt.title('Manhatan')
	plt.pcolor(manhatan)
	plt.colorbar()
	plt.subplot(1, 3, 3)
	plt.title('Cosine')
	plt.pcolor(cosine)
	plt.colorbar()
	plt.savefig('result_basic.png')

	scaler = MinMaxScaler()
	normalized_data = scaler.fit_transform(data[:])
	norm_euclidean, norm_manhatan, norm_cosine = distanceCalc(normalized_data)
	
	plt.figure(figsize = (20, 5))
	plt.subplot(1, 3, 1)
	plt.title('Normalized Euclidean')
	plt.pcolor(norm_euclidean)
	plt.colorbar()
	plt.subplot(1, 3, 2)
	plt.title('Normalized Manhatan')
	plt.pcolor(norm_manhatan)
	plt.colorbar()
	plt.subplot(1, 3, 3)
	plt.title('Normalized Cosine')
	plt.pcolor(norm_cosine)
	plt.colorbar()
	plt.savefig('result_normalized.png')

if __name__ == '__main__':
	main()
