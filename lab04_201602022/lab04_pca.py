import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from numpy import linalg as LA


def draw_graph(data, figname, mode='normal'):
	plt.figure(figsize=(10, 10))
	if mode == 'pca':
		plt.scatter(data, [0] * len(data), cmap='rainbow')
	else:
		plt.scatter(data[:, 0], data[:, 1], cmap='rainbow')
	plt.savefig(figname)


def get_covariance_matrix(data):
	h, w = data.shape
	for i in range(h):
		mean = np.sum(data[i]) / w
		data[i] -= mean
	covariance_matrix = np.dot(data, data.T) / (w - 1)
	return covariance_matrix


def get_projection_matrix(covariance_matrix, dim):
	eigenvalue, eigenvector = LA.eig(covariance_matrix)
	indices = eigenvalue.argsort()[::-1][:dim]								#내림차순으로 상위 dim개 인덱스만의 배열로 만들어줌.
	projection_matrix = eigenvector.T[indices]								#그 인덱스번째의 eigen vector들로 이뤄진 projection matrix를 만들어 반환
	return projection_matrix


def own_pca(data, dim, figname):
	data = data.T
	covariance_matrix = get_covariance_matrix(data)
	projection_matrix = get_projection_matrix(covariance_matrix, dim)
	own_result = np.dot(projection_matrix, data).T
	draw_graph(own_result, figname, mode='pca')


def sklearn_pca(data, dim, figname):
	pca = PCA(n_components=dim)
	pca_result = pca.fit_transform(data)
	draw_graph(pca_result, figname, mode='pca')


def main():
	data = np.loadtxt(
		fname = 'seoul_student.txt',
		encoding = 'utf-8',
		delimiter = '\t',
		skiprows = 1,
		usecols = [0, 1]
	)

	scaler = MinMaxScaler()
	normalized_data = scaler.fit_transform(data[:])	
	draw_graph(normalized_data, 'normalized_data.png')

	sklearn_pca(normalized_data, 1, 'sklearn_pca_result.png')
	own_pca(normalized_data, 1, 'own_pca_result.png')


if __name__=='__main__':
	main()
