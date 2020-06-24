import pandas as pd
import numpy as np
import torch.nn as nn
import torch
from torch import optim
import matplotlib.pyplot as plt


def read_data(train_path, test_path):
	train = pd.read_csv(train_path)
	y_train = train['label']
	
	y_list = np.zeros((y_train.size, 10))
			
	for i, y in enumerate(y_train):					#one-hot encoding
		y_list[i][y] = 1
	
	y_train = y_list

	del train['label']
	x_train = train.to_numpy() / 255				#정규화

	test = pd.read_csv(test_path)
	x_test = test.to_numpy() / 255

	return x_train, y_train, x_test


class MNISTModel(nn.Module):
	def __init__(self):
		super(MNISTModel, self).__init__()
		self.fc1 = nn.Linear(784, 512)				#fully-connected layer
		self.fc2 = nn.Linear(512, 512)				#데이터 추상화 
		self.fc3 = nn.Linear(512, 10)

	
	def forward(self, x):
		x1 = torch.relu(self.fc1(x))
		x2 = torch.relu(self.fc2(x1))
		x3 = self.fc3(x2)
		return x3


def get_acc(pred, answer):							#MSE는 차이를 나타내지만 야는 다르
	correct = 0
	for p, a in zip(pred, answer):					#가장큰 값과 그 값의 인덱스를 반환
		pw, pi = p.max(0)
		av, ai = a.max(0)
		if pi == ai:								#같다면 맞은것이므로 correct + 1
			correct += 1

	return correct / len(pred)						#정답이 전체의 얼마인지 반환 


def train(x_train, y_train, batch, lr, epoch):
	model = MNISTModel()
	model.train()

	loss_function = nn.MSELoss(reduction='mean')
	#Mean Square Error 실제 정답과 예측값의 차이를 이용한거. 작을수록 좋음 
	
	optimizer = optim.Adam(model.parameters(), lr=lr)
	
	x = torch.from_numpy(x_train).float()
	y = torch.from_numpy(y_train).float()			#학습에 맞는 데이터로 

	data_loader = torch.utils.data.DataLoader(list(zip(x, y)), batch, shuffle=True)
	#알아서 데이터를 섞고 나눠서 넣어줌
	
	epoch_loss = []
	epoch_acc =[]
	for e in range(epoch):
		total_loss = 0
		total_acc = 0
		for data in data_loader:
			x_data, y_data = data
			#풀이 
			pred = model(x_data)					#forward
			#채점 및 학습
			loss = loss_function(pred, y_data)		#예측값과 실제값 비교
			optimizer.zero_grad()					#이전 학습결과 리셋
			loss.backward()
			#업데이트, 학습반영
			optimizer.step()
			
			total_loss += loss.item()
			total_acc += get_acc(pred, y_data)

		epoch_loss.append(total_loss / len(data_loader))
		epoch_acc.append(total_acc / len(data_loader))
		print("Epoch [%d] Loss: %.3f\tAcc: %.3f" % (e + 1, epoch_loss[e], epoch_acc[e]))

	return model, epoch_loss, epoch_acc


def test(model, x_test, batch):
	model.eval()										#평가모드
	x = torch.from_numpy(x_test).float()
	data_loader = torch.utils.data.DataLoader(x, batch, shuffle=False)

	preds = []
	for data in data_loader:
		pred = model(data)
		for p in pred:
			pv, pi = p.max(0)
			preds.append(pi.item())
	
	return preds


def save_pred(result_path, preds):
	result = pd.DataFrame({'ImageId' : range(1, len(preds) + 1), 'Label' : preds})
	result = result.set_index('ImageId')
	result.to_csv(result_path)


def save_graph(graph_path, epoch_loss, epoch_acc):
	plt.plot(range(len(epoch_loss)), epoch_loss, label='loss')
	plt.plot(range(len(epoch_acc)), epoch_acc, label='accuracy')
	plt.legend()
	plt.savefig(graph_path)


if __name__ == '__main__':
	train_path = 'data/train.csv'
	test_path = 'data/test.csv'
	result_path = 'result.csv'
	figure_path = 'figure.png'

	x_train, y_train, x_test = read_data(train_path, test_path)
	
	batch = 128
	lr = 0.001
	epoch = 10

	model, epoch_loss, epoch_acc = train(x_train, y_train, batch, lr, epoch)
	preds = test(model, x_test, batch)

	save_pred(result_path, preds)
	save_graph(figure_path, epoch_loss, epoch_acc)
