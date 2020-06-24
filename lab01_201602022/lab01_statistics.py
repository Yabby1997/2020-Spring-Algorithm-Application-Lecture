import matplotlib.pyplot as plt
import numpy as np

data = open ('seoul.txt', encoding = 'utf-8')

index_string = data.readline()
index_string = index_string.replace('세', '')
index_string = index_string.replace('이상', '')
index_list = index_string.split()

lengthOfList = len(index_list)

total_list = [0] * lengthOfList
male_list = [0] * lengthOfList
female_list = [0] * lengthOfList

while True :
	data_string = data.readline()
	if data_string == '' :
		data.close()
		break
	
	data_list = data_string.split()	
	sex = data_list[1]

	if sex == '계':
		for i in range(2, lengthOfList) :
			total_list[i] = total_list[i] + int(data_list[i])
	
	elif sex == '남자' :
		for i in range(2, lengthOfList) :
			male_list[i] = male_list[i] + int(data_list[i])
	
	elif sex == '여자' :
		for i in range(2, lengthOfList) :
			female_list[i] = female_list[i] + int(data_list[i])

del index_list[0]
del index_list[0]
del total_list[0]
del total_list[0]
del male_list[0]
del male_list[0]
del female_list[0]
del female_list[0]

data.close()

plt.figure(figsize = (60, 15))

plt.subplot(131)
plt.title('total')
plt.xlabel('age')
plt.ylabel('population')
plt.bar(index_list, total_list, width = 0.5, color = 'green')

plt.subplot(132)
plt.title('male')
plt.xlabel('age')
plt.ylabel('population')
plt.bar(index_list, male_list, width = 0.5, color = 'blue')

plt.subplot(133)
plt.title('female')
plt.xlabel('age')
plt.ylabel('population')
plt.bar(index_list, female_list, width = 0.5, color = 'red')

plt.savefig('result.png')

def showStatistics(label, data) :
	np_data = np.array(data)
	print(label, ": ", end="")
	for number in data :
		print(number, end=" ") 
	print()
	print(label, "총합 :", np.sum(np_data))
	print(label, "평균 :", int(np.mean(np_data)))
	print(label, "분산 :", int(np.var(np_data)))

showStatistics("계", total_list)
showStatistics("남자", male_list)
showStatistics("여자", female_list)
