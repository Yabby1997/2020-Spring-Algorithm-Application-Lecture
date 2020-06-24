import numpy as np
import copy

INITIAL_SCORE = 1
INITIAL_SEED = 42
ITERATION = 10
QUANTITY = 5
K = 3

greedy_result = [[], INITIAL_SCORE]
beam_result = []
beam_temp = []

def random_generator(seed, quantity):
	np.random.seed(seed)
	generated =  np.random.rand(quantity)
	print("SEED :", seed, "GENERATED :", generated)
	return generated


def greedy_search(rand_array):
	print("\n===============GREEDY STARTED!===============")
	for i in range(ITERATION):
		current_max = rand_array.max()
		current_selection = rand_array.argmax()
		greedy_result[0].append(current_selection)
		greedy_result[1] *= current_max
		
		if i == ITERATION - 1:
			break
		else:
			new_seed = int(greedy_result[1] * 100)
			rand_array = random_generator(new_seed, QUANTITY)
	print("===============GREEDY DONE!===============")


def beam_search(rand_array):
	print("\n===============BEAM STARTED!===============")
	values = rand_array
	for i in range(ITERATION):
		indices = np.argsort(values)[::-1]
		beam_result.clear()
		
		for j in range(K):
			index = indices[j]
			if i == 0:
				beam_result.append([[index], rand_array[index]])
			else:
				beam_result.append(beam_temp[index])
			print('CURRENT BEST SEQUENCE', j, ':', beam_result[j])
		
		if i == ITERATION - 1:
			break
		else:
			new_seed = int(beam_result[0][1] * 100)
			rand_array = random_generator(new_seed, QUANTITY)
		
		beam_temp.clear()
		for result in beam_result:
			for rand in rand_array:
				sequence = copy.deepcopy(result[0])	
				sequence.append(np.where(rand_array==rand)[0][0])
				beam_temp.append([sequence, result[1] * rand])
		values = [each[1] for each in beam_temp]
	print("===============BEAM DONE!===============")


if __name__=='__main__':
	initial = random_generator(INITIAL_SEED, QUANTITY)
	print("==========INITIAL RANDOM GENERATED==========")
	greedy_search(initial)
	beam_search(initial)
	print("\n\n###############<GREEDY RESULT>###############")
	print(greedy_result)
	print("###############<BEAM RESULT>###############")
	for result in beam_result:
		print(result)
