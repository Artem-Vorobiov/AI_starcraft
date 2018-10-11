
X = {12:(22,33), 100:(77,66), 900:(100,10000)}
print(X)
print(X[900])			#	(100, 10000)
print(type(X[900]))		#	<class 'tuple'>

# print('\n', X[(22,33)])
print('\n', X.items())			#	dict_items([(12, (22, 33)), (100, (77, 66)), (900, (100, 10000))])
print('\n', type(X.items()))	#	<class 'dict_items'>

print('\n', X.keys())			#	dict_keys([12, 100, 900])
print('\n', type(X.items()))	#	<class 'dict_items'>

print('\n', X.values())			#	dict_values([(22, 33), (77, 66), (100, 10000)])
print('\n', type(X.values()))	#	<class 'dict_values'>

Y = list(X.values())
print('\n ', Y[0])				# 	Извлекаю кортеж из Листа
print('\n ', Y[0][1])			#	Извлекаю КОртеж из Листа и из кортежа второй элемент

Practice_1 = {5.70087712549569: (34, 126), 133.91228472399385: (127, 22), 164.85296478983932: (166, 18), 124.85296478983932: (1, 2), 111.222: (30, 30)}


Practice_1_sorted = sorted(k for k in Practice_1)
# Practice_2_sorted = sorted(Practice_1)
# print('\n SORTED:', type(Practice_1_sorted))
# print('\n SORTED:', Practice_1_sorted)
# for dist in Practice_1:
# 	print(dist)
# 	print(type(dist))
count = 1 
for dist in Practice_1_sorted:
	for key, value in Practice_1.items():
		if key == dist:
			print('Step {} \n Key: {}; Value: {}'.format(count,key,value))
			count += 1