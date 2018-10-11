#		TypeError: 'set' object does not support indexing

X = {1,2,3,4,5}
print(type(X))

Y = list(X)
print('\n', type(Y))
print(Y[1])