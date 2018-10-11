
print('\n')

#	Первый вариант

def fn1():
    print("One")

def fn2():
    print("Two")

def fn3():
    print("Three")

fndict = {"A": fn1, "B": fn2, "C": fn3}

keynames = ["A", "B", "C"]

fndict[keynames[1]]()
fndict['C']()


#		Второй вариант

def add(one,two):
	c = one+two
	print(c)
	print(type(c))

def sub(one,two):
	c = one-two
	print(c)
	print(type(c))

trainee = {1:add, 2:sub}

trainee[1](10,4)
print('\n PROVERKA TIPA', type(trainee[1]))
print('\n PROVERKA TIPA', type(trainee[1](10,4)))
