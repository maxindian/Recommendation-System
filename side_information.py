import tensorflow as tf 
import numpy as np 
import os 
import time
import data_helper
def user():
	user_data_file = 'ml-1m/ml-1m/users.dat'
	user_text = data_helper.load_user_data(user_data_file)
	user_text = np.array(user_text)
	user_text = user_text[:,1:4]
	user_text = user_text.tolist()
	user_all = data_helper.sex_id(user_text)
	#print(user_all)
	user = []
	for line in user_all:
		line1 = []
		line1.append(line[0])
		line1 += line[1]
		line1 += line[2]
		# line.extend(line[0])
		# print(line)
		# line.extend(line[1])
		# print(line)
		# line.extend([line[2]])
		# print(line)
		user.append(line1)
		# exit()
	user_all = np.array(user)
	return user_all


def item():
	movie_data_file='ml-1m/ml-1m/movies.dat'
	item_text = data_helper.load_data(movie_data_file)
	movie1 = []
	arr= []
	i = 1
	line1= []
	for line in item_text:
		if  i==line[0]:
			movie1.append(line)
			line1 = line
		#print(i)
		else:
			a = [0,0,0]
		#print(line[0])
			if line[0]==1404:
				movie1.append(a)
				movie1.append(a)
				movie1.append(line)
			elif line[0]==1453:
				movie1.append(a)
				movie1.append(a)
				movie1.append(line)
			elif line[0]==1493:
				movie1.append(a)
				movie1.append(a)
				movie1.append(line)
			elif line[0]==1507:
				movie1.append(a)
				movie1.append(a)
				movie1.append(line)
			elif line[0]==1639:
				movie1.append(a)
				movie1.append(a)
				movie1.append(line)
			elif line[0]==1738:
				movie1.append(a)
				movie1.append(a)
				movie1.append(line)
			elif line[0]==1804:
				movie1.append(a)
				movie1.append(a)
				movie1.append(line)
			else:
				movie1.append(a)
				movie1.append(line)
	#print(arr)
		i=line[0]+1
	item_text = np.array(movie1)
	item_text = item_text[:,2]
	item_text = item_text.tolist()
	item_all = data_helper.type_id(item_text)
	item_all = np.array(item_all)
	return item_all

a = item()