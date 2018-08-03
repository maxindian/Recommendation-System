import numpy as np 

def load_data(movie_data_file):
	movie = open(movie_data_file, "r",encoding="utf-8")
	all_line = movie.read().splitlines()
	movie_tmp=[]
	i = 0
	for line in all_line:
		if i == 0:
			tmp = line.split('::')
			tmp[0] = int('1')
			movie_tmp.append(tmp)
		else:
			tmp = line.split('::')
			tmp[0] = int(tmp[0])
			movie_tmp.append(tmp)
		i = i + 1
	return movie_tmp

def load_movielens_ratings(ratings_file):
    with open(ratings_file) as f:
        ratings = []
        for line in f:
            line = line.split('::')[:3]
            line = [int(l) for l in line]
            #line[0] = int(line[0])
            #line[1] = int(line[1])
            #line[2] = 1
            if line[2]>=5:
            	line[2]=1
            else:
            	line[2]=0
            ratings.append(line)
            #print(line)
            #print(ratings)
        ratings = np.array(ratings)
        #print(ratings)
    return ratings

def construct_R(path_rating):
	raw_ratings = open(path_rating, 'r')
	all_line = raw_ratings.read().splitlines()
	R = np.zeros((6040,3952))
	for line in all_line:
		tmp = line.split('::')
		u = int(tmp[0])-1
		#print(u)
		i = int(tmp[1])-1
		#print(i)
		R[u][i] = int(tmp[2])
	return R

def load_user_data(user_data_file):
	user = open(user_data_file, "r",encoding="utf-8")
	all_line = user.read().splitlines()
	user_tmp=[]
	i = 0
	for line in all_line:
		i = i + 1 
		tmp = line.split('::')
		#tmp = tmp[0] + tmp[1] +tmp[2] + tmp[3] 
		#tmp[0] = int(tmp[0])
		#tmp[0] = float(tmp[0])/6040
		user_tmp.append(tmp)
	print(len(user_tmp))
	return user_tmp
def softmax(x):
    x = np.exp(x)
    x.astype('float32')
    if x.ndim == 1:
        sumcol = sum(x)
        for i in range(x.size):
            x[i] = x[i]/float(sumcol)
    if x.ndim > 1:
        sumcol = x.sum(axis = 0)
        for row in x:
            for i in range(row.size):
                row[i] = row[i]/float(sumcol[i])
    return x

def sex_id(user_text):
	i = 0
	for line in user_text:
		#print(line[1])
		if line[0] == 'F':
		 	line[0] = 0
		else:
		 	line[0] = 1
		line[0] = int(line[0])
		line[1] = int(line[1])
		#line[2] = float(line[2])/35
		#print(line[2])
		zero=[0]*7
		if line[1] == 1:
			zero[0] = 1
		elif line[1] == 18:
			zero[1] = 1
		elif line[1] == 25:
			zero[2] = 1
		elif line[1] == 35:
			zero[3] = 1
		elif line[1] == 45:
			zero[4] = 1
		elif line[1] == 50:
			zero[5] = 1
		else :
			zero[6] = 1
		line[1] = zero
		line[1] = softmax(line[1])
		line[2] = int(line[2])
		zero=[0]*21
		if line[2] == 0:
			zero[0] = 1
		elif line[2] == 1:
			zero[1] = 1
		elif line[2] == 2:
			zero[2] = 1
		elif line[2] == 3:
			zero[3] = 1
		elif line[2] == 4:
			zero[4] = 1
		elif line[2] == 5:
			zero[5] = 1
		elif line[2] == 6:
			zero[6] = 1
		elif line[2] == 7:
			zero[7] = 1
		elif line[2] == 8:
			zero[8] = 1
		elif line[2] == 9:
			zero[9] = 1
		elif line[2] == 10:
			zero[10] = 1
		elif line[2] == 11:
			zero[11] = 1
		elif line[2] == 12:
			zero[12] = 1
		elif line[2] == 13:
			zero[13] = 1
		elif line[2] == 14:
			zero[14] = 1
		elif line[2] == 15:
			zero[15] = 1
		elif line[2] == 16:
			zero[16] = 1
		elif line[2] == 17:
			zero[17] = 1
		elif line[2] == 18:
			zero[18] = 1
		elif line[2] == 19:
			zero[19] = 1
		else :
			zero[20] = 1
		line[2] = zero
		line[2] = softmax(line[2])
		line[1] = line[1].tolist()
		line[2] = line[2].tolist()
	return user_text

def softmax(x):
    x = np.exp(x)
    x.astype('float32')
    if x.ndim == 1:
        sumcol = sum(x)
        for i in range(x.size):
            x[i] = x[i]/float(sumcol)
    if x.ndim > 1:
        sumcol = x.sum(axis = 0)
        for row in x:
            for i in range(row.size):
                row[i] = row[i]/float(sumcol[i])
    return x

def expand_list(nested_list):
    for item in nested_list:
        if isinstance(item, (list, tuple)):
            for sub_item in expand_list(item):
                yield sub_item
        else:
            yield item

def type_id(item_text):
	#global i
	j=0
	count = 0
	voc_list=[]
	type_id1={}
	text={}
	for line in item_text:
		#print(line)
		text=line
		#print(text)
		if text!=0:
			line_dr=text.split('|')
			count = count + len(line_dr)
			for i in range(len(line_dr)):
				voc_list.append(line_dr[i])
	# for line in movie_director_id:
	# 	print(line)
	voc_set = set(voc_list)
	for i in voc_set:
		type_id1[i] = j
		j=j+1
	#print(type_id1)
	movie_id_all=[]
	for line in item_text:
		movid_id=line
		zero=[0]*len(type_id1)
		#print(line)
		text=movid_id
		#print(text)
		# if text==0:
		# 	#text=zero
		# 	movid_id[2]=zero
		# 	print(movid_id[2])
		# 	print(movid_id)
		# else:
		# 	print(1)
		if text!=0:
			line_dr=text.split('|')
			for i in range(len(line_dr)):
				if line_dr[i] in type_id1:
					b=type_id1[line_dr[i]]
					if b == 0:
						zero[0] = 1
					elif b == 1:
						zero[1] = 1
					elif b == 2:
						zero[2] = 1
					elif b == 3:
						zero[3] = 1
					elif b == 4:
						zero[4] = 1
					elif b == 5:
						zero[5] = 1
					elif b == 6:
						zero[6] = 1
					elif b == 7:
						zero[7] = 1
					elif b == 8:
						zero[8] = 1
					elif b == 9:
						zero[9] = 1
					elif b == 10:
						zero[10] = 1
					elif b == 11:
						zero[11] = 1
					elif b == 12:
						zero[12] = 1
					elif b == 13:
						zero[13] = 1
					elif b == 14:
						zero[14] = 1
					elif b == 15:
						zero[15] = 1
					elif b == 16:
						zero[16] = 1
					else :
						zero[17] = 1
			movid_id=zero
		else:
			movid_id=zero
		movie_id_all1 = softmax(movid_id)
		movie_id_all.append(movie_id_all1)
		
	return movie_id_all
