import pickle
f1 = 'Orignial_R.pkl'
f2 = 'New_R.pkl'


def compute_recall(Orignial_R,New_R):
	pkl_file1 = open(Orignial_R, 'rb')
	data1 = pickle.load(pkl_file1)
	pkl_file1.close()

	pkl_file2 = open(New_R, 'rb')
	data2 = pickle.load(pkl_file2)
	pkl_file2.close()
	toatal_recall = 0 # the sum of all user's recall values 
	for i in range((len(data1))):
		sub_recall = 0 # every user's recall value
		count_n = 0
		taltal_num = int(data1[i][0])
		if taltal_num != 0:
			# len(data2[i])
			for j in range(taltal_num):  # 这里的值可以是taltal_num
				if data2[i][j] in data1[i][1:]:
					count_n += 1
			sub_recall = count_n/taltal_num
			print(sub_recall)
		else:
			sub_recall = 1
		toatal_recall += sub_recall
	recall = toatal_recall/len(data1)
	return recall

if __name__ == '__main__':
	item =  compute_recall(f1,f2)
	print(item)
 


