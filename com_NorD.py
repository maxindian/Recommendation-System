import numpy as np 
import  pickle 
'''
在得到第一个文件的时候，需要得到用户喜欢item的数值，将该数值作为 top_K 的值
在得到第二个文件的时候，需要根据top_K的值选取前top_K个item的下标值
'''
N=np.array([[0,0,0,0,0],[1,1,0,0,1],[0,1,0,1,1],[1,1,0,0,1],[1,0,1,0,1],[1,0,1,0,1],[0,1,0,0,1]])
N1=np.array([[0,0,0,0,0],[0,0,2,0,0],[0,0,0,0,0.1],[0,0,0.4,1],[0,0,0.3,0,0],[0.1,0.2,0.3,0.4,0],[0,0.03,0.02,0.9,0]])
def com_f1(N):
	f1_dic = {}
	user_num = []
	for i in range(N.shape[0]):
		f1_dic[i]=[sum(N[i])]
		user_num.append(sum(N[i]))
		for j in range(N.shape[1]):
			if N[i][j] == 1:
				f1_dic[i].append(j)

	return f1_dic,user_num

#num = [2, 3, 3, 3, 3, 3, 2]
# user_num is the second return value of from com_f1() 
def com_f2(N1,user_num):
	f2_dic = {}
	for i in range(N1.shape[0]):
		f2_dic[i] = []
		sub_dic = {}
		
		for key,eva in enumerate(N1[i]):
			sub_dic[key] = eva
		sub_tuple=sorted(sub_dic.items(),key=lambda item:item[1],reverse=True)
		
		for j in range(user_num[i]):
			f2_dic[i].append(sub_tuple[j][0])
	return f2_dic


def save_file(dict_obj,file_1_2):
	if file_1_2 == 1:
		output = open('Orignial_R.pkl', 'wb')
		pickle.dump(dict_obj, output)
		output.close()
	if file_1_2 == 2:
		output = open('New_R.pkl', 'wb')
		pickle.dump(dict_obj, output)
		output.close()

def main():
	f1_dic,user_num = com_f1(N)
	f2_dic = com_f2(N1,user_num)
	save_file(f1_dic,1)
	save_file(f2_dic,2)

if __name__ == '__main__':
	main()


