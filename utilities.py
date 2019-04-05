import numpy as np

def list_to_vec(number_list):
    """
    Accepts a python list of numeric values, and returns a numpy column vector.
    :param number_list:
    :return:
    """
    return np.array([[item] for item in number_list])







if __name__ == '__main__':
    a = [1,2,3]
    b = list_to_vec(a)
    mat = np.array([[1,2,3],[4,5,6]])
    #print(mat@b)
    #print(mat)
    print (np.array(a).T)
    print (b.T)
    print(mat.shape[0])
    vertList = [i + 1 for i in range(mat.shape[0])]
    print (vertList)
    print(mat[0][0])
    indexList = [str(x+1)+ str(y+1) for x, y in np.ndindex(mat.shape) if mat[x,y] != 1]
    print(indexList)
