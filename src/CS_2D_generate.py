import numpy as np

def CS_data_generate(no_of_clusters,odds_matrix,total_no_of_points):
    mean_x_matrix=500*np.random.randn(1,no_of_clusters)
    mean_y_matrix=500*np.random.randn(1,no_of_clusters)
    var_x_matrix=60*np.abs(np.random.randn(1,no_of_clusters))
    var_y_matrix=60*np.abs(np.random.randn(1,no_of_clusters))
    
    data_matrix_with_lables=np.zeros(((np.ceil(total_no_of_points/np.sum(odds_matrix)))*np.sum(odds_matrix)),3)
    
    l=1
    while l<=len(data_matrix_with_lables):
        for j in range(1,no_of_clusters):
            for k in range(1,odds_matrix[j]):
                data_matrix_with_lables[l,:]=[mean_x_matrix[j]+var_x_matrix[j]*np.random.randn(1) mean_y_matrix[j]+var_y_matrix[j]*np.random.randn(1) j]
                l=l+1
    random_permutation=np.random.permutation(len(data_matrix_with_lables))
    data_matrix_with_lables=data_matrix_with_lables[random_permutation,:]
    
    dist_matrix=np.zeros(len(data_matrix_with_lables),len(data_matrix_with_lables))
    len,wid =dist_matrix.shape

    for l in range(1,len):
        diff_vector=data_matrix_with_lables[:,1:2]-np.append(data_matrix_with_lables(l,1)*np.ones(len,1),data_matrix_with_lables(l,2)*np.ones(len,1))
        dist_matrix[l,:]=np.abs(diff_vector[:,1]+1j*diff_vector[:,2])
    
    return data_matrix_with_lables,dist_matrix
