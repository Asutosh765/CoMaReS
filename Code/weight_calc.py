import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import svds
from sklearn.metrics.pairwise import cosine_similarity

# reading data from weightage.csv
weightage = pd.read_csv('../Dataset/weightage.csv')


# coverting the weightage dataframe to a sparse matrix
temp_weig = weightage['Weightage'].astype('float')
temp_sympid = weightage['Symp_ID'].astype('category')
temp_diagid = weightage['Diag_ID'].astype('category')

weig_mat = coo_matrix((temp_weig,(temp_diagid.cat.codes,temp_sympid.cat.codes)))


# getting the unique sympid and diagid from the dataframe from weightage.csv 
uni_sympid = weightage.groupby('Symp_ID').size()
uni_diagid = weightage.groupby('Diag_ID').size()


# coverting the weightage sparse matrix to a dataframe and writing to a .csv file
weig_mat_df = pd.DataFrame.sparse.from_spmatrix(weig_mat,columns = list(uni_sympid.index), index = list(uni_diagid.index))
weig_mat_df.to_csv('../Generated_Weights/weightage_matrix.csv')


# defining the bm_25 algorithm
def bm_25_weight(mat, k1=1.4, b=0.85):

    N = float(mat.shape[0])

    idf = np.log((N+1)/(np.bincount(mat.col)+0.5))
    
    
    fieldlen = mat.sum(1)
    avg_fieldlen = fieldlen.sum()/N

    len_norm = k1*(1-b+b*(fieldlen/avg_fieldlen))

    
    weights = np.array(idf)*(mat.toarray()*(k1+1))/(mat+len_norm)
    
    return weights


# getting the bm25 scores for the sparse matrix
bm25_scored_weights = bm_25_weight(weig_mat)


U, S, V = svds(bm25_scored_weights, k =2)

scored_weights = pd.DataFrame(cosine_similarity(U, V.T), columns = list(uni_sympid.index), index = list(uni_diagid.index))

print(scored_weights)

scored_weights.to_csv('../Generated_Weights/scored_weights.csv')



