import pandas as pd
import numpy as np

# reading the csv_files
symp = pd.read_csv('../Dataset/symptoms.csv')

diag = pd.read_csv('../Dataset/diagnosis.csv')

weightage = pd.read_csv('../Generated_Weights/weightage_matrix.csv', index_col = 0, names = np.array(symp['Symptom_ID']), header = 0)

scored_weights = pd.read_csv('../Generated_Weights/scored_weights.csv', index_col = 0, names = np.array(symp['Symptom_ID']), header = 0)


# taking inputs of symptoms and processing it
print('Enter the symptoms,separated by single ","s and space, you feel from the below listed ones')
print('-----------------------------------------------------------------------------------')
print(list(symp['Symptom']),'\n')

inp_sym_list = input().split(', ')

inp_sym_id = [symp[symp.Symptom.str.casefold() == x.casefold()].Symptom_ID.item() for x in inp_sym_list]


buf1 = (weightage/weightage).fillna(0)

buf2= {}
for x in np.array(symp['Symptom_ID']):
    a = [1.0,1.0,1.0] if x in inp_sym_id else [0.0,0.0,0.0]
    buf2[x] = a

buf2 = pd.DataFrame(buf2, columns = np.array(symp['Symptom_ID']), index = np.array(diag['Diagnose_ID']))

buf3 = buf1.mul(buf2)

# finding scores for input sysmptoms
fin_score = scored_weights.mul(buf3)


fin_score['final'] = fin_score.sum(1)

sort_di_id = fin_score['final'].sort_values(ascending = False).index

# getting the diagnosis in descending order
for x in sort_di_id:
    print(diag[diag['Diagnose_ID']==x].Diagnose.item())








