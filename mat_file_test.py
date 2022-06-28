import scipy.io
import pandas as pd
mat = scipy.io.loadmat('test_author.mat') 

print(mat)
print(mat['test_author'])
print(len(mat['test_author']))
test_author_list = list(mat['test_author'])

df = pd.DataFrame (test_author_list, columns = ['test_author'])
print(df.head(5))
print(df['test_author'].unique())
