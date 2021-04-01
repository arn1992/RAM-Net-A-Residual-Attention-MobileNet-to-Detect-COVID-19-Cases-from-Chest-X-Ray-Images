import pandas as pd

df = pd.read_csv('D:/polynomial/covid19/new_version/train_covid19.txt', sep=" ", names=['file_name','image_name','diseases','Region Name'])

'''
#df['man']=pd.Categorical(df["diseases"])
#print(df['man'])
print(df['diseases'].value_counts())
#df=df.sample(frac=1).groupby('diseases', sort=False).head(100)
#print(df['diseases'].value_counts())
#print(df)
df['dis_cat']=pd.Categorical(df["diseases"]).codes
#print(df['kill'].unique())
#df.to_csv (r'D:/polynomial/covid19/test.csv', index = False, header=True)
df['image_address']='D:/Covid19/New_version/data/test/'+df['image_name']
df.to_csv (r'D:/polynomial/covid19/new_version/test_covid19.csv', index = False, header=True)
#print(x)'''
'''
df1=pd.DataFrame()
d1=[]
d2=[]
d3=[]
d4=[]
for i in range (182):
    d1.append('COVID'+ str(i+1))
    d2.append('COVID'+str(i+1)+'.png')
    d3.append('COVID-19')
    d4.append('man')
df1['file_name']=d1
df1['image_name']=d2
df1['diseases']=d3
df1['Region Name']=d4
print(df1.tail())

frames=[df,df1]
result_df= pd.concat(frames)

print(result_df)
print(result_df.tail())


#df['man']=pd.Categorical(df["diseases"])
#print(df['man'])
result_df['dis_cat']=pd.Categorical(result_df["diseases"]).codes
#print(df['kill'].unique())
#df.to_csv (r'D:/polynomial/covid19/test.csv', index = False, header=True)
result_df['image_address']='D:/Covid19/data/train/'+result_df['image_name']
result_df.to_csv (r'D:/polynomial/covid19/train.csv', index = False, header=True)
#print(x)
'''
print(df.diseases.value_counts())
print(len(df))

df1=pd.read_csv('D:/polynomial/covid19/new_version/train_covid19.csv')

print(df1.diseases.value_counts())
print(len(df1))