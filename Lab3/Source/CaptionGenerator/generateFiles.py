import pandas as pd

x=pd.read_csv('imageCaption.csv')
z=x.id.apply(lambda x:str(x)+'.jpg')
x.to_csv('tst2.txt', header=None, index=None, sep=' ')
z[0:600].to_csv('train.txt', header=None, index=None)
z[600:].to_csv('tst.txt', header=None, index=None)
