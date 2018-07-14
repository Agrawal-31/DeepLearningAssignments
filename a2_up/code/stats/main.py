import pandas as pd

df = pd.read_csv('6')
for i in range(1, len(df.index)):
  if df.iloc[i,2] < df.iloc[i - 1, 2]:
    print(i, df.iloc[i, 1], df.iloc[i, 2], df.iloc[i - 1, 2] - df.iloc[i, 2])
