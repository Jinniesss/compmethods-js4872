import pandas as pd
import tqdm

file_name = 'project/data/NSDUH_2023_Tab.txt'
df = pd.read_csv(file_name, sep="\t", dtype=str)
# print(df.head())
print(df['AGE3'].value_counts())