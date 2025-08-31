import pandas as pd
import sqlite3
import matplotlib.pyplot as plt

db_file = 'pset0-population.db'

with sqlite3.connect(db_file) as db:
    df = pd.read_sql_query(f"SELECT * FROM population", db)
    print(df.columns)
    print(df.size)

    def analyze(column, bins):
        print(df[column].describe())
        plt.figure()
        hist = df[column].hist(bins=bins)
        hist.set_title(f'{column} Distribution')
        hist.set_xlabel(column)
        hist.set_ylabel('Number')
        plt.show()

    # analyze('age', 100)
    # analyze('weight', 100)

    # Explore Relationships
    def scatter(col1, col2):
        plt.figure()
        plt.scatter(df[col1], df[col2], alpha=0.5)
        plt.title(f'{col1} vs {col2}')
        plt.xlabel(col1)
        plt.ylabel(col2)
        plt.show()

    # scatter('age', 'weight')

    # Identify the outlier
    print(df[(df['weight']<30) & (df['age']>40)])
