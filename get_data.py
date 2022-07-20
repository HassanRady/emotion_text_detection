import pandas as pd


# Importing the dataset
DATASET_COLUMNS  = ["sentiment", "ids", "date", "flag", "user", "text"]
DATASET_ENCODING = "ISO-8859-1"
DATASET_PATH = 'data/training.1600000.processed.noemoticon.csv'

dataset = pd.read_csv(DATASET_PATH,
                      encoding=DATASET_ENCODING , names=DATASET_COLUMNS)

# Removing the unnecessary columns.
dataset = dataset[['sentiment','text']]

# Replacing the values.
dataset['sentiment'] = dataset['sentiment'].replace(4,1)

dataset = dataset.sample(frac=1, random_state=42)
dataset[:100].to_csv('tfx-data/data.csv', index=False)