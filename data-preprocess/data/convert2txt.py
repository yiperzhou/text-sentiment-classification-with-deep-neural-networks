import pickle
import pandas as pd
with open('barcelonaTripadvisor_sentiment.pkl', 'rb') as f:
	data = pickle.load(f)

DF = pd.DataFrame()
DF["label"] = data[1]
DF["index"] = data[0]

DF.to_csv("barcelonaTripadvisor_sentiment.csv")
