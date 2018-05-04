import pickle
import pandas as pd
with open('/home/yi/sentimentAnalysis/sample-algorithm-code/LSTM-Sentiment-Analysis/RNN_Text_Classify/data/csv-zusammenfuehren.de_r922bdrm_XY.pkl', 'rb') as f:
	data = pickle.load(f)

DF = pd.DataFrame()
DF["label"] = data[1]
DF["index"] = data[0]

DF.to_csv("subj0.csv")


