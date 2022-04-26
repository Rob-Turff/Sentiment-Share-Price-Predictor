import json

import pandas as pd

# df = pd.read_csv(r"../data/stock_lex.csv")
# lexiconString = ""
# for index, row in df.iterrows():
#     line = str(row[0]) + "\t" + str((row[2] + row[3])/2) + "\n"
#     lexiconString += line

file = open("../../data/Lexicons/NTUSD_Fin_word_v1.0.json", "r")
data = json.load(file)
file.close()
lexiconString = ""
for dict in data:
    line = dict["token"] + "\t" + str(dict["market_sentiment"]) + "\n"
    lexiconString += line

text_file = open("../../data/Lexicons/NTUSD_lex.txt", "w")
n = text_file.write(lexiconString)
text_file.close()