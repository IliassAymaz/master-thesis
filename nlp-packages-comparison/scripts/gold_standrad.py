import pandas
import pprint
import pickle
import os
pp = pprint.PrettyPrinter(indent=4)

csv_path = '../data/gold_25_11.csv'
csv_path = os.path.join(os.path.dirname(__file__), csv_path)


# read obtained gold standard and compare
gold = pandas.read_csv(csv_path, sep=';', encoding='cp1252')
#print(gold)

# basic preprocessing
# remove \r and 'Unnamed'
gold = pandas.DataFrame(data=gold, columns=['sentence', 'lemmatized', 'pos tags', 'noun chunks']).\
    to_dict(orient='records')

for element in gold:
    for key in element.keys():
        element[key] = element[key].replace('\r', '')
pp.pprint(gold)

# store it for later comparison
pickle_path = os.path.join(os.path.dirname(__file__), '../data/gold_standard_nlp.pickle')
with open(pickle_path, 'wb') as f:
    pickle.dump(gold, f, protocol=2)
