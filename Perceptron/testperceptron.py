import pandas as pd
import numpy as np
from collections import Counter
from nltk import ngrams

# df = pd.read_csv('reviews_tr.csv', sep=',')
# training = np.array(df.values)
#
# x = training[:, 1]
#
# words = set([])
# for i in range(len(x)):
#     if (i % 100000 == 0):
#         print(i)
#     for each in x[i].split(" "):
#         words.add(each)
# print(len(words))

n=11
w_avg = {'hello': 4, 'test': 2}
changes = {'hello': [5, 2, -3, 8, 10], 'test': [6, -2]}
changes_i = {'hello': [2, 4, 5, 8, 9], 'test': [1, 4]}

for gram, changei in changes_i.items():
    # w_avg[gram] = w_avg[gram] / (n + 1)
    x = w_avg[gram]
    w_avg[gram] = w_avg[gram] * changei[0]
    print('{}: {}'.format(gram, w_avg[gram]))
    for i in range(len(changei)-1):
        print('{}: {}'.format(gram, w_avg[gram]))
        w_avg[gram] += (changei[i + 1] - changei[i]) * (changes[gram][i] + x)
        x += changes[gram][i]
        if i + 1 == len(changei)-1:
            print('{}: {}'.format(gram, w_avg[gram]))
            w_avg[gram] += (n - changei[i+1]) * (changes[gram][i+1] + x)
print(w_avg)