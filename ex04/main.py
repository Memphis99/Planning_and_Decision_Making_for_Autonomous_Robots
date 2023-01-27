import csv
import numpy as np
y = np.array();
with open('train.csv', newline='') as train:
    training_file = csv.reader(train, delimiter =' ', quotechar =',')
    next(training_file)
    for feature in training_file:
        x=feature.astype(np.double)
        y=np.append(y, np.mean(x[2:end]))






