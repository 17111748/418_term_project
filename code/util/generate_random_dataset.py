import numpy as np
import csv
import os

# os.chdir("C:\\Users\\Wei Xin\\Documents\\Wei Xin\\Coursework\\Senior Spring 2020\\418\\Project")

n_entries = 10000
values = np.random.rand(n_entries, 30)
classify = np.random.random_integers(0, 1, (n_entries,1)).astype(float)
data = np.concatenate((values, classify), axis=1)

with open('random_data_n_'+str(n_entries)+'.csv', mode='w', newline='') as random_file:
    random_writer = csv.writer(random_file, delimiter=',')
    for row in data:
            random_writer.writerow(row)
