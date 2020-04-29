import csv
import os


with open('../data/data.csv') as in_file:
    with open('../data/clean_data.csv', mode='w', newline='') as clean_file:
        in_reader = csv.reader(in_file, delimiter=',')
        clean_writer = csv.writer(clean_file, delimiter=',')

        start = True
        for row in in_reader:
            if (start):
                start = False
                clean_writer.writerow(row)
            else:
                if (row[-1] == 'B'):
                    row[-1] = '0'
                else:
                    row[-1] = 1
                clean_writer.writerow(row)
