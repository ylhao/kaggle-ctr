#!/bin/bash


wc -l train.csv
wc -l test.csv


head -n 1000000 train.csv > train_sub100w.csv
head -n 1000000 test.csv > test_sub100w.csv


wc -l train_sub100w.csv
wc -l test_sub100w.csv
