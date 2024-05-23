import pandas as pd
import numpy as np 
import io
import os
import gc

df_testset_names = pd.read_csv('labels_very_large_data/test_labels.csv')
df_conf_pred = pd.read_csv('pred.csv', names = ['prob_TD','prob_ASD'])
df_pred = pd.read_csv('class_preds.csv', names = ['pred'])
df_truth = pd.read_csv('truth.csv', names = ['truth'])

df_testset_names['id'] = df_testset_names.id.str.replace('.wav' , '')
df_testset_names= df_testset_names['id'].str.split('_part_',expand=True)

frames = [df_testset_names, df_conf_pred, df_pred, df_truth]
df_merged = pd.concat(frames, axis = 1)
df_merged = df_merged.rename(columns={0: 'video', 1: 'part'})

print(df_merged.head())
df_merged.to_csv('predictions.csv',index = False)
#os.remove('pred.csv')
#os.remove('class_preds.csv')
#os.remove('truth.csv')
