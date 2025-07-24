emg3=readtable("s3_feat.csv");
emg2=readtable("s2_feat.csv");
emg=readtable("s1_feat.csv"); 

total_feats = [emg;emg2;emg3]; 

writetable(total_feats,'total_feats.csv')