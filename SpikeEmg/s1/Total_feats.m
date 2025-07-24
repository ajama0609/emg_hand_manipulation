emg3=readtable("s3_feat.csv");
emg2=readtable("s2_feat.csv");
emg=readtable("s1_feat.csv"); 

total_feats = [emg{2:end, :},emg2{2:end, :},emg3{2:end, :}]; 

writematrix(total_feats,'total_feats.csv')