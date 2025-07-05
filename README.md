#EMG Hand gestures classification 
The results achieved with this methodoloy is 89.5% which replicates some research in the field ~90% classification for hand gestures classification.The results can further be improved if 
the rest state can be identified with high accuracies.Currently even in research this seems to be an issue as illustrated in the image below.![image](https://github.com/user-attachments/assets/6f788eb1-edc4-4ff4-a570-d6818716243c) 


#Results 
I have been able to produce currently 89.5% accuracy between 6 gestures including resting,fist clenched,wrist flexion,wrist extension,ulnar deviation and radial deviation.Ultimately the hardest aspect is the fact that at rest the muscles are not sending much of a signal to move the hand and therefore the signal quality is poor (low SNR).This is to be improved on.Ulitmately most research I have look at have achieved ~90% with multiclass gesture classification w/o using a deep neural networks.Below is attached the best performing training's confusion matrix. 
![image](https://github.com/user-attachments/assets/c915a66e-754b-4ec8-949b-e35fbeea93bc)

 


#How to use this resource  
1. First use the matlab script to filter out the data and generate a .csv file. 
2. Use the python script with the EMG_classifier and train a model to classify the data based on the dataset.
3. Evaluate the model by inspecting the accuracies and the confusion matrix.



