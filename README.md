#EMG Hand gestures classification 
The results achieved with this methodoloy is 89.5% which replicates some research in the field ~90% classification for hand gestures classification.The results can further be improved if 
the rest state can be identified with high accuracies.Currently even in research this seems to be an issue as illustrated in the image below.![image](https://github.com/user-attachments/assets/6f788eb1-edc4-4ff4-a570-d6818716243c) 


Ultimately what has been achieved with this open source project is a end-end processing pipeline for one to use to begin to classify hand gestures based on electromyography data.

The dataset was open source and is in the folder 'emg+data+for+gestures' and it contains further documentation outlining how they collected their data.  
#How to use this resource  
1. First use the matlab script to filter out the data and generate a .csv file. 
2. Use the python script with the EMG_classifier and train a model to classify the data based on the dataset.
3. Evaluate the model by inspecting the accuracies and the confusion matrix.



