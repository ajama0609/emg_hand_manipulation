# emg_hand_manipulation
emg HIL with arduino matlab manipulation of allegroHand 
# Objectives 
1.Use the allegro hand and control it with a arduino based on real time gestures done by a user hooked up to a emg sensors. 
2. Set up the simulink env and model the RL agent 
3. Extract the data from the arduino and post process it in matlab
4. feed emg data into a neural network that can use linear regression to model joint torques and joint velocities 
5. Train the whole pipeline to extract data from EMG -->Arduino --> Gestures clasification --> RL agent --> Real time simulation in simscape