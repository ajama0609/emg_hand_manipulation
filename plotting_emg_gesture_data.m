file = readtable("emg_data_1.csv") 
time = file{:, 1};           % First column = Time (ms) 
channels = file{:, 2:9};     % Columns 2-9 = Channel 1 to 8  
gesture_class = file{:, 10}; % Class labels (0 to 7)
figure;
plot(time, channels);
xlabel('Time (ms)');
ylabel('EMG Signal (Amplitude)');
title('EMG Signals from 8 Channels');
legend('Channel 1', 'Channel 2', 'Channel 3', 'Channel 4', ...
       'Channel 5', 'Channel 6', 'Channel 7', 'Channel 8');
grid on;
