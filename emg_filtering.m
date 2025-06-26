file = readtable("emg_data_1.csv") 
time = file{:,1};           % First column = Time (ms) 
channels = file{:, 2:9};     % Columns 2-9 = Channel 1 to 8  
gesture_class = file{:, 10}; % Class labels (0 to 7) 

% Parameters
window_size = 200;    % in samples
step_size = 100;      % in samples
threshold = 2e-4;

% Make a copy to modify
cleaned_channels = channels; 
cleaned_winows={}; 
window_times={};

for start_idx = 1:step_size:(num_samples - window_size + 1)
    end_idx = start_idx + window_size - 1;
    window = cleaned_channels(start_idx:end_idx, :);
    
    % Zero out if max window amplitude < threshold
    if max(window(:)) < threshold
        cleaned_channels(start_idx:end_idx, :) = 0;
    end 
    
     if any(window(:) ~= 0)
        cleaned_windows{end+1} = window;
        window_times{end+1} = time(start_idx:end_idx); 
     end  
end 


% Overwrite channels with cleaned data for further processing
channels = cleaned_channels; 


% Plot the full signal with zeroed-out quiet windows
figure;
plot(time, cleaned_channels);
xlabel('Time (ms)');
ylabel('Amplitude (|EMG|)');
title('EMG Signals with Quiet Windows Zeroed Out');
legend('Channel 1','Channel 2','Channel 3','Channel 4', ...
       'Channel 5','Channel 6','Channel 7','Channel 8');
grid on; 

 window = cleaned_windows{1};
 t = window_times{1}; 
 Fs = 1000;  % Sampling frequency in Hz (adjust if different)
 L = size(window, 1);  % Window length (number of samples)

    % FFT for each channel
    figure;
    for ch = 1:size(window, 2)
        y = window(:, ch);
        Y = fft(y);
        
        % Two-sided spectrum P2, then single-sided P1
        P2 = abs(Y/L);
        P1 = P2(1:floor(L/2)+1);
        P1(2:end-1) = 2*P1(2:end-1);
        
        f = Fs*(0:floor(L/2))/L;  % Frequency vector
        
        subplot(4,2,ch);
        plot(f, P1);
        title(['FFT Channel ', num2str(ch)]);
        xlabel('Frequency (Hz)');
        ylabel('|P1(f)|');
        grid on;
    end