file = readtable("emg_data_1.csv") 
time = file{:,1};           % First column = Time (ms) 
channels = file{:, 2:9};     % Columns 2-9 = Channel 1 to 8  
gesture_class = file{:, 10}; % Class labels (0 to 7) 

% Parameters
window_size = 200;    % in samples
step_size = 100;      % in samples
threshold = 2e-4;
num_samples = size(channels, 1);

cleaned_windows = {};  
cleaned_labels = [];
window_times = {};

for start_idx = 1:step_size:(num_samples - window_size + 1)
    end_idx = start_idx + window_size - 1;
    window = channels(start_idx:end_idx, :);
    
    if any(window(:) ~= 0)
        cleaned_windows{end+1} = window;
        window_times{end+1} = time(start_idx:end_idx);
        cleaned_labels(end+1) = gesture_class(start_idx + floor(window_size/2));
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


    % Butterworth bandpass filter design
    order = 10;
    low_cutoff = 20;   % Hz
    high_cutoff = 450; % Hz
    Wn = [low_cutoff high_cutoff] / (Fs/2);  % Normalize frequencies
    
    [b, a] = butter(order, Wn, 'bandpass');

   % Filter and FFT per channel
    figure;
    for ch = 1:size(window, 2)
        y = window(:, ch);
        
        % Apply Butterworth bandpass filter
        y_filtered = filtfilt(b, a, y);
        
        % FFT of filtered signal
        Y = fft(y_filtered);
        P2 = abs(Y/L);
        P1 = P2(1:floor(L/2)+1);
        P1(2:end-1) = 2*P1(2:end-1);
        
        f = Fs*(0:floor(L/2))/L;
        
        subplot(4,2,ch);
        plot(f, P1);
        title(['Filtered FFT Channel ', num2str(ch)]);
        xlabel('Frequency (Hz)');
        ylabel('|P1(f)|');
        grid on;
    end