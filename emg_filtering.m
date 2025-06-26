file = readtable("emg_data_1.csv");
time = file{:,1};           % First column = Time (ms)
channels = file{:, 2:9};    % Columns 2-9 = Channel 1 to 8
gesture_class = file{:, 10}; % Class labels (0 to 7)

% Parameters
window_size = 200;    % in samples
step_size = 100;      % in samples
threshold = 2e-4;
num_samples = size(channels, 1);

% Initialize cleaned windows and labels storage
cleaned_windows = {};
cleaned_labels = [];
window_times = {};

% Extract windows that contain any nonzero values
for start_idx = 1:step_size:(num_samples - window_size + 1)
    end_idx = start_idx + window_size - 1;
    window = channels(start_idx:end_idx, :);
    
    if any(window(:) ~= 0)
        cleaned_windows{end+1} = window;
        window_times{end+1} = time(start_idx:end_idx);
        cleaned_labels(end+1) = gesture_class(start_idx + floor(window_size/2));
    end
end

% Butterworth bandpass filter design
Fs = 1000;  % Sampling frequency (Hz)
order = 10;
low_cutoff = 20;
high_cutoff = 450;
Wn = [low_cutoff high_cutoff] / (Fs/2);
[b, a] = butter(order, Wn, 'bandpass');

num_windows = length(cleaned_windows);
num_channels = size(cleaned_windows{1}, 2);

% Preallocate filtered windows and RMS feature matrix
filtered_windows = cell(num_windows, 1);
rms_features = zeros(num_windows, num_channels);

% Filter each window and compute RMS per channel
for w = 1:num_windows
    window = cleaned_windows{w};
    filtered_window = zeros(size(window));
    
    for ch = 1:num_channels
        y = window(:, ch);
        y_filtered = filtfilt(b, a, y);
        filtered_window(:, ch) = y_filtered;
    end
    
    filtered_windows{w} = filtered_window;
    
    % Compute RMS for filtered window per channel
    rms_features(w, :) = sqrt(mean(filtered_window.^2, 1));
end

disp('RMS feature matrix size:');
disp(size(rms_features));  % Should be [num_windows x 8]

% Plot RMS features across windows
colors = lines(num_channels);
figure;
hold on;
%for ch = 1:num_channels
    plot(1:num_windows, rms_features(:, 1), 'Color', colors(1, :), 'LineWidth', 1.5);
%end
xlabel('Window Number');
ylabel('RMS Amplitude');
title('RMS Features Across Windows for Each Channel');
legend(arrayfun(@(x) sprintf('Channel %d', x), 1:num_channels, 'UniformOutput', false));
grid on;
hold off;
