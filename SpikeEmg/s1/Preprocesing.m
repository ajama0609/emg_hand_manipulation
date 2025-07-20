Fs = 1000;            % Sampling frequency                    
T = 1/Fs;             % Sampling period       
L = 101014;             % Length of signal
t = (0:L-1)*T;  
f = Fs/L*(0:(L/2));   
window_ms = 200;                  % window size in milliseconds
window_samples = round(window_ms * Fs / 1000);  % window in samples 
num_windows = floor(L / window_samples);
figure;  % Create a single figure window
hold on;  % Keep all plots on the same axes 

%filterDesigner ; %IIR Highpass 15Hz Fs=1000Hz


emg_filtered = sosfilt(SOS, emg);
for ch = 1:size(emg,2) 
    for n=1:size(G,2)
        emg_filtered(:,ch) = emg_filtered(:,ch) * G(n); 
    end
end

for i=1:size(emg,2)
    Y=fft(emg_filtered(:,i)); 
    
    P2 = abs(Y/L);
    P1 = P2(1:L/2+1);
    P1(2:end-1) = 2*P1(2:end-1); 
    
    plot(f,P1,"LineWidth",3) 

end 
hold off;  % Stop adding to the same plot

num_channels = size(emg_filtered,2);
RMS = zeros(num_windows, num_channels);
VAR = zeros(num_windows, num_channels);
MAV = zeros(num_windows, num_channels);
fft_feats = zeros(num_windows, num_channels, 2); 
labels=zeros(num_windows,1);
for w=1:num_windows 
  start_idx = (w-1)*window_samples + 1;
  end_idx = w*window_samples;
  labels(w)=mode(stimulus(start_idx:end_idx)); 
end  

for ch = 1:num_channels
    for w = 1:num_windows
        start_idx = (w-1)*window_samples + 1;
        end_idx = w*window_samples;
        window_data = emg_filtered(start_idx:end_idx, ch); 

        Y = fft(window_data);
        P2 = abs(Y/window_samples);
        P1 = P2(1:floor(window_samples/2)+1);
        P1(2:end-1) = 2*P1(2:end-1); % single-sided spectrum
        
        % Then extract features like total power, peak freq, etc.
        fft_power = sum(P1.^2);  % total power in window
        fft_max = max(P1);       % max magnitude in freq spectrum

        fft_feats(w, ch, 1) = fft_power;
        fft_feats(w, ch, 2) = fft_max;

        RMS(w, ch) = rms(window_data);
        VAR(w, ch) = var(window_data);
        MAV(w, ch) = mean(abs(window_data));
    end
end
%raw_flat = emg_filtered(:)'; 

fft_feats_reshaped = reshape(fft_feats, num_windows, num_channels * 2);

features = [RMS, VAR, MAV,fft_feats_reshaped,labels]; 
writematrix(features,'s1_feat.csv');
xlabel('Frequency (Hz)');
ylabel('|P1(f)|');
title('FFT of All 10 EMG Channels');
legend("Channel 1","Channel 2","Channel 3","Channel 4","Channel 5", ...
       "Channel 6","Channel 7","Channel 8","Channel 9","Channel 10");
grid off;  