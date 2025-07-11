%FIR vs IIR filterdesign I found a 100 tap FIR filter to perform the same
%as a 10 order IIR butterworth filter for this application at least in its
%magnitude spectrum phase is distorted but that isn't important for us as
%we are used RMS,MAV,SSC which are not distorted with phase.
data=readtable('emg_data_1.csv') 
time=data{:,1}; 
channels=data{:,2:9}; 
channel_length = size(channels,2);
window_length=200; 
step_size=100; 
labels=data{:,10}; 


Fs=1000; 
L=63196; 
T=1/Fs; 
t = (0:L-1)*T; 
f = Fs/L*(0:(L/2)); 

num_windows=floor((L-window_length)/step_size)+1;   
%filterDesigner %run this once to design the filter

env=zeros(num_windows,channel_length);
for ch = 1: size(channels,2) 
    for i=1:num_windows  
        start_idx =(i-1)*step_size +1 ;  
        end_idx=start_idx+window_length -1; 
        window=channels(start_idx:end_idx,ch);  
        env(i,ch)=mean(abs(hilbert(window)));  
    end 
end
y = zeros(size(channels)); 
threshold=9e-4;
for ch = 1:channel_length
    if any(env(:, ch) > threshold)
        % Filter channel only if any window has envelope > thgireshold
        y(:, ch) = sosfilt(SOS, channels(:, ch));
        y(:, ch) = y(:, ch) * prod(G);
    else
        % Otherwise, keep raw data
        y(:, ch) = channels(:,ch);
    end
end 

[b, a] = sos2tf(SOS, G);

RMS=zeros(num_windows,channel_length); 
MAV=zeros(num_windows,channel_length); 
ZC=zeros(num_windows,channel_length); 
VAR = zeros(num_windows,channel_length); 
SSC=zeros(num_windows,channel_length);  
threshold = 0.01;
for ch = 1: size(channels,2) 
    for i=1:num_windows  
        start_idx =(i-1)*step_size +1 ;  
        end_idx=start_idx+window_length -1; 
        window=y(start_idx:end_idx,ch); 
     %   env(i,ch)=mean(abs(hilbert(window)));
        RMS(i,ch) = sqrt(mean((window.^2))); 
        MAV(i,ch) = mean(abs(window)); 
        VAR(i,ch) = var(window);
        zc_sign_changes = sum(abs(diff(sign(window))) == 2);
        ZC(i,ch) = zc_sign_changes;  
        for k =2:length(window)-1 
           left_diff = window(k) - window(k-1);
           right_diff = window(k) - window(k+1); 
           if(left_diff * right_diff) > threshold  
                SSC(i,ch)=SSC(i,ch) + 1; 
           end 
        end
    end  
end
   
segment_labels = cell(num_windows, 1);  
window_labels=zeros(num_windows,1);   
  for i=1:num_windows  
        start_idx=(i-1)*step_size + 1;  
        end_idx=window_length+start_idx -1 ; 
        segment_labels{i}=labels(start_idx:end_idx); 
        window_labels(i)=mode(segment_labels{i});
  end

features = cell(channel_length, 1);
for ch = 1:channel_length
    features{ch} = [RMS(:, ch), MAV(:, ch), VAR(:, ch), ZC(:, ch)];  % 4 features per window
end 

feature_matrix=[];  
for i =1:num_windows 
    row_features=[];
    for ch =1:size(channels,2)
        row_features=[row_features,features{ch}(i,1:4)]; 
    end  
    feature_matrix(i,:) = [row_features, window_labels(i)];
end 

writematrix(feature_matrix,'features1.csv');