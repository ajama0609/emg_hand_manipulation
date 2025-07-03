%FIR vs IIR filterdesign I found a 100 tap FIR filter to perform the same
%as a 10 order IIR butterworth filter for this application at least in its
%magnitude spectrum phase is distorted but that isn't important for us as
%we are used RMS,MAV,SSC which are not distorted with phase.
data=readtable('emg_data_1.csv') 
time=data{:,1}; 
channel1=data{:,2};

Fs=1000; 
L=63196; 
T=1/Fs; 
t = (0:L-1)*T; 
f = Fs/L*(0:(L/2)); 


%filterDesigner %run this once to design the filter

%y = filter(Num,1,channel1); FIR filter 

y = sosfilt(SOS, channel1); 
y = prod(G) * y;

[b, a] = sos2tf(SOS, G);

P2 = abs(fft(y) / L);        % two-sided spectrum (normalized)
P1 = P2(1:L/2+1);          % single-sided spectrum
P1(2:end-1) = 2*P1(2:end-1); 

%figure;
%plot(f, P1);
%xlabel('Frequency (Hz)');
%ylabel('|Y(f)|');
%title('Single-Sided Amplitude Spectrum of Y'); 

window_length=200; 
step_size=100; 
num_windows=floor((L-window_length)/step_size)+1;
RMS=zeros(num_windows,1); 
MAV=zeros(num_windows,1); 
ZC=zeros(num_windows,1); 
VAR = zeros(num_windows,1); 
SSC=zeros(num_windows,1); 
threshold = 0.01;
for i=1:num_windows  
    start_idx =(i-1)*step_size +1 ;  
    end_idx=start_idx+window_length -1; 
    window=y(start_idx:end_idx); 
    RMS(i) = sqrt(mean((window.^2))); 
    MAV(i) = mean(abs(window)); 
    VAR(i) = mean(window.^2) - (mean(window).^2);
    zc_sign_changes = sum(abs(diff(sign(window))) == 2);
    ZC(i) = zc_sign_changes;  
    for k =2:length(window)-1 
       left_diff = window(k) - window(k-1);
       right_diff = window(k) - window(k+1); 
       if(left_diff * right_diff) > threshold  
            SSC(i)=SSC(i) + 1; 
       end 
    end
end  

%figure; 
%plot(num_windows,y,num_windows,RMS,num_windows,MAV); 
%xlabel('Time(s)'); 
%ylabel('Amplitude(mV)'); 
%title('Filtered EMG vs Time')  

labels=data{:,10}; 
window_labels=zeros(num_windows,1); 
for i=1:num_windows 
    start_idx=(i-1)*step_size + 1;  
    end_idx=window+start_idx; 
    segment_labels=labels(start_idx:end_idx); 
    window_labels(i)=mode(segment_labels);
end

feature_matrix=[RMS,MAV,VAR,ZC,window_labels]; 

fileID = fopen('features.csv','w','n','windows-1258');  
for i = 1:size(feature_matrix,1)
    fprintf(fileID,'%.11f, %.11f, %.11f, %.11f, %d\n',feature_matrix(i,:)); 
end
fclose(fileID);