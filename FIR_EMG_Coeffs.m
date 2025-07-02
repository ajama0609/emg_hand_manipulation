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


%filterDesigner run this once to design the filter

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

N=100;
RMS=zeros(L,1); 
MAV=zeros(L,1); 
ZC=zeros(L,1); 
VAR = zeros(L,1); 
SSC=zeros(L,1); 
threshold = 0.01;
for i=N:L   
    window=y(i-N+1:i);
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

figure; 
plot(time,y,time,RMS,time,MAV); 
xlabel('Time(s)'); 
ylabel('Amplitude(mV)'); 
title('Filtered EMG vs Time') 

feature_matrix=[RMS,MAV,VAR,ZC,SSC]; 

fileID = fopen('features.csv','w','n','windows-1258');  
for i = 1:size(feature_matrix,1)
    fprintf(fileID,'%.11f, %.11f, %.11f, %.2f, %.2f\n',feature_matrix(i,:)); 
end
fclose(fileID);