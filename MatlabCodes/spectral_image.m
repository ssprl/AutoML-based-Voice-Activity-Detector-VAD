clear
close all

addpath('C:\Users\gxs160730\Documents\Ph.D\CNN-SE\CNN-VAD\Training Code\');
addpath('C:\Users\gxs160730\Documents\Ph.D\Siren Noise\Features');

Speech_path             = 'C:\Users\gxs160730\Documents\MATLAB\pure_clean_timit\clean_timit\all100';
Noise_path              = 'C:\Users\gxs160730\Documents\MATLAB\TOP 10 NOISES FROM DATABASE\Traffic';

speechFiles             = dir(Speech_path);
sirenFiles              = dir(Noise_path);

data.speechFiles        = speechFiles(~ismember({speechFiles.name},{'.' '..'}));
data.sirenFiles         = sirenFiles(~ismember({sirenFiles.name},{'.' '..'}));

data.nSpeechFiles       = numel(data.speechFiles);
data.nSirenFiles        = numel(data.sirenFiles);


Folder   = strcat('C:\Users\gxs160730\Documents\MATLAB\pure_clean_timit\clean_timit\all100');
FileList = dir(fullfile(Folder, '*.wav'));
Cell = struct2cell(FileList);
Cell = Cell(1,:);
Cell = natsort(Cell);
N        = data.nSpeechFiles;

M        = floor(data.nSpeechFiles/data.nSirenFiles);

dirname  = 'C:\Users\gxs160730\Documents\Ph.D\AutoML\images1\25x257_224_noisy_images';
cnt      = 1;
trainData = cell(N,1);

params.overlap           = 8;
params.nfft              = 448;
params.window            = hanning(2 * params.overlap);

clean_sp_directory = 'C:\Users\gxs160730\Documents\Ph.D\AutoML\images1\224x224x3\clean';
noisy_sp_directory = 'C:\Users\gxs160730\Documents\Ph.D\AutoML\images1\224x224x3\noisy';
%% For every noise
for n    = 1:data.nSirenFiles
    
    sirenFile           = [data.sirenFiles(n).folder '\' data.sirenFiles(n).name];
    [sirenSig_2,fs_si]  = audioread(sirenFile);
    [p,q]               = rat(16000/fs_si,0.001);
    sirenSig_1          = resample(sirenSig_2(:,1),p,q);
    sirenSig1           = [sirenSig_1;sirenSig_1;sirenSig_1;sirenSig_1;sirenSig_1;sirenSig_1;sirenSig_1;sirenSig_1;sirenSig_1;sirenSig_1];
    %% for each speech
    for m = 1:N
  
        current_name = fullfile(Folder, Cell(m));
        current_name = string(current_name);
        [filepath,filename,fileext] = fileparts(current_name);
        [speech,fs] = audioread(current_name);
        sirenSig = sirenSig1(1:length(speech));
        snr = 5;
        speech_noise  = addnoise(speech,sirenSig,snr,16000);
        len=floor(2*fs/1000); % Frame size in samples
        if rem(len,2)==1, len=len+1; end;
        PERC=50; % window overlap in percent of frame size
        len1=floor(len*PERC/100);
        len2=len-len1; % update rate in samples
        win = hanning(len);  % define window
        win = win*len2/sum(win);  % normalize window for equal level output
        Nframes = floor(length(speech)/len2)-floor(len/len2);
        nFFT = 448;
        k=1;
        count = 0;
        num = 0;
        %% each frame
        for n1 = 1:Nframes
            %% noisy speech
            insign = win.*speech_noise(k:k+len-1);
            count  = count + 1;
            spec   = fft(insign,nFFT);
            mag_procFFT_noi     = abs(spec).^2;
            STFTMdb1_s        = 10*log10(mag_procFFT_noi);
            %% noise
            insign1 = win.*sirenSig(k:k+len-1);
            spec1   = fft(insign1,nFFT);
            mag_procFFT_noi1     = abs(spec1).^2;
            STFTMdb1_noi1        = 10*log10(mag_procFFT_noi1);
            STFTMdb_noi(count,:) = STFTMdb1_noi1(1:nFFT/2 ,:);
            STFTMdb_s(count,:)= STFTMdb1_s(1:nFFT/2 ,:);
            
            %% for every 224 frames
            if (count == 224)
                num = num + 1;
              STFTMdb  = STFTMdb_s;
              A1 = flipud(STFTMdb');
              A11 = uint8(round((A1 - min(min(A1))) * 255 ./ (max(max(A1))-min(min(A1)))));
              A4 = parula(256);
              A51 = ind2rgb(A11,A4);
              A5 = uint8(round((A51 - min(min(A51))) * 255 ./ (max(max(A51))-min(min(A51)))));
%                imshow(A5);
                STFTMdb1 = STFTMdb_noi;
                A1 = flipud(STFTMdb1');
                A11 = uint8(round((A1 - min(min(A1))) * 255 ./ (max(max(A1))-min(min(A1)))));
                A4 = parula(256);
                A51 = ind2rgb(A11,A4);
                A6 = uint8(round((A51 - min(min(A51))) * 255 ./ (max(max(A51))-min(min(A51)))));
              subplot(1,2,1),imshow(A5);
              subplot(1,2,2),imshow(A6);
                %% save
                name_noisy = strcat(noisy_sp_directory,'\',num2str(n),'_',num2str(m),'_',num2str(num),'_NOISU2','.jpg');
                name_clean = strcat(clean_sp_directory,'\',num2str(m),'_',num2str(n),'_',num2str(num),'_CLEANU','.jpg');
                imwrite(A5, name_clean);
                if (m==1)
                    imwrite(A6, name_noisy);
                end
                count = 0;
            end
            
            k=k+len2;
            
        end
    end
end

