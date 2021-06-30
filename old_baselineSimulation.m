function [percentCorrect, T1] = baselineSimulation(baselineVector, E1, E2, varargin)

% function to obtain the value of T1 when there is no end of trial

if nargin < 3
    E1 = [3 4];
    E2 = [1 2];
end

% initialize variables
T1 = -10;
maxiter = 20;
percentCorrect = 0;
limitPercentCorrect = [40, 50];  %% 33+- 5
trialDuration = 30;
frameRate = evalin('base','hSI.hRoiManager.scanFrameRate/hSI.hFastZ.numFramesPerVolume');
relaxationTime = 5;  %sec
perT1 = 0.3; %initial percentil to obtain T1
relaxationFrames = round(relaxationTime * frameRate); %hSI.hRoiManager.scanFrameRate); %

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%               

numberTrials = round(length(baselineVector)/frameRate/trialDuration);
% first inital cursor with T1 by default to initialize the cursor
cursor = obtainCursor(baselineVector, E1, E2, T1, frameRate);
iter = 0;
tol = 0.1;

while (percentCorrect > limitPercentCorrect(2)) || (percentCorrect < limitPercentCorrect(1))
    iter = iter +1;
    if iter>= 10
        tol = 0.5;
        if iter>=maxiter
            disp('too much iter')
            disp(percentCorrect)
            return
        end
    end
    T1 = prctile(cursor, perT1); % in (%)
    cursor = obtainCursor(baselineVector, E1, E2, T1, frameRate);
    actBuffer = zeros(relaxationFrames, 1);
    
    success = 0;

    for ind = 1: length(cursor)
        activationAUX = 0;
        if cursor(ind) <= T1 && sum(actBuffer)== 0
            activationAUX = 1; 
            success = success +1;

        end
        actBuffer(1:end-1) = actBuffer(2:end);
        actBuffer(end) = activationAUX;

    end
    percentCorrect = success/numberTrials*100;
    if percentCorrect > limitPercentCorrect(2); perT1 = perT1-tol; end
    if percentCorrect < limitPercentCorrect(1); perT1 = perT1+tol; end
    disp([num2str(percentCorrect), ' ', num2str(T1)])
    if perT1 >= 100 || perT1 <= 0
        disp("Warning imposible to initiate T1"); 
        return;
    end
end
% figure()
% plot(cursor/T1)
% xlabel('cursor')
% figure()
% freqmed = (18000 - 2000)/2;
% freqplot = double(round(freqmed + 2000 + cursor/T1*freqmed));
% plot(freqplot)
% xlabel('freq')
    
