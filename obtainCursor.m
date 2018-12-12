function cursor = obtainCursor(baselineVector, E1, E2, T1, varargin)



if nargin < 4
    T1 = -100;
end
if nargin < 3
    T1 = -100;
    E1 = [1:2];
    E2 = [3:4];
end

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%these are parameters that we will play with and eventually not change
baseLength = 2*60;  % seconds for baseline period

frameRate = evalin('base','hSI.hRoiManager.scanFrameRate/hSI.hFastZ.numFramesPerVolume');

movingAverage= ceil(1*frameRate); % we may want to make this guys global
ens_thresh=0.95;


baseFrames = round(baseLength * frameRate); %hSI.hRoiManager.scanFrameRate); %

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%               
    
cursor = zeros(length(baselineVector), 1);
baseval = 0;
% First pass at T1=1000 so every bit counts
for ind = 1: size(baselineVector, 2)
    
    if ind > movingAverage
        signal = nanmean(baselineVector(:, ind-movingAverage:ind),2);
    else
        signal = nanmean(baselineVector(:, 1:ind),2);
    end
    if ind > baseFrames
        baseval = (baseval*(baseFrames - 1) + signal)/baseFrames;
    else
        baseval = (baseval*(ind - 1) + signal)/ind;
    end
    
    dff = (signal - baseval) ./ baseval;
    dff(dff<T1*ens_thresh) = T1*ens_thresh; % limit the contribution of each neuron to a portion of the target
    % it is very unprobable (almost imposible) that one cell of E2 does
    % it on its own, buuut just in case:
    dff(dff>-T1*ens_thresh) = -T1*ens_thresh;

    cursor(ind) = nansum([nansum(dff(E1)),-nansum(dff(E2))]);
end

cursor(cursor==0) = [];
% we want 1/3 of the times in 30 sec trials so once in 90sec ~ 1 in 100sec 
% with the framerate being ~10 
