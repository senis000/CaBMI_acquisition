

%check csv file when cells in different frames

% Define the fname
fname = 'bmi_00001.tif';
% Define the fbase
fbase = 'baseline_00001.tif';
% Define the fcsv file 
fcsv = 'bmi_IntegrationRois_00001.csv';
% Define the framerate
fr = evalin('base','hSI.hRoiManager.scanFrameRate/hSI.hFastZ.numFramesPerVolume');
% Define initialZ
initialZ=577.1;
E1 = [3 4];
E2 = [1 2];
T1 = -0.7;
ensemble_neur = [2 21 34 37];
allmask = zeros(256,256,4);
allmask(round(-0.5656474345*112.2807+128), round(-0.4320299303*112.2807+128), 1) = 1;
allmask(round(-0.8239746094*112.2807+128), round(-0.2271497572*112.2807+128), 2) = 1;
allmask(round(-0.2449654244*112.2807+128), round(0.2271497572*112.2807+128), 3) = 1;
allmask(round(0.4231220967*112.2807+128), round(-0.5122004329*112.2807+128), 3) = 1;
figure
imshow(sum(allmask,3))



% Save the whole workspace (besides hSI and hSIcontrol) in wmat.mat inside
% folder of experiment
