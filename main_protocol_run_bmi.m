%{ 
this script follows step by step the code you need to do to run the BMI experiment.
To make things cleared, this doc only contains the code that you need to
execute. Please refer to the document "HOWTO_2P_SCANIMAGE" for hardware and
software adjustements during BMI.
%}


%*********************************************************************
%*********************************************************************
% TO CHANGE EVERYDAY!!!!
%*********************************************************************
%*********************************************************************

mice_name = 'IT01';
day = '210906';
experiment = 'normal_bmi';
%{
to choose from:
normal_bmi -> normal bmi without any fancy stuff
dstim_bmi -> BMI + using stim to give reward
block_bmi_reward -> BMI + blocking D1 when given a reward
block_bmi -> BMI + blocking D1 when closer to target
random_stim -> BMI + stim random time
%}
round = 'test';
debug_mode = false;

%*********************************************************************
%% parameters to start the protocol
%*********************************************************************
folder_main = 'F:\Dopamine_BMI\';

% define paths
folder_path = fullfile(folder_main, round, mice_name, day); %debug
if not(isfolder(folder_path))
    mkdir(folder_path)
end
% mice specific sessings

%*********************************************************************
%% Run the baseline for calibration
%*********************************************************************
% TODO!

baseline_file ='baseline_IntegrationRois_00001.csv';
%%*********************************************************************
%% Select the ensemble neurons
%*********************************************************************
% import the baseline info
baseline_file_path = fullfile(folder_path, baseline_file); %debug
opts = detectImportOptions(baseline_file_path);
opts.VariableTypes(:) = {'double'};
baseline_online = table2array(readtable(baseline_file_path, opts));
baseline_data = baseline_online(10:end,3:end)';

% input E1 and E2 values
[E1, E2] = select_ensemble_neurons(baseline_data);

%*********************************************************************
%% Calibrate the BMI
%*********************************************************************

% calibration settings
calibration_settings = obtain_settings_calibration(E1, E2, ...
    folder_path, baseline_file_path, debug_mode);


% task settings
% you can jump_start the bmi by using base_val_seed from a previous run
% with the same animal/session. If base_val_seed is provided it wont spend
% time collecting the dynamic_baseline
task_settings = obtain_settings_task (experiment, mice_name, folder_path,...
    debug_mode);



calibration_results = obtain_calibration(baseline_data([calibration_settings.E_sorted], :),  ...
    calibration_settings , task_settings);

%*********************************************************************
%% Remove all Indirect (non ensemble) neurons
%*********************************************************************
% Please refer to "HOWTO_2P_SCANIMAGE"

%% define globals
global history
global data
history.baseval = [];
history.buffer = [];  %define a windows buffer
history.index = 1 ;
history.last_volume = 0;  %careful with this it may create problems
history.nZ=1;
history.number_hits = 0; 
history.number_miss = 0; 
history.number_rewards = 0; 
history.number_trials = 0;
history.number_stims = 0;
data.cursor = [];  %define a very long vector for cursor
data.frequency = [];
data.hits = [];
data.miss = [];
data.rewards = [];
data.stims = [];
data.trial_end = [];
data.trial_start = [];
data.time_vector = [];

%*********************************************************************
%% Run BMI
%*********************************************************************

%******************************************
%% Finishing up
%*******************************************
roiGroup = hSI.hIntegrationRoiManager.roiGroup;
clear hSI
clear hSICtl
filename_path = fullfile(folder_path, 'workspace.mat'); %debug;
save(filename_path)