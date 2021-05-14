%{ this script follows step by step what you need to do to run the BMI experiment.
%}


%*********************************************************************
%*********************************************************************
% TO CHANGE EVERYDAY!!!!
%*********************************************************************
%*********************************************************************

mice_name = 'IT01';
day = '210903';
experiment = 'normal_bmi';
round = 'D1exp';

%*********************************************************************
%% parameters to start the protocol
%*********************************************************************
folder_main = 'C:\Users\Nuria\Documents\DATA\';
baseline_file ='baseline_IntegrationRois_00001.csv';

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


%*********************************************************************
%% Select the ensemble neurons
%*********************************************************************
% TODO


%*********************************************************************
%% Calibrate the BMI
%*********************************************************************
% import the baseline info
baseline_file_path = fullfile(folder_path, baseline_file); %debug
baseline_data = table2array(readtable(baseline_file_path));

% calibration settings

calibration_settings = obtain_settings_calibration(mice_name, folder_path, baseline_file_path, debug_mode);


% task settings
% you can jump_start the bmi by using base_val_seed from a previous run
% with the same animal/session. If base_val_seed is provided it wont spend
% time collecting the dynamic_baseline
task_settings = obtain_settings_task (experiment, E1, E2, T1, ...
    mice_name, folder_path, debug_mode);

%*********************************************************************
%% Calibrate target
%*********************************************************************



