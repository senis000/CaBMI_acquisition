%{ this script follows step by step what you need to do to run the BMI experiment.
%}

% parameters to start the protocol
folder_main = 'C:\Users\Nuria\Documents\DATA\';
mice_name = 'IT01';
day = '210903';
experiment = 'normal_bmi';

baseline_file ='baseline_IntegrationRois_00001.csv';

% define paths
folder_path = fullfile(folder_main, 'ITPT'); %debug
% folder_path = fullfile(folder_main, mice_name, day);
folder_plots = fullfile(folder_path, 'plots');

if not(isfolder(folder_plots))
    mkdir(folder_plots)
end

% task settings
task_settings = obtain_task_settings (exptStr, E1, E2, T1, baseValSeed);
% mice specific sessings

% import the baseline info
baseline_file_path = fullfile(folder_path, baseline_file); %debug
baseline_data = table2array(readtable(baseline_file_path));

