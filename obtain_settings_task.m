function task_settings = obtain_settings_task (experiment, mice_name, folder_path, debug_mode, base_val_seed)
%{ function to initialize the variables that the BMI needs %}
    
    %% basic settings
    task_settings.experiment = experiment;
    
    %% parameteres
    task_settings.mice_settings = obtain_settings_mice(mice_name);
    task_settings.params = define_params_task();
    
    if nargin < 4
        debug_mode = false;
    end
    
    %% parameters depending on frames
    if debug_mode
        task_settings.expected_length_experiment = 10000;
        task_settings.frame_rate = 30;
    else
        task_settings.expected_length_experiment = evalin('base','hSI.hFastZ.numVolumes'); % frames that will last the online experiment (less than actual exp)
        task_settings.frame_rate = evalin('base','hSI.hRoiManager.scanFrameRate/hSI.hFastZ.numFramesPerVolume');
    end

    task_settings.base_frames = round(task_settings.params.base_length * task_settings.frame_rate);
    task_settings.motion_relaxation_frames = 0;  %number of frames that we will wait to compute BMI after motion
    task_settings.moving_average_frames = round(task_settings.params.moving_average * task_settings.frame_rate); 
    task_settings.relaxation_frames = round(task_settings.params.relaxation_time * task_settings.frame_rate);
    task_settings.timeout_frames = round(task_settings.params.timeout * task_settings.frame_rate);
    
    %% Seed for the baseline if provided
    if nargin < 8
        task_settings.base_val_seed = NaN(task_settings.base_frames);
    else
        task_settings.base_val_seed = base_val_seed;
    end
    
    %% folders
    task_settings.folder_plots = fullfile(folder_path, 'plots');

    if not(isfolder(task_settings.folder_plots))
        mkdir(task_settings.folder_plots)
    end
