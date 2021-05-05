function task_settings = obtain_task_settings (experiment, E1, E2, T1, mice_name, base_val_seed)
%{ function to initialize the variables that the BMI needs %}
    
    task_settings.experiment = experiment;
    task_settings.E2 = E2;
    task_settings.E1 = E1;
    task_settings.T1 = T1;
    task_settings.units = length(E1)+length(E2); 
    task_settings.back_2_base = 1/2*T1;
    
    task_settings.mice_settings = define_mice_settings(mice_name);
    
    task_settings.expected_length_experiment = evalin('base','hSI.hFastZ.numVolumes'); % frames that will last the online experiment (less than actual exp)
    task_settings.frame_rate = evalin('base','hSI.hRoiManager.scanFrameRate/hSI.hFastZ.numFramesPerVolume');
    task_settings.params = define_params();
    
    task_settings.base_frames = round(task_settings.params.base_length * task_settings.frame_rate);
    task_settings.motion_relaxation_frames = 0;  %number of frames that we will wait to compute BMI after motion
    task_settings.moving_average_frames = round(task_settings.params.moving_average * task_settings.frame_rate); 
    task_settings.relaxation_frames = round(task_settings.params.relaxation_time * task_settings.frame_rate);
    task_settings.timeout_frames = round(task_settings.params.timeout * task_settings.frame_rate);
    
    if nargin < 6
        task_settings.base_val_seed = NaN(task_settings.base_frames);
    else
        task_settings.base_val_seed = base_val_seed;
    end