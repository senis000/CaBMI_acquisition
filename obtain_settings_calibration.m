function calibration_settings = obtain_settings_calibration (mice_name, E1, E2, folder_path, baseline_file_path, debug_mode)
%{ function to initialize the variables that the BMI needs %}
    
    %% input settings
    calibration_settings.baseline_file_path = baseline_file_path;
    calibration_settings.E2 = E2;
    calibration_settings.E1 = E1;
    calibration_settings.units = length(E1)+length(E2); 

    %% parameteres
    calibration_settings.mice_settings = define_mice_settings(mice_name);
    calibration_settings.params = define_params_calibration();
    
    if nargin < 6
        debug_mode = false;
    end
    
    %% parameters depending on frames
    if debug_mode
        calibration_settings.frame_rate = 30;
    else
        calibration_settings.frame_rate = evalin('base','hSI.hRoiManager.scanFrameRate/hSI.hFastZ.numFramesPerVolume');
    end
    calibration_settings.frames_per_reward_range = ...
        calibration_settings.params.sec_per_reward_range*calibration_settings.frame_rate;
    calibration_settings.reward_per_frame_range = 1./calibration_settings.frames_per_reward_range;
    
    %% folders
    calibration_settings.folder_plot = fullfile(folder_path, 'plots', 'calibration');

    if not(isfolder(calibration_settings.folder_plot))
        mkdir(calibration_settings.folder_plot)
    end
    
    
%     
%     
%     
%     calibration_settings.T1 = T1;
%     calibration_settings.back_2_base = 1/2*T1;