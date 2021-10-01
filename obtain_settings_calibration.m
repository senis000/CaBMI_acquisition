function calibration_settings = obtain_settings_calibration (E1, E2, folder_path, baseline_file_path, debug_mode)
%{ function to initialize the variables that the BMI needs %}
    
    %% input settings
    calibration_settings.baseline_file_path = baseline_file_path;
    calibration_settings.E2 = E2;
    calibration_settings.E1 = E1;
    calibration_settings.units = length(E1)+length(E2); 
    
    [E_sorted, ind] = sort([E1 E2]);
    E1_ind = zeros(1,length(E1));
    E2_ind = zeros(1,length(E1));
    for i = 1:length(E1)
        e_aux = find(E_sorted==E1(i));
        if ~isempty(e_aux)
            E1_ind(i) = e_aux;
        end
    end
    for i = 1:length(E2)
        e_aux = find(E_sorted==E2(i));
        if ~isempty(e_aux)
            E2_ind(i) = e_aux;
        end
    end
        

    E_id = ones(1, length(ind));
    E_id(E1_ind) = 1;
    E_id(E2_ind) = 2;
    
    calibration_settings.E_id = E_id;
    calibration_settings.E_sorted = E_sorted;
    calibration_settings.E1_ind = E1_ind;
    calibration_settings.E2_ind = E2_ind;

    %% parameteres
    calibration_settings.params = define_params_calibration();
    
    if nargin < 5
        debug_mode = false;
    end
    
    %% parameters depending on frames
    if debug_mode
        calibration_settings.frame_rate = 10;
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