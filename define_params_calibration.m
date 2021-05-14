function params = define_params_calibration()
%{ 
    function that contains the required general parameters to be used on the bmi
%}
    % timers in seconds
    params.sec_per_reward_range = [100 85];     
    params.target_on_cov_bool = 0;
    params.maxiter = 10000;
    params.back2base_alpha = 0.5;
    params.T_delta = 0.01;

    
    % plotting
    params.plot_calibration = true;
    params.plot_audio_mapping = true;
    %E1: blue-ish.  E2: orange-ish
    params.plot_raster_colors = {[0    0.4470    0.7410], [0.8500    0.3250    0.0980]}; 