function params = define_params_task()
%{ 
    function that contains the required general parameters to be used on the bmi
%}
    % timers in seconds
    params.base_length = 2*60; % number of seconds without BMI to stablish Baseline
    params.moving_average = 1; % Moving average in sec to calculate BMI 
    params.relaxation_time = 4;  % there can't be another hit in this many sec
    params.timeout = 5; %seconds of timeout if no hit in duration trial (sec)
    params.trial_max_time = 30;  % max seconds in trial
    params.water_time = 0.005;  % amount of water reward
    params.stim_pulse = 0.005;  % pulse width
    
    % feedback
    params.fb_settings.target_low_freq        = 1; 
    %Set the target cursor value to be the low frequency
    params.fb_settings.freq_min               = 6000; 
    params.fb_settings.freq_max               = 20000; 
    params.fb_settings.arduino.com            = 'COM15';
    params.fb_settings.arduino.name            = 'Uno';
    params.fb_settings.arduino.duration       = 0.3; %ms, tones update at rate of BMI code, this is the longest a tone will play for
    params.fb_settings.min_prctile            = 5; %The lowest percentile allowed for E2 minus E1
    params.fb_settings.max_prctile            = 100; %The highest percentile allowed for E2 minus E1
    params.fb_settings.middle_prctile         = 50; 
    
    % parameters
    params.back_2_base_frame_thresh = 2; %need to be back2Base for x frames before another target can be achieved
    params.f0_win = 4;  % number of frames to smooth over the raw frequency
    params.duration_trial = 30; % maximum time (sec) that mice have for a trial
    params.initial_count = 40;  % number of frames to start experiment
    params.motion_thresh = 6;  %value to define maximum movement to flag a motion-relaxation time
    params.length_trials = 500;  % over-estimated maximum number of trials per experiment
    params.block_at_alpha = 1/3; % value to define when to give blocking stim
    
    % ploting


    return 
