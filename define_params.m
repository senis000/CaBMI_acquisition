function params = define_params()
%{ 
    function that contains the required parameters to be used on the bmi
%}
    % timers in seconds
    params.base_fength = 2*60; % number of seconds without BMI to stablish Baseline
    params.moving_average = 1; % Moving average in sec to calculate BMI 
    params.relaxation_time = 4;  % there can't be another hit in this many sec
    params.timeout = 5; %seconds of timeout if no hit in duration trial (sec)
    
    % feedback
    params.fb_settings.target_low_freq        = 1; 
    %Set the target cursor value to be the low frequency
    params.fb_settings.freq_min               = 6000; 
    params.fb_settings.freq_max               = 20000; 
    params.fb_settings.arduino.com            = 'COM15';
    params.fb_settings.arduino.name            = 'Uno';
    params.fb_settings.arduino.duration       = 0.3; %ms, tones update at rate of BMI code, this is the longest a tone will play for
    params.fb_settings.min_prctile            = 10; %The lowest percentile allowed for E2 minus E1
    params.fb_settings.max_prctile            = 100; %The lowest percentile allowed for E2 minus E1
    params.fb_settings.middle_prctile         = 50; 
    
    % parameters
    params.back_2_base_frame_thresh = 2; %need to be back2Base for x frames before another target can be achieved
    params.duration_trial = 30; % maximum time that mice have for a trial
    params.initial_count = 40;  % count to start experiment
    params.motion_thresh = 6;  %value to define maximum movement to flag a motion-relaxation time

    return 