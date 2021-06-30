function calibration_results = obtain_calibration(ensemble_base_activity,  ...
    calibration_settings , task_settings)
%{
    Function to obtain the calibration of the BMI
     inputs:
    ensemble_base_activity -> matrix MxN where M is the number of ensemble
    neurons and N is time
    - E1_ind - E1 idxs in baseline data
    - E2_ind - E2 idxs in baseline data
    - calibration_settings - all settings/parameters to calibrate the BMI
    - task_settings - all settings/parameteres needed to run the BMI
    %}

    % parameter
    
    
    %% BMI data from baseline 
    %Throw out prefix frames:
    ind = [calibration_settings.E1_ind calibration_settings.E2_ind];
    f_raw = ensemble_base_activity(ind, (task_settings.params.initial_count + 1):end); %first E1, then E2


    %% Decoder information
    decoder = def_decoder(calibration_settings.units, calibration_settings.E_id);
  
    %%Calculate f0 as in BMI: 
    f0 = zeros(calibration_settings.units, size(f_raw,2)-task_settings.base_frames+1); 
    f0(:, 1) = mean(f_raw(:, 1:task_settings.base_frames), 2);
    for i = 2:length(f0)
        f0(:, i) = f0(:, i-1)*((task_settings.base_frames-1)/ ...
            task_settings.base_frames) + ... 
            f_raw(:, (i+task_settings.base_frames-1))/ ...
            task_settings.base_frames; 
    end
    %Truncate data based on the task_settings.base_frames:
    f_postf0 = f_raw(:, task_settings.base_frames:end); 

    
    %% smooth f:
    if task_settings.params.f0_win > 0     
        f_smooth = zeros(calibration_settings.units, size(f_postf0,2)); 
        smooth_filt = ones(task_settings.params.f0_win,1)/task_settings.params.f0_win;     
        for i=1:calibration_settings.units
            f_smooth(i, :) = conv(f_postf0(i, :), smooth_filt, 'same'); 
        end
    else
        f_smooth = f_postf0; 
    end
    
    %% compute dff:
    dff = (f_smooth-f0)./f0;
    % to remove nans (which will occur on all units at the same time
    dff = dff(:, (~isnan(dff(1, :))));  % TODO CHECK IF THIS IS TRUE 

    %% Variables required for calibration
    % cursor
    cursor_obs = decoder * dff; 
    
    % T 
    T0 = max(cursor_obs);
    T_vec = []; 
    reward_per_frame_vec = []; 
    
    %% Iterate to find T
    iter = 0;
    task_complete = false;
    T = T0;
    while(~task_complete)
        if(iter == calibration_settings.params.max_iter)
            task_complete = true;
            error('Max Iter reached, check reward rate / baseline data'); 
        else
            T_vec = [T_vec T];
            %1) E2-E1 > alpha
            hit_idxs_no_b2base = find(cursor_obs >= T); 
            
            %Remove hits that fall in a back2base
            hits_valid = ones(length(hit_idxs_no_b2base),1, 'logical'); 
            if length(hit_idxs_no_b2base) > 1
                for i = 2:length(hit_idxs_no_b2base)
                    hits_valid(i) = sum(cursor_obs(hit_idxs_no_b2base(i-1):hit_idxs_no_b2base(i)) <= ...
                        calibration_settings.params.back2base_alpha*T) >= ...
                        task_settings.params.back_2_base_frame_thresh;
                end
            end
            valid_hit_idxs = hit_idxs_no_b2base(hits_valid); 
            reward_prob_per_frame   = sum(hits_valid)/length(dff);    

            reward_per_frame_vec = [reward_per_frame_vec reward_prob_per_frame]; 
            %Update T:
            if((reward_prob_per_frame >= calibration_settings.reward_per_frame_range(1)) ...
                    && (reward_prob_per_frame <= calibration_settings.reward_per_frame_range(2)))
                task_complete = true;
                disp('target calibration complete!');
            elseif(reward_prob_per_frame > calibration_settings.reward_per_frame_range(2))
                %Task too easy, make T harder:
                T = T + calibration_settings.params.T_delta; 
            elseif(reward_prob_per_frame < calibration_settings.reward_per_frame_range(1))
                %Task too hard, make T easier:
                T = T - calibration_settings.params.T_delta; 
            end
            iter = iter+1;
        end

    end 

    %%
    % Calculate parameters for auditory feedback
    fb_mapping = cursor_to_audio_feedback(cursor_obs, task_settings.params.fb_settings, T);
    
    %% to return results
    calibration_results.cursor_obs = cursor_obs;
    calibration_results.T = T;
    calibration_results.dff = dff;
    calibration_results.decoder = decoder;
    calibration_results.T_vec = T_vec;
    calibration_results.hit_idxs_no_b2base = hit_idxs_no_b2base;
    calibration_results.valid_hit_idxs = valid_hit_idxs;
    calibration_results.reward_prob_per_frame = reward_prob_per_frame;
    calibration_results.reward_per_frame_vec = reward_per_frame_vec;
    calibration_results.fb_mapping = fb_mapping;
    
    %% plots: 
    if calibration_settings.params.plot_calibration
        % plot raw signals
        plot_2signals(f_postf0, f0, calibration_settings.E_id, ...
            calibration_settings.params.plot_raster_colors, ...
            'frame', 'fluorescence', 'rolling baseline', ...
            calibration_settings.folder_plot, 'rolling_baseline', ind, 0, -mean(f_postf0,2))
        
        % plot smooth 
        if task_settings.params.f0_win > 0
            plot_2signals(f_postf0, f_smooth, calibration_settings.E_id, ...
                calibration_settings.params.plot_raster_colors, ...
                'frame', 'fluorescence', 'f smooth', ...
                calibration_settings.folder_plot, 'f_smooth', ind)
        end
        
        % plot covariance
        plot_covariance(ensemble_base_activity(ind, :)', calibration_settings.folder_plot)

        %%
        fb_obs = cursor_to_audio(cursor_obs, fb_mapping, task_settings.mice_settings.target_low); % cursor2audio_freq(cursor_obs, cal);
        num_fb_bins = 100; 
        h = figure;
        histogram(fb_obs, num_fb_bins); 
        plot_saving(h, 'audio freq', 'baseline counts', 'cursor', ...
            calibration_settings.folder_plot, 'base_freq_hist.png'); 
        %%
       
        offset = 0; 
        legend_str = {'cursor', 'E1', 'E2'};
        signal_n = [cursor_obs; nanmean(dff(calibration_settings.E1_ind,:),1); nanmean(dff(calibration_settings.E2_ind,:),1)];
        h = plot_nsignals(signal_n, '', '', '', '', '', offset, legend_str, false);
        for i=1:length(valid_hit_idxs)
            xline(valid_hit_idxs(i), '--');
        end
        plot_saving(h, 'Time (frames)', 'DFF', ['valid_hits: '  num2str(length(valid_hit_idxs))], ...
            calibration_settings.folder_plot, 'cursor_E1_E2')
        

        %%
        %Plot PSTH of neural activity locked to target hit: 
        psth_win = [-10 3]*task_settings.frame_rate; 
        [psth_mean, psth_sem, ~] = calc_psth(dff, valid_hit_idxs, psth_win);
        
        h = figure; hold on;
        offset = 0; 
        for i=1:calibration_settings.units
            y_plot = psth_mean(i,:); 
            y_plot = y_plot-min(y_plot);
            y_amp = max(y_plot); 
            offset = offset + y_amp; 
            y_sem = psth_sem(i, :)-min(y_plot); 

            plot(y_plot-offset, ...
                'Color', calibration_settings.params.plot_raster_colors{calibration_settings.E_id(i)}); 
            errbar(1:length(y_plot), y_plot-offset,y_sem, ...
                'Color', calibration_settings.params.plot_raster_colors{calibration_settings.E_id(i)});
        end
        xline(abs(psth_win(1))); 
        plot_saving(h, 'frame', 'dff', 'PSTH of Baseline Activity Locked to Target Hit', ...
            calibration_settings.folder_plot, 'PSTH_locked_to_hit_baseline.png'); 
    end
    
    if calibration_settings.params.plot_audio_mapping
        %% Plot auditory feedback
        plot_cursor = linspace(min(cursor_obs), max(cursor_obs), 1000); 
        plot_freq   = cursor_to_audio(plot_cursor, fb_mapping, task_settings.mice_settings.target_low);
        h = figure;
        plot(plot_cursor, plot_freq, 'linewidth',3); 
        hold on
        ylabel('Audiory Freq'); 
        xline(T); 
        yyaxis right
        plot(cursor_obs, 1:length(cursor_obs))
        plot_saving(h, 'Cursor E2-E1', 'Frame', 'freq_cursor', ...
            calibration_settings.folder_plot, 'cursor2freq.png');

    end

        
    %% Save the results: 
    date_str = datestr(datetime('now'), 'yyyymmddTHHMMSS'); 
    save(fullfile(task_settings.folder_path, ['BMI_target_info_' date_str '.mat']), 'calibration_results'); 

end

function decoder = def_decoder(num_neurons, E_id)
%{
    function to return the decoder weights
    %}

    E1_sel = E_id==1;  
    E2_sel = E_id==2; 
    E1_proj = zeros(1, num_neurons); 
    E1_proj(E1_sel) = 1;
    E1_norm = sum(E1_sel); %can replace with vector norm.   
    E1_proj = E1_proj/E1_norm;

    E2_proj = zeros(1, num_neurons, 1); 
    E2_proj(E2_sel) = 1; 
    E2_norm = sum(E2_sel); 
    E2_proj = E2_proj/E2_norm;

    decoder = E2_proj - E1_proj;
end

function [fb_mapping] = cursor_to_audio_feedback(cursor_obs, fb_settings, T)
%{ 
    function to map cursor to auditory feedback.
    Assumes cursor = E2-E1, and T is positive.
    freq = a*exp(b*(cursor_trunc-cursor_min))
    %}

    fb_mapping.cursor_min = prctile(cursor_obs, fb_settings.min_prctile); 
    fb_mapping.cursor_max = T; 
    fb_mapping.cursor_range = fb_mapping.cursor_max - fb_mapping.cursor_min; 
    % % freq = a*exp(b*(cursor_trunc-cursor_min))
    fb_mapping.a = fb_settings.freq_min; 
    fb_mapping.b = (log(fb_settings.freq_max) - ...
        log(fb_mapping.a))/fb_mapping.cursor_range; 
end

