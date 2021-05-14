function calibration_results = obtain_calibration(ensemble_base_activity,  ...
    E1_base, E2_base, calibration_settings , task_settings)
%{
    Function to obtain the calibration of the BMI
     inputs:
    ensemble_base_activity -> matrix MxN where M is the number of ensemble
    neurons and N is time
    - E1_base - E1 idxs in baseline data
    - E2_base - E2 idxs in baseline data
    - calibration_settings - all settings/parameters to calibrate the BMI
    -task_settings - all settings/parameteres needed to run the BMI
    %}

    %% BMI data from baseline 
    %Throw out prefix frames:
    f_raw = [ensemble_base_activity(E1_base, (task_settings.params.initial_count + 1):end) 
        ensemble_base_activity(E2_base, (task_settings.params.initial_count + 1):end)]; %first E1, then E2


    %% Decoder information
    E_id = [1*ones(length(E1_base), 1); 2*ones(length(E2_base), 1)];
    decoder = def_decoder(calibration_settings.units, E_id);
  
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
    cursor_obs = dff * decoder; 
    calibration_results.cursor_obs = cursor_obs;

    % T 
    T0 = max(cursor_obs);
    T_vec = []; 
    reward_per_frame_vec = []; 
    
    %% Iterate to find T
    iter = 0;
    task_complete = false;
    T = T0;
    while(~task_complete)
        if(iter == max_iter)
            task_complete = true;
            disp('Max Iter reached, check reward rate / baseline data'); 
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
            elseif(reward_prob_per_frame > reward_per_frame_range(2))
                %Task too easy, make T harder:
                T = T + calibration.params.T_delta; 
            elseif(reward_prob_per_frame < reward_per_frame_range(1))
                %Task too hard, make T easier:
                T = T - calibration.params.T_delta; 
            end
            iter = iter+1;
        end

    end 

    %% 
    cursor_amp = (max(cursor_obs)-min(cursor_obs));
    cursor_offset = cursor_amp/10; 
    max_cursor = max(cursor_obs); 

    %%
    % Calculate parameters for auditory feedback
    fb_mapping = cursor2audio_fb(cursor_obs, task_settings.params.fb_settings, T);
    
    %% to return results
    calibration_results.T = T;
    calibration_results.T_vec = T_vec;
    calibration_results.hit_idxs_no_b2base = hit_idxs_no_b2base;
    calibration_results.valid_hit_idxs = valid_hit_idxs;
    calibration_results.reward_prob_per_frame = reward_prob_per_frame;
    calibration_results.reward_per_frame_vec = reward_per_frame_vec;
    calibration_results.fb_mapping = fb_mapping;
    
    %% plots: 
    if calibration_settings.params.plot_calibration
        % plot raw signals
        plot_2signals(f_postf0, f0, E_id, ...
            calibration_settings.params.plot_raster_colors, ...
            'frame', 'fluorescence', 'rolling baseline', ...
            calibration_settings.folder_plot, 'rolling_baseline', 0, -mean(f_postf0,2))
        
        % plot smooth 
        if task_settings.params.f0_win > 0
            plot_2signals(f_postf0, f_smooth, E_id, ...
                calibration_settings.params.plot_raster_colors, ...
                'frame', 'fluorescence', 'f smooth', ...
                calibration_settings.folder_plot, 'f_smooth')
        end

        % plot dff
        plot_1signal(dff, E_id, calibration_settings.params.plot_raster_colors, ...
            'frame', 'dff', 'dff', ...
            calibration_settings.folder_plot, 'dff')
        
        % plot covariance
        plot_covariance(neuronal_activity, calibration_settings.folder_plot)
    end
    
    if params.plot_audio_mapping
        %% Plot auditory feedback
        plot_cursor = linspace(min(cursor_obs), max(cursor_obs), 1000); 
        plot_freq   = cursor2audio_freq_v2(plot_cursor, fb_mapping, mice_settings.target_low);
        h = figure;
        plot(plot_cursor, plot_freq); 
        xlabel('Cursor E2-E1'); 
        ylabel('Audiory Freq'); 
        vline(T); 
        saveas(h, fullfile(plotPath, 'cursor2freq.png')); 

        %%
        fb_obs = cursor2audio_freq_v2(cursor_obs, fb_mapping); % cursor2audio_freq(cursor_obs, cal);
        num_fb_bins = 100; 
        h = figure;
        hist(fb_obs, num_fb_bins); 
        xlabel('audio freq'); 
        ylabel('baseline counts'); 
        saveas(h, fullfile(plotPath, 'base_freq_hist.png')); 

        %%
        h =figure; hold on;
        scatter(c1, ones(length(c1),1)*max_cursor + cursor_offset, 15, 'r'); %plot(cursor_obs-cursor_offset, 'k'); 
        scatter(c2, ones(length(c2),1)*max_cursor + 2*cursor_offset, 15, 'g'); %plot(cursor_obs-cursor_offset, 'k'); 
        scatter(c3, ones(length(c3),1)*max_cursor + 3*cursor_offset, 15, 'b'); %plot(cursor_obs-cursor_offset, 'k'); 
        plot(cursor_obs); 
        hline(T); 
        plot(E1_meadff-cursor_amp); 
        plot(E2_subord_meadff-2*cursor_amp); 
        xlabel('frame'); 
        title(['hits with b2base: ' num2str(num_valid_hits)]); 
        legend({'c1', 'c2 - E1 cond', 'c3 - E2 cond', 'cursor', 'E1 mean', 'E2 subord mean'}); 
        vline(valid_hit_idxs); 
        saveas(h, fullfile(plotPath, 'cursor_hit_ts.png')); 

        %%
        offset = 0; 
        [h, offset_vec] = plot_cursor_E1_E2_activity(cursor_obs, E1_meadff, E2_meadff, dff, E_id, E_color, offset)
        hold on; hline(T); 
        saveas(h, fullfile(plotPath, 'cursor_E1_E2_ts.png')); 
        %%
        cursor_obs = dff*decoder; 
        h = figure;
        hold on; 
        hist(cursor_obs, 50); 
        vline(T); 
        xlabel('Cursor'); 
        ylabel('Number of Observations'); 
        title(['E2-E1 thr on E2-E1 hist, num valid hits: ' num2str(num_valid_hits) ...
            ' num hits no b2base: ' num2str(num_hits_no_b2base) ...
            ' num cursor hits: ' num2str(num_cursor_hits)]); 
        saveas(h, fullfile(plotPath, 'cursor_dist_T.png')); 

        % %%
        % %Plot the hit times: 
        % [h, offset_vec] = plot_E_activity(dff, E_id, E_color);
        % xlabel('frame'); 
        % title(['Num Baseline Hits ' num2str(num_hits)]); 
        % offset = 5; 
        % %c1:
        % c1_offset = offset_vec(end)+offset;
        % plot(1:length(cursor_obs), cursor_obs-c1_offset);
        % % hline(T-c1_offset)
        % 
        % %c2:
        % c2_offset = offset_vec(end)+2*offset;
        % plot(1:length(E1_meadff), E1_meadff-c2_offset);
        % % hline(E1_thresh-c2_offset)
        % 
        % %c3:
        % c3_offset = offset_vec(end)+3*offset;
        % plot(E2_subord_meadff-c3_offset); 
        % plot(E2_subord_thresh(E2_dom_sel)-c3_offset);
        % 
        % % for i=1:length(hit_times)
        % %     vline(hit_times(i)); 
        % % end
        % 
        % saveas(h, fullfile(plotPath, 'neural_hit_constraints.png')); 

        %%
        %Plot PSTH of neural activity locked to target hit: 
        psth_win = [-30 30]*3; 
        [psth_mean, psth_sem, psth_mat] = calc_psth(dff, valid_hit_idxs, psth_win);
        h = figure; hold on;
        offset = 0; 
        for i=1:calibration_settings.units
            y_plot = psth_mean(:,i); 
            y_plot = y_plot-min(y_plot);
            y_amp = max(y_plot); 
            offset = offset + y_amp; 
            y_sem = psth_sem(:,i)-min(y_plot); 

            plot(y_plot-offset, 'Color', E_color{(E_id(i))}); 
            errbar(1:length(y_plot), y_plot-offset,y_sem, 'Color', E_color{(E_id(i))}); 
        end
        % vline((psth_win(2)-psth_win(1))/2+1); 
        xlabel('frame')
        title('PSTH of Baseline Activity Locked to Target Hit'); 

        saveas(h, fullfile(plotPath, 'PSTH_locked_to_hit_baseline.png')); 

        % %%
        % h = figure; hold on;
        % for i =1:size(psth_mat,3)
        %     plot(psth_mat(:,2,i)); 
        % end
        %
        %%
        %Save the results: 
        %1) All the steps here
        %2) Just the target parameters for running BMI

        %1)All the steps here
        clear h
        date_str = datestr(datetime('now'), 'yyyymmddTHHMMSS'); 
        save_path = fullfile(save_dir, ['target_calibration_ALL_' date_str '.mat']); 
        target_cal_ALL_path = save_path; 
        save(save_path); 

        %2)Just the target parameters for running BMI
        target_info_file = ['BMI_target_info_' date_str '.mat'];
        save_path = fullfile(save_dir, target_info_file); 
        target_info_path = save_path; 
        %Change variable names for BMI code:
        T1 = T; %Change to T1, as this is what BMI expects
        save(save_path, 'AComp_BMI', 'n_mean', 'n_std', 'decoder', 'E_id', 'E1_sel_idxs', 'E2_sel_idxs', 'E1_base', 'E2_base', 'T1', 'E1_thresh', 'E1_coeff', 'E1_std', 'E2_subord_thresh', 'E2_coeff', 'E2_subord_mean', 'E2_subord_std'); 

        disp('T'); 
        T
    end
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

function [fb_mapping] = cursor2audio_fb(cursor_obs, fb_settings, T)
%{ 
    function to map cursor to auditory feedback.
    Assumes cursor = E2-E1, and T is positive.
    freq = a*exp(b*(cursor_trunc-cursor_min))
    %}

    fb_mapping.cursor_min = prctile(cursor_obs, fb_settings.min_perctile); 
    fb_mapping.cursor_max = T; 
    fb_mapping.cursor_range = fb_mapping.cursor_max - fb_mapping.cursor_min; 
    % % freq = a*exp(b*(cursor_trunc-cursor_min))
    fb_mapping.a = fb_mapping.settings.freq_min; 
    fb_mapping.b = (log(fb_mapping.settings.freq_max) - ...
        log(fb_mapping.a))/fb_mapping.cursor_range; 
end

