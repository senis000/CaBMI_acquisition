function [target_info_path, target_cal_ALL_path, fb_cal] = obtain_calibration(n_f_file, Acomp_file, onacid_bool,  ...
    E1_base, E2_base, frames_per_reward_range, target_on_cov_bool, ...
    prefix_win, f0_win_bool, f0_win, dff_win_bool, dff_win, save_dir, ...
    cursor_zscore_bool, f0_init_slide, E2mE1_prctile, fb_settings, task_settings)
%{

inputs:
-n_f_file - contains matrix, neural fluorescence from baseline file, num_samples X num_neurons_baseline 
-Acomp_file - 
-E1_base - E1 idxs in baseline data
-E2_base - E2 idxs in baseline data
-frames_per_reward_range - a range on how many frames should elapse before a
reward is expected.  Used to calibrate the target patterns.
-target_on_cov_bool - if false, calibrate target to baseline data directly.
if true, use the covariance of data to ca librate target.  Potentially
useful if the data itself does not exhibit strong co-modulation of E
units.
-prefix_win - number of frames at start of baseline that calibration
should NEGLECT.  (these starting frames are just ignored.)
-f0_win_bool - if true, estimate f0 with a window of activity.  if false, estimate f0 using the full baseline, 
-f0_win - number of frames to use for estimating f0.  (This code uses a
rolling average, as used during the BMI)
-dff_win_bool - whether to smooth dff in a window
-dff_win - number of frames to use for smoothing dff
-save_dir - directory to save baseline results in in.
-cursor_zscore_bool - if 1, neural activity is zscored before going into
-cursor calculation. if 0, neural activity is not zscored.  
-f0_init_slide - if 0, f0 is only used after f0_win samples.  if 1, f0 is
adapted in the window from 0 to f0_win samples.
-E2mE1_prctile: it is the lowest acceptable E2mE1_prctile for deciding the
target threshold.  
%}

% %%
% %Debug parameters: 
% % (n_f_file, E_id, f0_win_bool, f0_win, dff_win_bool, dff_win, save_dir)
% 
% 
% folder = '/Users/vivekathalye/Dropbox/Data/holo_bmi_debug_190512'
% animal = 'test'
% day = 'test'
% 
% save_dir = fullfile(folder, animal, day); %[folder, animal, '/',  day, '/'];
% if ~exist(save_dir, 'dir')
%     mkdir(save_dir);
% end
% 
% baselineCalibrationFile = fullfile(save_dir, 'BMI_target_info.mat'); 
% baselineDataFile = fullfile(save_dir, 'BaselineOnline190512T025909.mat'); 
% n_f_file = baselineDataFile;
% exist(baselineCalibrationFile)
% exist(baselineDataFile)
% load(baselineCalibrationFile)
% 
% target_on_cov_bool = 0; 
% 
% frame_rate = 30; 
% sec_per_reward_range = [100 80]; 
% frames_per_reward_range = sec_per_reward_range*frame_rate; %[1.5*60*frame_rate 1*60*frame_rate];
% prefix_win = 40; %number frames to remove from the start of imaging
% f0_win_bool = 1;
% f0_win = 2*60*frame_rate; %frames
% dff_win_bool = 1; 
% dff_win = 2; %frames
% f0_init_slide = 0; 
% cursor_zscore_bool = 1;
% on_acid_bool = 0;

%OLD DEBUG INPUTS: 
% load(fullfile(savePath, baselineDataFile)); 
% %baseActivity
% %removenan:
% f_data = baseActivity; 
% f_data(:,isnan(f_data(1,:))) = [];
% num_samples = size(f_data, 2); 


% data_dir = '/Users/vivekathalye/Dropbox/Work/projects/holo_BMI/baseline'; 
% n_f_file = fullfile(data_dir, 'baseline_IntegrationRois_00001_1.csv'); 
% 
% save_dir = '/Users/vivekathalye/Dropbox/Work/projects/holo_BMI/baseline/results'
% mkdir(save_dir); 


% target_on_cov_bool = 0; 
% 
% frame_rate = 10; 
% frames_per_reward_range = [1.5*60*frame_rate 1*60*frame_rate];
% prefix_win = 40; %number frames to remove from the start of imaging
% f0_win_bool = 1;
% f0_win = 2*60*frame_rate; %frames
% dff_win_bool = 1; 
% dff_win = 2; %frames
% 
% 
% %Load baseline data:
% A = csvread(n_f_file, 1, 0);
% ts = A(:,1); 
% frame = A(:,2); 
% n = A(:, 3:end); 
% num_n = size(n,2); 
% [n_var, I] = sort(var(n, 0, 1), 'descend');
% n = n(:,I); 
% n(:,find(isnan(n_var))) = [];
% n_var(find(isnan(n_var))) = [];
% 
% direct = [1 2 4 5 6 17 27 39]; %Hand chosen for debugging
% f_raw = n(:,direct).'; %direct neuron fluorescence
% %Transposed to be num_neurons x num_samples, as expected for the input to
% %this file
% E_id = [2 2 2 2 1 1 1 1]'; 

%%
% back2BaseFramesThresh = 2; %TODO: make this an input which is shared with BMI

%%
%Plots to show: 
%Plot colors: 
% plot_b = [0    0.4470    0.7410];
% plot_o = [0.8500    0.3250    0.0980]; 
% E_color = {plot_b, plot_o}; 
% %E1: blue-ish.  E2: orange-ish
% 
% plot_raw_bool = 1; 
% plot_f0_bool = 1; 
% plot_smooth_bool = 1; 
% plot_dff_bool = 1; 
% plot_cov_bool = 1; 
% %-

% plotPath = fullfile(save_dir, 'plots'); 
% if ~exist(plotPath,'dir')
%     mkdir(plotPath); 
% end
%%sa
%Select BMI ensemble temporal activity from baseline
%Create Ensemble Information
%Select BMI ensemble spatial components
%Decoder Information


%1) Select Temporal: BMI E data from baseline 

load(n_f_file); 
size(baseActivity)
f_base = baseActivity; 
%n_f_file - contains matrix, neural fluorescence from baseline file, num_samples X num_neurons_baseline 
f_base(:,isnan(f_base(1,:))) = [];
f_base = f_base.'; %num_samples X num_neurons

%f_base = f_base(1:18000,:)

% f_base = f_base(1:1000, :); %debugging input data with nans... 
%Assume variable is called f_base

E1_temp = f_base(:,E1_base); 
E2_temp = f_base(:,E2_base); 
f = [E1_temp E2_temp]; 

%Throw out prefix frames:
E1_raw = f_base((prefix_win+1):end,E1_base); 
E2_raw = f_base((prefix_win+1):end,E2_base); 
f_raw = [E1_raw E2_raw]; %first E1, then E2

%2) Ensemble information
num_E1 = length(E1_base); 
num_E2 = length(E2_base); 
num_neurons = num_E1 + num_E2;

E_id = [1*ones(num_E1, 1); 2*ones(num_E2, 1)]; 
E1_sel = E_id==1; 
E1_sel_idxs = find(E1_sel); 
E2_sel = E_id==2; 
E2_sel_idxs = find(E2_sel); 

%%
%3) Select Spatial Components for BMI E
% % Uncomment and test with a mosue:

E_base_sel = [E1_base, E2_base];

if onacid_bool
    load(Acomp_file, 'AComp');
    AComp_BMI = AComp(:, E_base_sel);
    strcMask = obtainStrcMask(AComp_BMI, px, py);
    save(fullfile(save_dir, 'strcMask.mat'), 'strcMask', 'E_base_sel', 'E_id');
else
    AComp_BMI = []
    load(Acomp_file, 'roi_mask'); 
    holoMask = roi_mask; 
    EnsembleMask = zeros(size(holoMask));
    for indn = 1:length(E1_base)
        auxmask = holoMask;
        auxmask(auxmask~=E1_base(indn)) = 0;
        auxmask(auxmask~=0) = indn;
        EnsembleMask = auxmask + EnsembleMask;
    end
    for indn = 1:length(E2_base)
        auxmask = holoMask;
        auxmask(auxmask~=E2_base(indn)) = 0;
        auxmask(auxmask~=0) = indn + length(E1_base);
        EnsembleMask = auxmask + EnsembleMask;
    end
    strcMask = obtainStrcMaskfromMask(EnsembleMask);
    save(fullfile(save_dir, 'strcMask.mat'), 'strcMask', 'E_base_sel', 'E_id'); 
end

%%
%4) Decoder information
%Decoder information

[decoder, E1_proj, E2_proj, E1_norm, E2_norm] = ...
    def_decoder(num_neurons, E1_sel, E2_sel);

%%
%First process f0: 
if(f0_win_bool)    
    %Calculate f0 as in BMI: 
    
    if f0_init_slide
        num_samples = size(f_raw,1);
        f0 = zeros(num_samples,num_neurons); 
        for i=1:length(f0)
            if i==1
                f0(i,:) = f_raw(i,:);
            elseif i<f0_win
                f0(i,:) = (f0(i-1,:)*(i-1)+f_raw(i,:))/i;
            else
                f0(i,:) = (f0(i-1,:)*(f0_win-1)+f_raw(i,:))/f0_win;
            end
        end
        f_postf0 = f_raw;
        f0_mean = repmat(nanmean(f_postf0, 1), size(f_postf0,1), 1);
    else
        num_samples = size(f_raw,1);
        f0 = zeros(num_samples-f0_win+1, num_neurons); 
        f0(1,:) = mean(f_raw(1:f0_win, :), 1);
        for i = 2:length(f0)
            f0(i,:) = f0(i-1,:)*((f0_win-1)/f0_win) + f_raw((i+f0_win-1), :)/f0_win; 
        end
        %Truncate data based on the f0_win:
        f_postf0 = f_raw(f0_win:end, :); 
        f0_mean = repmat(nanmean(f_postf0, 1), size(f_postf0,1), 1);        
    end
else
    f_postf0 = f_raw; 
    f0_mean = repmat(nanmean(f_postf0, 1), size(f_postf0,1), 1);
    f0 = f0_mean; 
end

%%
%--------------------------------------------------------------------------
%raw data plot: 
if(plot_raw_bool)
    t_plot = 1:length(f_postf0);
    [h, offset_vec] = plot_E_activity(t_plot, f_postf0, E_id, E_color);
    
%     h = figure;
%     hold on; 
%     offset = 0; 
%     for i=1:num_neurons
%         y_plot = f_raw(:,i);
%         y_plot = y_plot-min(y_plot); 
%         y_amp = max(y_plot); 
% 
%         if(i>1)
%             offset = offset+y_amp; 
%         end
% 
%         plot_color = E_color{E_id(i)};
%         plot(y_plot-offset, 'Color', plot_color); 
%     end
    xlabel('frame'); 
    ylabel('fluorescence'); 
    title('raw fluorescence in baseline'); 
    im_path = fullfile(plotPath, 'baseline_fraw.png'); 
    saveas(h, im_path); 
end

%%
%Compare f0win to f0mean:
if(plot_f0_bool)
    if(f0_win_bool)
        %Plot for one neuron: 
        n_i = 1; 
        h = figure; hold on;
        plot(f_postf0(:,n_i)); 
        plot(f0_mean(:,n_i), 'LineWidth', 5); 
        plot(f0(:,n_i), 'LineWidth', 5); 
        legend({'fraw', 'f0mean', 'f0win'})
%         h = figure; hold on; 
%         plot(f0(:,n_i)); 
%         plot(f0_mean(:,n_i)); 
%         legend({'f0win', 'f0mean'})
        xlabel('frame'); 
        ylabel('fluorescence'); 
        title('F0 for one neuron'); 
        saveas(h, fullfile(plotPath, 'f0.png')); 
    else
        %Plot for one neuron: 
        n_i = 1; 
        h = figure; hold on;
        plot(f_postf0(:,n_i)); 
        plot(f0_mean(:,n_i), 'LineWidth', 5); 
        legend({'fraw', 'f0mean'})
%         h = figure; hold on; 
%         plot(f0(:,n_i)); 
%         plot(f0_mean(:,n_i)); 
%         legend({'f0win', 'f0mean'})
        xlabel('frame'); 
        ylabel('fluorescence'); 
        title('F0 for one neuron');
        saveas(h, fullfile(plotPath, 'f0.png')); 
    end
end
%Note: a more sophisticated method would calculate f0 based on low-pass
%filtered calcium.  our f0 estimate is biased by ca transients.

%%
%Second, smooth f:
if(dff_win_bool)
    num_samples = size(f_postf0,1);     
	f_smooth = zeros(num_samples, num_neurons); 
    smooth_filt = ones(dff_win,1)/dff_win;     
    for i=1:num_neurons
        f_smooth(:,i) = conv(f_postf0(:,i), smooth_filt, 'same'); 
    end
else
    f_smooth = f_postf0; 
end

if(plot_smooth_bool && dff_win_bool)
	h = figure; hold on;
    plot(f_postf0(:,1)); 
    plot(f_smooth(:,1)); 
    legend({'f', 'f smooth'}); 
    xlabel('frame'); 
    ylabel('F'); 
    title('F vs smoothed F'); 
end

%%
%Third, compute dff and dff_z:
dff = (f_smooth-f0)./f0;
%mean center the dff:
n_mean = nanmean(dff,1); %1 x num_neurons
mean_mat = repmat(n_mean, size(dff,1), 1);
dffc = dff-mean_mat;
%divide by std:
n_std = nanstd(dffc, 0, 1); %var(dffc, 0, 1).^(1/2); %1 x num_neurons
dff_z = dffc./repmat(n_std, [size(dff,1) 1]); 
if(plot_dff_bool)
    %plot dff
    t_plot = 1:length(dff); 
    h = plot_E_activity(t_plot, dff, E_id, E_color);
    xlabel('frame'); 
    ylabel('dff'); 
    title('dff'); 
    im_path = fullfile(plotPath, 'dff.png'); 
    saveas(h, im_path);
    
    %plot dffz
    t_plot = 1:length(dff_z); 
    h = plot_E_activity(t_plot, dff_z, E_id, E_color);
    xlabel('frame'); 
    ylabel('dff_z');    
    title('zscore dff'); 
    im_path = fullfile(plotPath, 'dffz.png'); 
    saveas(h, im_path); 
end


%%
if cursor_zscore_bool
    n_analyze = dff_z;
else
    n_analyze = dff;
end
 
valid_idxs  = find(~isnan(n_analyze(:,1)));
n_analyze   = n_analyze(valid_idxs, :); 
analyze_cov = cov(n_analyze);
analyze_mean = nanmean(n_analyze); %takes mean along dim 1.  n_analyze is num_samples X num_neurons.
if(plot_cov_bool)
    h = figure;
    imagesc(analyze_cov); 
    colorbar
    axis square
    xlabel('roi')
    ylabel('roi')
    colormap;
    caxis([-0.2 0.5]); 
    title('neural cov'); 
    saveas(h, fullfile(plotPath, 'cov_mat_baseline.png'))

    [u,s,v] = svd(analyze_cov); 
    s_cumsum = cumsum(diag(s))/sum(diag(s)); 
    h = figure;
    plot(s_cumsum, '.-', 'MarkerSize', 20); 
    axis square
    xlabel('PC'); 
    ylabel('Frac Var Explained'); 
    title('DFF Smooth PCA Covariance');
    saveas(h, fullfile(plotPath, 'cov_pca_baseline.png'))
end

%%
%Cursor Cov:

cursor_cov = decoder'*analyze_cov*decoder; 
% cursor_cov = decoder'*test_cov*decoder; 
% cursor_cov

%%
%Inputs: 
%frames_per_reward_range
%cov_bool
reward_per_frame_range  = 1./frames_per_reward_range;
E1_mean                 = mean(analyze_mean(E1_sel));
E1_std                  = sqrt((E1_sel/num_E1)'*analyze_cov*(E1_sel/num_E1));
E2_subord_mean          = zeros(num_E2,1);
E2_subord_std           = zeros(num_E2,1); 
E1_analyze              = n_analyze(:,E1_sel); 
E2_analyze              = n_analyze(:,E2_sel); 
for E2_i = 1:num_E2
    subord_sel                      = E2_sel;
    subord_sel(E2_sel_idxs(E2_i))   = 0; 
    E2_subord_mean(E2_i)            = mean(analyze_mean(subord_sel));     
    var_i                           = subord_sel'*analyze_cov*subord_sel; 
    E2_subord_std(E2_i)             = sqrt(var_i);     
end

E2_sum_analyze = sum(E2_analyze,2); 

%signals needed for target detection:
cursor_obs                      = n_analyze*decoder; 
E1_mean_analyze                 = mean(E1_analyze,2);
E2_mean_analyze                 = mean(E2_analyze, 2); 
E1_mean_max                     = max(E1_mean_analyze); 
[E2_dom_samples, E2_dom_sel]    = max(E2_analyze, [], 2);
E2_subord_mean_analyze          = (E2_sum_analyze - E2_dom_samples)/(num_E2-1);

%
% h = figure;
% hist(mean(E1_analyze,2)); 
% vline(E1_mean); 
% vline(E1_mean+E1_std); 

%%
%Iterate on T value, until perc correct value is achieved using truncated
%neural activity

%T:
min_prctile         = E2mE1_prctile; %A good default is 98
T0                  = max(cursor_obs);
T                   = T0; 
T_min               = prctile(cursor_obs, min_prctile);

%E2:
E2_coeff0           = 0.5;
E2_coeff            = E2_coeff0; %multiplies the std dev, for figuring out E2_subord_thresh
E2_coeff_min        = 0.05; 
E2_subord_thresh    = E2_subord_mean+E2_subord_std*E2_coeff;

%E1:
E1_coeff0           = 0;
E1_coeff            = E1_coeff0;
E1_thresh           = E1_mean + E1_coeff*E1_std; %E1_mean_max; %E1_mean;


T_delta = 0.05;
E2_coeff_delta = 0.05; %0.05 
E1_coeff_delta = 0.05; %0.05 
task_complete = 0;

T_vec = []; 
E2_coeff_vec = []; 
E1_coeff_vec = []; 

reward_per_frame_vec = []; 

max_iter = 10000;
iter = 0;

%%
%If using data covariance:
rand_num_samples = 500000;
while(~task_complete)
    T_vec           = [T_vec T];
    E2_coeff_vec    = [E2_coeff_vec E2_coeff]; 
    E1_coeff_vec    = [E1_coeff_vec E1_coeff];    

    
    %1) E2-E1 > alpha
    c1 = find(cursor_obs >= T); 
    %2) E1 < mu
    c2 = find(E1_mean_analyze <= E1_thresh);
    %3) E2_subord > mu (anded with previous constraint)
    %For each idx, subtract the 
    c3 = find(E2_subord_mean_analyze >= E2_subord_thresh(E2_dom_sel)); 
    hit_idxs_no_b2base = intersect(intersect(c1, c2), c3);
    %Remove hits that fall in a back2base

    %----------------------------------------------------------------------
    %Remove hits that fall in a back2base
    b2base_thresh = 0.5*T;
    hits_valid = ones(length(hit_idxs_no_b2base),1); 
    if length(hit_idxs_no_b2base) > 1
        for i = 2:length(hit_idxs_no_b2base)
            b2base_bool = sum(cursor_obs(hit_idxs_no_b2base(i-1):hit_idxs_no_b2base(i)) <= b2base_thresh) >= back2BaseFramesThresh; 
            hits_valid(i) = b2base_bool; 
        end
    end
    hit_idxs_b2base         = hit_idxs_no_b2base(find(hits_valid)); 
    valid_hit_idxs          = hit_idxs_b2base;
    reward_prob_per_frame   = sum(hits_valid)/length(n_analyze);    
    %----------------------------------------------------------------------
    
    
    reward_prob_per_frame
    reward_per_frame_vec = [reward_per_frame_vec reward_prob_per_frame]; 
   
    %Update T:
    if((reward_prob_per_frame >= reward_per_frame_range(1)) && (reward_prob_per_frame <= reward_per_frame_range(2)))
        task_complete = 1;
        disp('target calibration complete!');
    elseif(reward_prob_per_frame > reward_per_frame_range(2))
        %Task too easy, make T harder:
        T = T+T_delta; 
    elseif(reward_prob_per_frame < reward_per_frame_range(1))
        %Task too hard, make T easier:
        T = T-T_delta; 
        %If we swept the full range of T, lower E2_coeff, reset T:
        if(T<T_min)
            disp('T through full range, lower E2_coeff, reset T:'); 
            T=T0; 
            E2_coeff = E2_coeff - E2_coeff_delta;
            E2_subord_thresh = E2_subord_mean+E2_subord_std*E2_coeff; 
        end
        %If we swept the full range of E2, increase E1_coeff, reset E2:
        if(E2_coeff < E2_coeff_min)
            disp('E2 coeff through full range, increase E1_coeff, reset E2_coeff:'); 
            E2_coeff = E2_coeff0;
            E1_coeff = E1_coeff + E1_coeff_delta;
            E1_thresh = E1_mean + E1_coeff*E1_std; %E1_mean_max; %E1_mean;
        end
    end
%     T
%     E2_coeff
%     E2_subord_thresh
    iter = iter+1;
    if(iter == max_iter)
        task_complete = 1;
        disp('Max Iter reached, check reward rate / baseline data'); 
    end
end

%%
h = figure;
plot(T_vec, '.-', 'MarkerSize', 7); 
xlabel('alg iteration'); 
ylabel('target'); 
title('Target Value over Calibration'); 
saveas(h, fullfile(plotPath, 'target_val_over_calibration.png')); 

% h = figure;
% hold on; 
% plot(reward_per_frame_vec, '.-', 'MarkerSize', 7); 
% hline(reward_per_frame_range(1)); 
% hline(reward_per_frame_range(2)); 
% xlabel('alg iteration'); 
% ylabel('reward per Frame'); 
% title('Reward Per Frame over Calibration'); 
% saveas(h, fullfile(save_dir, 'reward_per_frame_over_calibration.png')); 

%%
%Summary results of cal: 
disp('T'); 
T

% cursor_obs = n_analyze*decoder; 
% c1 = find(cursor_obs >= T); 
disp('num E2-E1 >= T'); 
num_c1 = length(c1)

% E1_mean_analyze = mean(E1_analyze,2)
% c2 = find(E1_mean_analyze <= E1_thresh);
disp('E1 >= b'); 
num_c2 = length(c2)

% E2_mean_analyze = mean(E2_analyze,2); 
% [E2_dom_samples, E2_dom_sel] = max(E2_analyze, [], 2);
% E2_subord_mean_analyze = (E2_sum_analyze - E2_dom_samples)/(num_E2-1);
% %For each idx, subtract the 
% c3 = find(E2_subord_mean_analyze >= E2_subord_thresh(E2_dom_sel)); 
disp('E2 subord >= c'); 
num_c3 = length(c3)

num_cursor_hits = length(c1); 
disp('num cursor target hits (wo E1<thr, E2sub>thr :'); 
num_cursor_hits

disp('num baseline hits WITHOUT B2BASE:'); 
num_hits_no_b2base = length(hit_idxs_no_b2base)

disp('num valid hits (WITH B2BASE):'); 
num_valid_hits = length(valid_hit_idxs)

% b2base_thresh = 0.5*T;
% hits_valid = ones(length(hit_times),1); 
% if length(hit_times) > 1
%     for i = 2:length(hit_times)
%         b2base_bool = sum(cursor_obs(hit_times(i-1):hit_times(i)) <= b2base_thresh) > 1; 
%         hits_valid(i) = b2base_bool; 
%     end
% end
% disp('num baseline hits WITH B2BASE:'); 
% num_valid_hits = sum(hits_valid)

% valid_hit_times = hit_times(find(hits_valid)); 


cursor_amp = (max(cursor_obs)-min(cursor_obs));
cursor_offset = cursor_amp/10; 
max_cursor = max(cursor_obs); 

%%
% Calculate parameters for auditory feedback
cursor_target   = T;
[fb_cal]        = cursor2audio_fb(cursor_obs, fb_settings, cursor_target);

%%
%PLOTS
%TODO: decomposition into plotting functions
%--------------------------------------------------------------------------
%%
%Plot auditory feedback
% plot_cursor = linspace(cal.fb.cursor_min, cal.fb.cursor_max, 1000); 
plot_cursor = linspace(min(cursor_obs), max(cursor_obs), 1000); 
plot_freq   = cursor2audio_freq_v2(plot_cursor, fb_cal);
h = figure;
plot(plot_cursor, plot_freq); 
xlabel('Cursor E2-E1'); 
ylabel('Audiory Freq'); 
vline(cursor_target); 
saveas(h, fullfile(plotPath, 'cursor2freq.png')); 

%%
fb_obs = cursor2audio_freq_v2(cursor_obs, fb_cal); % cursor2audio_freq(cursor_obs, cal);
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
plot(E1_mean_analyze-cursor_amp); 
plot(E2_subord_mean_analyze-2*cursor_amp); 
xlabel('frame'); 
title(['hits with b2base: ' num2str(num_valid_hits)]); 
legend({'c1', 'c2 - E1 cond', 'c3 - E2 cond', 'cursor', 'E1 mean', 'E2 subord mean'}); 
vline(valid_hit_idxs); 
saveas(h, fullfile(plotPath, 'cursor_hit_ts.png')); 

%%
offset = 0; 
[h, offset_vec] = plot_cursor_E1_E2_activity(cursor_obs, E1_mean_analyze, E2_mean_analyze, n_analyze, E_id, E_color, offset)
hold on; hline(T); 
saveas(h, fullfile(plotPath, 'cursor_E1_E2_ts.png')); 
%%
cursor_obs = n_analyze*decoder; 
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
% [h, offset_vec] = plot_E_activity(n_analyze, E_id, E_color);
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
% plot(1:length(E1_mean_analyze), E1_mean_analyze-c2_offset);
% % hline(E1_thresh-c2_offset)
% 
% %c3:
% c3_offset = offset_vec(end)+3*offset;
% plot(E2_subord_mean_analyze-c3_offset); 
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
[psth_mean, psth_sem, psth_mat] = calc_psth(n_analyze, valid_hit_idxs, psth_win);
h = figure; hold on;
offset = 0; 
for i=1:num_neurons
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

function [decoder, E1_proj, E2_proj, E1_norm, E2_norm] = ...
    def_decoder(num_neurons, E1_sel, E2_sel)

E1_proj = zeros(num_neurons, 1); 
E1_proj(E1_sel) = 1;
E1_norm = sum(E1_sel); %can replace with vector norm.  
disp('E1 proj'); 
E1_proj = E1_proj/E1_norm;
E1_proj

E2_proj = zeros(num_neurons, 1); 
E2_proj(E2_sel) = 1; 
E2_norm = sum(E2_sel); 
disp('E2 proj')
E2_proj = E2_proj/E2_norm;
E2_proj

disp('decoder:')
decoder = E2_proj - E1_proj
end

function [fb_cal] = cursor2audio_fb(cursor_obs, fb_settings, cursor_target)
%INPUT: 
%cursor_obs
%cal
%FIELDS: 
% -- target.T
% -- fb.freq_min
% -- fb.freq_max

%Assumes cursor = E2-E1, and cursor_target is positive.
%Maps cursor to auditory feedback.
% freq = a*exp(b*(cursor_trunc-cursor_min))

%Copy from task_settings:
fb_cal.settings = fb_settings; 
% fb_settings.target_low_freq        = 1; % %Set the target cursor value to be the low frequency
% fb_settings.freq_min               = 6000; 
% fb_settings.freq_max               = 19000; 
% fb_settings.arduino.com            = 'COM11';
% fb_settings.arduino.label          = 'Mega2560';
% fb_settings.arduino.pin            = 'D3';
% fb_settings.arduino.duration       = 0.3; %ms, tones update at rate of BMI code, this is the longest a tone will play for
% fb_settings.min_perctile            = 10;

%Calculate:
fb_cal.cursor_min         = ...
    prctile(cursor_obs, fb_cal.settings.min_perctile); 
%min(cursor_obs); %for fb, cursor is ceil to this value
fb_cal.cursor_max         = ...
    cursor_target; %for fb, cursor is floor to this value
fb_cal.cursor_range       = ...
    fb_cal.cursor_max - fb_cal.cursor_min; 
% % freq = a*exp(b*(cursor_trunc-cursor_min))
fb_cal.a                  = ...
    fb_cal.settings.freq_min; 
fb_cal.b                  = ...
    (log(fb_cal.settings.freq_max) - log(fb_cal.a))/fb_cal.cursor_range; 
end

%%
%ToDo: re-add target_on_cov_bool functionality:
% if(target_on_cov_bool)
%     %Generate samples
%     mvn_samples = mvnrnd(analyze_mean,analyze_cov,rand_num_samples);
% 
%     %Check probability of success condition
%     cursor_samples = mvn_samples*decoder; 
%     E1_samples = mvn_samples(:,E1_sel); 
%     E2_samples = mvn_samples(:,E2_sel); 
%     E1_mean_samples = mean(E1_samples,2); 
%     E2_mean_samples = mean(E2_samples,2); 
% 
%     %1) E2-E1 > alpha
%     c1 = find(cursor_samples>= T); 
% 
%     %2) E1 < mu
%     c2 = find(E1_mean_samples <= E1_thresh); 
% 
%     %3) E2_subord >= mu (anded with previous constraint)
%     [E2_dom_samples, E2_dom_sel_samples] = max(E2_samples, [], 2);
%     E2_subord_sum_samples = E2_mean_samples*num_E2 - E2_dom_samples;
%     E2_subord_mean_samples = E2_subord_sum_samples/(num_E2-1); 
%     %For each idx, subtract the 
%     c3 = find(E2_subord_mean_samples >= E2_subord_thresh(E2_dom_sel_samples)); 
%     %E2_subord_mean(E2_dom_sel_samples)); 
% 
%     %Intersect the indices:
%     hits = intersect(intersect(c1, c2), c3); 
%     reward_prob_per_frame = length(hits)/rand_num_samples; 