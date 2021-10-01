function [psth_mean, psth_sem, psth_mat] = calc_psth(data_mat, event_idxs, win)
%Return mean and stderr at each point
%data_mat: assumed to be (num_samples X num_var)
%event_idxs: samples to lock to
%win: number of samples before and after to use for psth

%in current implementation, a time series will be neglected if the window
%is larger than the data supports.

num_events = length(event_idxs); 
num_samples = size(data_mat, 1); 
num_var = size(data_mat, 2); 

psth_len = win(2)-win(1)+1; 
psth_mat = zeros(floor(psth_len), num_var, num_events); 

event_valid = ones(num_events, 1); 
for event_i = 1:num_events
    event_sel = (win(1):win(2))+event_idxs(event_i);
    if(min(event_sel) < 1 || max(event_sel) > num_samples)
        event_valid(event_i) = 0; 
    else
        %event_sel
        psth_mat(:,:,event_i) = data_mat(event_sel,:); 
    end
end
num_valid_events = sum(event_valid); 
%Delete entries for invalid data: 
psth_mat(:,:,find(~event_valid))=[]; 

%Average over events: 
psth_mean = mean(psth_mat, 3); 
psth_var = var(psth_mat, 0, 3); 
psth_std =  psth_var.^(1/2); 
psth_sem = psth_std/sqrt(num_valid_events); 