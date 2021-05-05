function out = BMIAcqnvs_v2(vals,varargin)

%{
    Scanimage needs a linear function script style to run between frames.
    For this reason the code needs to be executed line by line. Besides
    each frame the function gets executed and returns 0, which means,
    variables that are not persistent will be lost. This function needs to
    be as light as possible. Scanimage will not execute the function if it
    does not have time enough for it. Avoid large computations, saving to
    memory, large matrices copying and anything that requires computational
    power, and therefore time, from the CPU

    Inputs: vals is the activity of the neurons selected on scanimage

    expt_str --> buffer:
    0) normal_bmi
    flagBMI = true; (use self-generated hits to give waterer reward, no stim)
    flagwater = true; flagstim = False

    0) dstim_bmi
    flagBMI = true; (use self-generated hits to send D1/D2 stim)
    flagwater = false; flagstim = true;
    
    3) block_bmi
    flagBMI = true; (use self-generated hits to give water reward block D1/D2)
    flagwater = true; flagstim = true;
%}

    %% To initialize the recording:
    % this is required to clean all the persistent variables if need be
    flush =0; 

    % this is only to initialize. History keeps most of the persisten vars
    persistent history
    
    if flush 
        vars = whos;
        vars = vars([vars.persistent]);
        varName = {vars.name};
        clear(varName{:});
        out = []; 
        clear a;
        clear history;
        disp('flushiiiing')
        history.experiment = [];
        return
    end
    
    
    %%
    %**********************************************************
    %****************  PARAMETERS  ****************************
    %**********************************************************
    
    %% BMI parameters
    % parameters for the function that do not change (and we may need in each
    % iteration. 
    task_settings = evalin('base','task_settings');
    out = 0;       %required output to ScanImage
    init_frame_base = task_settings.params.initial_count + 1;
    
    %% needed variables
    % Define persistent variables that will remain in each iteration of this
    % function
    persistent a flags counters
    global data
    
    %% experiment FLAGS
    switch lower(task_settings.exptStr) %make it case insensitive. all lower cases
        case 'normal_bmi'
            flags.BMI = true;
            flags.water = true;
            flags.stim = false;
        case 'dstim_bmi'
            flags.BMI = true;
            flags.water = false;
            flags.stim = true;
        case 'block_bmi'
            flags.BMI = true;
            flags.water = true;
            flags.stim = true;
    end   
    
   %%
    %*********************************************************************
    %******************  INITIALIZE  ***********************************
    %*********************************************************************
    
    % initialize arduino for playing sound (if not initialized already).
    if ~isa(a, 'arduino')
        a = arduino('COM15', 'Uno');
        disp('starting arduino')
    end
      
    if isempty(history.experiment) %if this is the first time it runs this program (after flush)
        % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %global parameters hSI;
        counters.do_not_update_buffer = task_settings.params.initial_count;
        counters.miss = 0;
        counters.after_motion = 0;
        counters.trial = 1;
        counters.back_2_base = 0;
        counters.baseline_full = task_settings.base_frames;
        flags.back_2_baseline = false;
        flags.base_buffer_full = false;
        flags.init_baseline = true;
        flags.motion = false;
        data.cursor = single(nan(1, task_settings.expected_length_experiment));  %define a very long vector for cursor
        data.frequency = single(nan(1, task_settings.expected_length_experiment));
        assignin('base','data.cursor', data.cursor);
        assignin('base','data.frequency', data.frequency);
        assignin('base','data.hits', []);
        assignin('base','data.miss', []);
        assignin('base','data.stims', []);
        assignin('base','data.trialEnd', []);
        assignin('base','data.trialStart', []);
        history.baseval = single(ones(task_settings.units,1).*vals);
        history.buffer = single(nan(units, task_settings.moving_average_frames));  %define a windows buffer
        history.index = 1 ;
        history.last_volume = 0;  %careful with this it may create problems
        history.nZ=evalin('base','hSI.hFastZ.numFramesPerVolume');
        history.number_rewards = 0; 
        history.number_trials = 0;
        history.number_stims = 0;
        history.tim = tic;
    end

    
     %% Prepare the stim !!!TODO!!!
 
        
    %%
    %************************************************************************
    %*************************** RUN ********************************
    %************************************************************************
    
    % acquire the actual frame
    this_frame = evalin('base','hSI.hScan2D.hAcq.hFpga.AcqStatusAcquiredFrames');
    this_volume = floor(this_frame/nZ);
    %if we've completed a new volume, update history 
    % store nans on frames that we've skipped so we know we skipped
    % them
    if this_volume > history.last_volume  % if this is a new volume

        % handle ******* MOTION***********
        % because we don't want to stim or reward or update buffer if there is motion
        mot = evalin('base', 'hSI.hMotionManager.motionCorrectionVector');
        if ~isempty(mot)
            motion = sqrt(mot(1)^2 + mot(2)^2 + mot(3)^2); 
        else
            motion = 0;
        end

        if motion > task_settings.params.motion_thresh 
            flags.motion = true; 
            counters.after_motion = 0;
        else
            if counters.after_motion >= task_settings.motion_relaxation_frames
                flags.motion = false;
            else
                counters.after_motion = counters.motion + 1;
            end
        end
        

        if counters.do_not_update_buffer == 0 && ~flags.motion
            %update frame
            steps = this_volume - history.last_volume;
            history.last_volume = this_volume;
            history.index = history.index + steps;
            assignin('base','duration', history.index);
            % variable to hold nans in unseen frames
            placeholder = nan(numel(vals),steps-1);
            mVals = [placeholder vals];

            % update buffer of activity history
            if steps < task_settings.moving_average_frames
                history.buffer(:, 1: end-steps) = history.buffer(:, steps+1:end);
                history.buffer(:,end-steps+1:end) = mVals;    
            else
                history.buffer = mVals(:, end-task_settings.moving_average_frames+1:end);
            end
            
            signal = single(nanmean(history.experiment, 2));
            
            % update dynamic baseline. baseline may be seeded if another BMI was run before 
            if flags.init_baseline && ~isnan(sum(task_settings.base_val_seed))
                flags.base_buffer_full = true; 
                history.baseval = task_settings.base_val_seed; 
                disp('baseBuffer seeded!'); 
            elseif ~ flags.base_buffer_full && counters.baseline_full > 0
                history.baseval = (history.baseval*(single(history.index) - 1) + signal)./single(history.index);
                counters.baseline_full = counters.baseline_full - 1;
                if counters.baseline_full == 0
                    disp('baseline full')
                    flags.base_buffer_full = true;
                end
            else
                history.baseval = (history.baseval*(task_settings.base_frames - 1) + signal)./task_settings.base_frames;
            end
            
            % baseline has been initiated one way or another
            if flags.init_baseline
                flags.init_baseline = false;
            end
            
            if flags.base_buffer_full   % only after it finishes with baseline it will start
                 % calculation of DFF
                dff = (signal - history.baseval) ./ history.baseval;
                data.cursor(history.index) = nansum([nansum(dff(task_settings.E1)),-nansum(dff(task_settings.E2))]);
                
                % obtain frequency
                if task_settings.mice_settings.target_low
                    %This means cursor up makes auditory freq go down:
                    aux_cursor  = - data.cursor(history.index); 
                    aux_cursor_min  = - task_settings.fb_cal.cursor_max;
                    aux_cursor_max  = - task_settings.fb_cal.cursor_min;
                else
                    %This means cursor up makes auditory freq go up:
                    aux_cursor = data.cursor(history.index);
                    aux_cursor_min  = task_settings.fb_cal.cursor_min;
                    aux_cursor_max  = task_settings.fb_cal.cursor_max;
                end

                %%
                cursor_trunc    = max(aux_cursor, aux_cursor_min); 
                cursor_trunc    = min(cursor_trunc, aux_cursor_max); 
                freq = task_settings.fb_cal.a*exp(task_settings.fb_cal.b*(cursor_trunc-aux_cursor_min));
                
                data.frequency(history.index) = freq;
                


            
                     %Passing smoothed dff to "decoder"
                    [~, cursor_i, target_hit, c1_bool, c2_val, c2_bool, c3_val, c3_bool] = ...
                        dff2cursor_target(dff, bData, cursor_zscore_bool);
%                     data.bmidffz(:,data.frame) = dff_z;
                    data.cursor(data.frame) = cursor_i;
                    m.Data.cursor(data.frame) = data.cursor(data.frame); % saving in memmap
                    if debug_bool
                        data.dff(:,data.frame)      = dff;
                        data.c1_bool(data.frame)    = c1_bool; 
                        data.c2_val(data.frame)     = c2_val;
                        data.c2_bool(data.frame)    = c2_bool;
                        data.c3_val(data.frame)     = c3_val;
                        data.c3_bool(data.frame)    = c3_bool;
                    end
                    
                    disp(['Cursor: ' num2str(cursor_i)]); 
%                     disp(['Target : ' num2str(target_hit)]); 
%                     disp(['C1 - cursor: ' num2str(c1_bool)]); 
%                     disp(['C2 - E1 : ' num2str(c2_bool)]); 
%                     disp(['C3 - E2 subord : ' num2str(c3_bool)]);                     
                    % c1: cursor
                    % c2: E1_mean > E1_thresh
                    % c3: E2_subord_mean > E2_subord_thresh                    
                    %----------------------------------------------------------
                end
                
                if (BufferUpdateCounter == 0) && base_buffer_full
%                     disp('HERE'); 
                    % Is it a new trial?
                    if trialFlag && ~back_2_baselineFlag
                        data.trialStart(data.frame) = 1;
                        m.Data.trialStart(data.frame) = 1;
                        data.trialCounter = data.trialCounter + 1;
                        trialFlag = 0;
                        %start running the timer again
                        disp('New Trial!')
                    end

                    if back_2_baselineFlag 
                        if data.cursor(data.frame) <= back_2_base 
                            back_2_baseCounter = back_2_baseCounter+1;

                        end
                        if back_2_baseCounter >= back_2_baseFrameThresh
                            back_2_baselineFlag = 0;
                            back_2_baseCounter = 0;
                            disp('back to baseline')
                        end
                    else
%                         disp('HERE2'); 
                        if target_hit      %if it hit the target
                            disp('target hit')
                            if(HoloTargetDelayTimer > 0)
                                disp('Holo Target Achieved')
                                HoloTargetDelayTimer = 0; 
                                data.holoTargetCounter = data.holoTargetCounter + 1;
                                data.holoHits(data.frame) = 1;
                                m.Data.holoHits(data.frame) = 1;
                                
                                if flagVTAtrig
                                    disp('RewardTone delivery!')
                                    if(~debug_bool)
%                                         play(reward_sound);
%                                         outputSingleScan(s,ni_reward); pause(0.001); outputSingleScan(s,ni_out)
                                    end
                                    rewardDelayCounter = rewardDelayFrames; 
                                    deliver_reward = 1;                                     
                                    nonBufferUpdateCounter = shutterVTA;   
                                    
                                    data.holoTargetVTACounter = data.holoTargetVTACounter+1;
                                    data.holoVTA(data.frame) = 1;
                                    m.Data.holoVTA(data.frame) = 1;
                                end
                                
                                %Back to baseline, and new trial
                                BufferUpdateCounter = relaxationFrames; 
                                back_2_baselineFlag = 1;
                                disp(['Trial: ', num2str(data.trialCounter), 'VTA stimS: ', num2str(data.holoTargetVTACounter + data.selfTargetVTACounter)]);
                                % update trials and hits vector
                                trialFlag = 1; 
                            else
                                %Self hit:
                                data.selfTargetCounter = data.selfTargetCounter + 1;
                                data.selfHits(data.frame) = 1;
                                m.Data.selfHits(data.frame) =1;
                                disp('self hit')
                                if(flagBMI)
                                    if(flagVTAtrig)
                                        deliver_reward = 1;
                                        data.selfTargetVTACounter = data.selfTargetVTACounter + 1;
                                        data.selfVTA(data.frame) = 1;
                                        m.Data.selfVTA(data.frame) = 1;
                                        disp(['Trial: ', num2str(data.trialCounter), 'VTA stimS: ', num2str(data.holoTargetVTACounter + data.selfTargetVTACounter)]);
                                    else
                                        disp(['Trial: ', num2str(data.trialCounter), 'Num Self Hits: ', num2str(data.selfTargetCounter)]); 
                                    end
                                    
                                    nonBufferUpdateCounter = shutterVTA;
                                    disp('Target Achieved! (self-target)')
                                    disp('RewardTone delivery!')
                                    if ~debug_bool
%                                         play(reward_sound);
                                    end                                        
                                    rewardDelayCounter = rewardDelayFrames; 
                                    BufferUpdateCounter = relaxationFrames; 
                                    back_2_baselineFlag = 1;
                                    % update trials and hits vector
                                    trialFlag = 1;                                    
                                else
                                    disp(['Num Self Hits: ', num2str(data.selfTargetCounter)]); 
                                end
                            end
                        end
                        if ~trialFlag
%                             disp(['HERE ' num2str(data.frame)]); 
                            if flagHolosched
                                if ismember(data.frame, data.vectorHoloCL)
                                    disp('SCHEDULED HOLO stim'); 
                                    currHoloIdx = find(data.vectorHoloCL == data.frame);                                                                   
                                        %Check E1, if lower than threshold, do
                                        %stim, and save frame
                                    if(c2_bool)                                    
                                        disp('HOLO stim')
                                        HoloTargetDelayTimer = HoloTargetWin;
                                        data.schedHoloCounter = data.schedHoloCounter + 1;
                                        if(~debug_bool)
                                            outputSingleScan(s,ni_holo); pause(0.001); outputSingleScan(s,ni_out)                                    
                                        end
                                        %Also, save the frame we do this!!
                                        data.holoDelivery(data.frame) = 1;
                                    else
                                        data.vectorHoloCL(currHoloIdx:end) = data.vectorHoloCL(currHoloIdx:end)+1;
                                    end
                                end
                            elseif flagVTAsched
                                if ismember(data.frame, vectorVTA)
                                    disp('scheduled VTA stim')
                                    disp('RewardTone delivered!'); 
                                    if(~debug_bool)
%                                         play(reward_sound); 
%                                         outputSingleScan(s,ni_reward); pause(0.001); outputSingleScan(s,ni_out)
                                    end
                                    rewardDelayCounter = rewardDelayFrames; 
                                    deliver_reward = 1; 

                                    nonBufferUpdateCounter = shutterVTA;                                
                                    data.schedVTACounter = data.schedVTACounter + 1; 
                                end
                            end
                        end
                            
                    end
                else
                    if(BufferUpdateCounter>0)
                        BufferUpdateCounter = BufferUpdateCounter - 1;
                    end
                end
            else
                if(nonBufferUpdateCounter>0)
                    nonBufferUpdateCounter = nonBufferUpdateCounter - 1;
                end
            end
            
            if(HoloTargetDelayTimer > 0)
                HoloTargetDelayTimer = HoloTargetDelayTimer-1;
            end
            
            if(rewardDelayCounter > 0)
                rewardDelayCounter = rewardDelayCounter -1; 
            elseif(deliver_reward && rewardDelayCounter==0)
                if(~debug_bool && ~strcmp(expt_str, 'BMI_no_reward'))
                    outputSingleScan(s,ni_reward); pause(0.001); outputSingleScan(s,ni_out);
                end
                deliver_reward = 0; 
                disp('reward delivered!'); 
            end
                
            data.frame = data.frame + 1;
            data.timeVector(data.frame) = toc;
            counterSame = 0;
            if (~debug_bool && data.timeVector(data.frame) < 1/(frameRate*1.2))
                pause(1/(frameRate*1.2) - data.timeVector(data.frame))
            end
        else
            counterSame = counterSame + 1;
        end
    end
%    pl.Disconnect();
%     save(fullfile(savePath, ['BMI_online', datestr(datetime('now'), 'yymmddTHHMMSS'), '.mat']), 'data', 'bData')
end
% 

