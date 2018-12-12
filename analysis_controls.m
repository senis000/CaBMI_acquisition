if ~exist('Exp.FPM','var')
    Exp.FPS = 15.2;
    Exp.FPM = Exp.FPS*60;
end


%%  Register correlation and distnace of online ROIs and offline ROIs
score = zeros(size(OnlineData,2),size(signals.Fraw,1));
dist = zeros(size(OnlineData,2),size(signals.Fraw,1));
S2P_ID = zeros(size(OnlineData,2), 1);
for k = 1:size(OnlineData,2)  %for each online ROI
    SourceData=OnlineData(:,k);  %pull the online recorded F values 
    for m = 1:size(signals.Fraw,1)  %for each offline Suite2P ROI
        S2PData=signals.Fraw(m,onlineFrames)';  %extract the same frames that were recorded on line (some are skipped)
        score(k,m) = corr(S2PData,SourceData);  %score of k,m is the correlation between the online and offline ROI
        dist(k,m)=sqrt((loc(m,1)-(centerXY(k,1)-minx))^2  +   (loc(m,2)-(centerXY(k,2)-miny))^2);  %dist k,m is the euclidean distance between the online/offline roi
    end
end

for k = 1:size(OnlineData,2)  %for each online ROI
    [dist_sort, DistIndx]=sort(dist(k,:),'Ascend');  %find the S2P roi that is closest
    [score_sort, ScoreIndx]=sort(score(k,:),'Descend');%and most similiar 
    
    if DistIndx(1) == ScoreIndx(1)  % if they're the same
        S2P_ID(k)=ScoreIndx(1);   % then we have successfully registered them
    else
        candidates = find(dist_sort<20);
        if isempty(candidates)
            disp(['could not ID cell ' num2str(k)])
            S2P_ID(k)=nan;
        elseif numel(candidates)>1
            figure();
            plot(onlineFrames,OnlineData(:,k)); hold on;
            for jj = 1:numel(candidates)
            plot(onlineFrames,signals.Fraw(DistIndx(candidates(jj)),onlineFrames));
            end
            int=input('which is best match?');
            S2P_ID(k)=DistIndx(int);
        else
            S2P_ID(k)=DistIndx(candidates);
        end
    end       
     
end

%% Figure Out Trial IDs 
[stimID, uniqueStims, uStimCount, stims] = stimReader(ExpStruct);  %stimreader pulls the file IDs the correposond to baseline and post trials

baselineTrialID = mode(stims(1:40));
PostTrialID= mode(stims(1:40)); 

%find file starts that coorespond to baseline and post
nonStimTrials = find(stims == baselineTrialID);
BaselineTrials=nonStimTrials(nonStimTrials<numel(stims)/2);
PostTrials=nonStimTrials(nonStimTrials>numel(stims)/2);
HoloStimTrials = find(stims>PostTrialID);

TrialOffset=BaselineTrials(1);  %which DAQ sweeps cooresponds to the first tiff file
 
BaselineTrials=BaselineTrials-TrialOffset+1;
PostTrials=PostTrials-TrialOffset+1;
HoloStimTrials =HoloStimTrials-TrialOffset+1;
%% Compute Basline Pairwise Corr
%plot entire experiment
BL_frames=fileStart(BaselineTrials(1)):fileStart(BaselineTrials(length(BaselineTrials+1)))-1; %end 1 frame before first holostim trial
Post_frames=fileStart(PostTrials(1)):size(signals.Fraw,2);

%get reward frames
rewardFrames =find(vtaStimHistory);
%find and delete duplicaes (hopefully obselete after this pone)
% indxDel=find(diff(rewardFrames)<40);
% rewardFrames(indxDel+1)=[];

figure();
v=[E1 E2(2:end)];%duplicate
for n = 1:numel(v) 
    if n>4
        plot((1:size(signals.F,2))/Exp.FPM,signals.F(S2P_ID(v(n)),:)-(n*7),'k');
    else
        plot((1:size(signals.F,2))/Exp.FPM,signals.F(S2P_ID(v(n)),:)-(n*7),'r');
    end
    hold on
end

scatter(rewardFrames/Exp.FPM, ones(size(rewardFrames))+10,120,'.b')
plot([BL_frames(1) BL_frames(end)]/Exp.FPM,[15 15],'k','LineWidth',5);
plot([Post_frames(1) Post_frames(end)]/Exp.FPM,[15 15],'m','LineWidth',5);
scatter(fileStart(HoloStimTrials)/Exp.FPM,ones(size(HoloStimTrials))+12,120,'.c')


%This graph shows the entire experiment.   
%red traces = E1
%black traces = E2
%black bar baseline, magenta bar post
%blue dots = vta reward
%cyan dots, hologstim times
%% compute pairwise pearsons p for all cells during baseline

% allocate memory
BlCorrs = zeros(size([E1 E2],2), size([E1 E2],2));
PostCorrs = zeros(size([E1 E2],2), size([E1 E2],2));

for c1=1:size([E1 E2],2)
    for c2=1:size([E1 E2],2)
        if c2>=c1
            BlCorrs(c1,c2)=nan;
        else
            BlCorrs(c1,c2)=corr(signals.Fraw(S2P_ID(c1),BL_frames)',signals.Fraw(S2P_ID(c2),BL_frames)');
        end
        if BlCorrs(c1,c2)>.99  %it may be the same cell
            BlCorrs(c1,c2)=nan;
        end
    end
end

%compute pairwise pearsons p for all cells during post
for c1=1:size([E1 E2],2)
    for c2=1:size([E1 E2],2)
        if c2>=c1
            PostCorrs(c1,c2)=nan;
        else
            PostCorrs(c1,c2)=corr(signals.Fraw(S2P_ID(c1),Post_frames)',signals.Fraw(S2P_ID(c2),Post_frames)');
        end
        if PostCorrs(c1,c2)>.999;
            PostCorrs(c1,c2)=nan;
        end
    end
end
 
subplot(1,2,1);
plotSpread({BlCorrs(E1,E1),PostCorrs(E1,E1)},'showMM',4);
title('E1 pairwise correlations baseline - post');
ylim([-.1 .2])
ylabel('pairwise correlation');
subplot(1,2,2);
plotSpread({BlCorrs(E2,E2),PostCorrs(E2,E2)},'showMM',4);
title('E2 pairwise correlations baseline - post');
ylim([-.1 .2])
ylabel('pairwise correlation');

preCorrsE1=BlCorrs(E1,E1);
preCorrsE1(isnan(preCorrsE1))=[];
postCorrsE1=PostCorrs(E1,E1);
postCorrsE1(isnan(postCorrsE1))=[];
 
[~, p] = ttest2(preCorrsE1,postCorrsE1);

subplot(1,2,1);
text(2, 0.18, ['p = ', num2str(p)])
 
preCorrsE2=BlCorrs(E2,E2);
preCorrsE2(isnan(preCorrsE2))=[];
postCorrsE2=PostCorrs(E2,E2);
postCorrsE2(isnan(postCorrsE2))=[];
 
[~, p] = ttest2(preCorrsE2,postCorrsE2);
subplot(1,2,2);
text(2, 0.18, ['p = ', num2str(p)])

%% Check Efficacy of Holographic Stimulation
clear HoloRecord
prepost = [-30, 80];
HoloRecord = zeros(numel(HoloStimTrials),prepost(2)-prepost(1)+1,numel(S2P_ID));
for k = 1:numel(HoloStimTrials)
    fVec=fileStart(HoloStimTrials(k))+prepost(1):fileStart(HoloStimTrials(k))+prepost(2);
    for c=1:numel(S2P_ID)
        HoloRecord(k,:,c)=signals.spikes(S2P_ID(c),fVec);
    end
end
%% Show E1 Response locked to holostim
figure()
for j=1:numel(E1)
    time_sec=(1:size(HoloRecord,2))/Exp.FPS;
    trace=squeeze(HoloRecord(:,:,E1(j)));
    fillPlot(trace-(j*7),time_sec','ci');
    hold on
end
xlim([0 4]);
ylabel('Seconds')
xlabel('OASIS Deconvolution')

hold on
plot([29/Exp.FPS 35/Exp.FPS],[0 0],'r','LineWidth',4);
% plot([29/Exp.FPS 29/Exp.FPS],[-0.5 2.9],'--r','LineWidth',1); grid on;

figure()
for j = 1:numel(E1)
    subplot(2,2,j);
    imagesc(HoloRecord(:,:,E1(j)));  caxis([-1 3]);
    ylabel('Trial Number');
    xlabel('Frames');
end

%% This just plots rewards and holostims to see if we can see rewards 
figure()
t=ones(size((rewardFrames)));

scatter(rewardFrames', t+10,50,'.m')
hold on
T=ones(size(fileStart(HoloStimTrials)));
scatter(fileStart(HoloStimTrials),T+10,150,'.k')
%% Find rewards and associate them with either holography or VTA

VTAreal = rewardFrames;
VTAholo = rewardFrames;
k1=1;k2=1;ToDelreal=[];ToDelholo=[];
for ind=1:size(rewardFrames,1)
    auxHolo = rewardFrames(ind) - fileStart(HoloStimTrials) ;
    auxHolo(auxHolo<0)=[];
    if  min(auxHolo) < 40
        ToDelreal(k1)=ind;
        k1=k1+1;
    else
        ToDelholo(k2)=ind;
        k2=k2+1;
    end
end
VTAreal(ToDelreal)=[];
VTAholo(ToDelholo)=[];

endbl = BL_frames(end);
VTAreal = VTAreal(VTAreal>endbl);

figure();  %show emergence of reward over time
VTAreal_sec = (VTAreal/(Exp.FPS*60));
ecdf(VTAreal_sec)
k=xlim;    
xlim([0 k(2)]);
xlabel('minutes')
ylabel('Cum Volitional Rewards');
hold on;
VTAholo_sec = (VTAholo/(Exp.FPS*60));
ecdf(VTAholo_sec)
xlabel('Minutes');
ylabel('Cumulative Distribution');
legend('Volitional Reactivation','Holographic Activation')
%% show E1 response locked to rewards (not to holostims)
clear HoloRecord
prepost = [-30, 80];
HoloRecord = zeros(numel(HoloStimTrials),prepost(2)-prepost(1)+1,numel(S2P_ID));
for k = 1:numel(VTAholo)
    fVec=VTAholo(k)+ prepost(1):VTAholo(k)+prepost(2);
    for c=1:numel(S2P_ID)
        HoloRecord(k,:,c)=signals.F(S2P_ID(c),fVec);
    end
end
figure()
for j=1:numel(E1)
    time_sec=(1:size(HoloRecord,2))/Exp.FPS;
    trace=squeeze(HoloRecord(:,:,E1(j)));
    fillPlot(trace-(j*7),time_sec','ci');
    hold on
end
xlim([0 6]);
xlabel('Seconds')
ylabel('Z-Scored Fluor')

hold on
plot([29/Exp.FPS 35/Exp.FPS],[0 0],'r','LineWidth',4);
% plot([29/Exp.FPS 29/Exp.FPS],[-0.5 2.9],'--r','LineWidth',1); grid on;

figure()
for j = 1:numel(E1)
    subplot(2,2,j);
    imagesc(HoloRecord(:,:,E1(j)));  caxis([-1 3]);
    ylabel('Trial Number');
    xlabel('Frames');
end
%% Show E2 Ensemble in response to HoloStim
figure()
for j=1:numel(E2)
    time_sec=(1:size(HoloRecord,2))/Exp.FPS;
    trace=squeeze(HoloRecord(:,:,E2(j)));
    fillPlot(trace-(j*7),time_sec','ci');
    hold on
end
xlim([0 6]);
xlabel('Seconds')
ylabel('Z-Scored Fluo')

hold on
plot([29/Exp.FPS 35/Exp.FPS],[0 0],'r','LineWidth',4);
% plot([29/Exp.FPS 29/Exp.FPS],[-0.5 2.9],'--r','LineWidth',1); grid on;

figure()
for j = 1:numel(E2)
    subplot(2,2,j);
    imagesc(HoloRecord(:,:,E2(j)));  caxis([-1 3]);
    ylabel('Trial Number');
    xlabel('Frames');
end

%% This is the important part of the code  show spont reactiations
clear HoloRecord ylim kE1
prepost = [-30, 30];
HoloRecord = zeros(numel(HoloStimTrials),prepost(2)-prepost(1)+1,numel(S2P_ID));
for k = 1:numel(VTAreal)
    fVec=VTAreal(k)+prepost(1):VTAreal(k)+prepost(2);
    for c=1:numel(S2P_ID)
        HoloRecord(k,:,c)=signals.F(S2P_ID(c),fVec);
    end
end

figure()
for j=1:numel(E2)
    time_sec=(1:size(HoloRecord,2))/Exp.FPS;
    trace=squeeze(HoloRecord(:,:,E1(j)));
    fillPlot(trace,time_sec','ci',[],[],[],.2);
    hold on
end
hold on
kE1=ylim;
plot([29/Exp.FPS 29/Exp.FPS],kE1,'--r','LineWidth',1); grid on;

figure()
for j = 1:numel(E1)
    subplot(2,2,j);
    imagesc(zscore(HoloRecord(:,:,E1(j)),[],2));  caxis([-2 2]);
    ylabel('trials');
    xlabel('frames')
end

%% and for E2

clear HoloRecord
prepost = [-30, 30];
HoloRecord = zeros(numel(HoloStimTrials),prepost(2)-prepost(1)+1,numel(S2P_ID));
for k = 1:numel(VTAreal)
    fVec=VTAreal(k)+prepost(1):VTAreal(k)+prepost(2);
    for c=1:numel(S2P_ID)
        HoloRecord(k,:,c)=signals.F(S2P_ID(c),fVec);
    end
end

figure()
for j=1:numel(E2)
    time_sec=(1:size(HoloRecord,2))/Exp.FPS;
    trace=squeeze(HoloRecord(:,:,E2(j)));
    fillPlot(trace,time_sec','ci',[],[],[],.2);
    hold on
end
hold on
ylim = kE1;
plot([29/Exp.FPS 29/Exp.FPS],kE1,'--r','LineWidth',1); grid on;

figure()
for j = 1:numel(E2)
    subplot(2,2,j);
    imagesc(zscore(HoloRecord(:,:,E2(j)),[],2)); % caxis([-1 100])
    ylabel('trials');
    xlabel('frames')
end



%% Rerun the algo on processed data and see how it did relative to real data

% activationFrames = BMI2_offline(OnlineData',Exp,.25,E1,E2,frames);

[activationFrames, dffrecord] = BMI2_offline(signals.F(S2P_ID,:),Exp,.25,E1,E2);
activationFrames = find(activationFrames);

figure();
v=[E1 E2(2:end)];%duplicate
for n = 1:numel(v)  
    if n>4
        plot(signals.F(S2P_ID(v(n)),:)-(n*7),'k');
    else
        plot(signals.F(S2P_ID(v(n)),:)-(n*7),'r');
    end
    hold on
end

scatter(rewardFrames', ones(size(rewardFrames))+10,50,'.b')
plot([BL_frames(1) BL_frames(end)],[15 15],'k','LineWidth',5);
plot([Post_frames(1) Post_frames(end)],[15 15],'m','LineWidth',5);
scatter(activationFrames', ones(size(activationFrames))+14,50,'.g')


preProcess_preEPM=60*numel(intersect(rewardFrames,BL_frames))/(numel(BL_frames)/Exp.FPS);
preProcess_postEPM=60*numel(intersect(rewardFrames,Post_frames))/(numel(Post_frames)/Exp.FPS);

postProcess_preEPM=60*numel(intersect(activationFrames,BL_frames))/(numel(BL_frames)/Exp.FPS);
postProcess_postEPM=60*numel(intersect(activationFrames,Post_frames))/(numel(Post_frames)/Exp.FPS);

figure();
plot([1 2],[postProcess_preEPM postProcess_postEPM],'-ko')
hold on
plot([1 2],[preProcess_preEPM preProcess_postEPM],'-bo')
hold on
xlim([0 3])
ylabel('Events Per Minute')
legend('Post Processing','Online')

%% Coactivation
coAct=zeros(numel(E1),size(signals.F,2));

for c=1:numel(E1)

    Svec = (smooth(signals.spikes(S2P_ID(E1(c)),:),3));
    Svec(Svec<.4)=0;
    Svec(find(Svec))=1;
    coAct(c,:)=Svec;

end

A=sum(coAct);
A=A/numel(E1);

coActE2=zeros(numel(E2),size(signals.F,2));

for c=1:numel(E2)

    Svec = (smooth(signals.spikes(S2P_ID(E2(c)),:),3));
    Svec(Svec<.4)=0;
    Svec(find(Svec))=1;

    coActE2(c,:)=Svec;

end

B=sum(coActE2);
B=B/numel(E2);

t=(1:size(signals.F,2))/Exp.FPM;
k=100;
plot(t,movmean(A,k))
hold on
plot(t,movmean(B,k))
j=ylim;

plot([BL_frames(1) BL_frames(end)]/Exp.FPM,[j(2) j(2)],'k','LineWidth',5);
plot([Post_frames(1) Post_frames(end)]/Exp.FPM,[j(2) j(2)],'m','LineWidth',5);
xlabel('Minutes')
ylabel('Coactivation Indx');

%% noise analysis



%% population corr matrix
clear Bl_Pop_Corrs Post_Pop_Corrs
theNeurons = 1:size(signals.F,1);
theNeurons(S2P_ID)=[];    %take out the BMI neurons.  only look at non bmi neurons.
Bl_Pop_Corrs = zeros(numel(theNeurons));
Post_Pop_Corrs = zeros(numel(theNeurons));

 for c1=1:numel(theNeurons) 
     for c2=1:numel(theNeurons)      
         if c2>=c1                         
            Bl_Pop_Corrs(c1,c2)=nan;
         else
            Bl_Pop_Corrs(c1,c2)=corr(signals.Fraw(theNeurons(c1),BL_frames)',signals.Fraw(theNeurons(c2),BL_frames)');
         end
                  
         if Bl_Pop_Corrs(c1,c2)>.99
             Bl_Pop_Corrs(c1,c2)=nan;
         end
         
     end
 end
 
 %compute pairwise pearsons p for all cells during post
  for c1=1:numel(theNeurons) 
     for c2=1:numel(theNeurons)     
         if c2>=c1                    
            Post_Pop_Corrs(c1,c2)=nan;
         else
            Post_Pop_Corrs(c1,c2)=corr(signals.Fraw(theNeurons(c1),Post_frames)',signals.Fraw(theNeurons(c2),Post_frames)');
         end
                  
         if Post_Pop_Corrs(c1,c2)>.99
             Post_Pop_Corrs(c1,c2)=nan;
         end
         
     end
  end
 
  
  
 %% analyze pairwise corrs
 
figure();
plotSpread({Bl_Pop_Corrs,Post_Pop_Corrs},'showMM',4);
title('E1 pairwise correlations baseline - post');
%ylim([-.1, .2])
ylabel('pairwise correlation');

Post_Pop_Corrs(isnan(Post_Pop_Corrs))=[];
Bl_Pop_Corrs(isnan(Bl_Pop_Corrs))=[];

[~, p] = ttest2(Bl_Pop_Corrs,Post_Pop_Corrs);

text(1.5, 0.4, ['p = ', num2str(p)])

figure()
ecdf(Bl_Pop_Corrs)
hold on
ecdf(Post_Pop_Corrs)
xlabel('Pairwise Correlations');
ylabel('Cum Dist');
xlim([-.3 .3])
legend('pre','post')


%% redo but without the nans for pretty
clear Bl_Pop_Corrs Post_Pop_Corrs
theNeurons = 1:size(signals.F,1);
theNeurons(S2P_ID)=[];    %take out the BMI neurons.  only look at non bmi neurons.
Bl_Pop_Corrs = zeros(numel(theNeurons));
Post_Pop_Corrs = zeros(numel(theNeurons));

for c1=1:numel(theNeurons)
     for c2=1:numel(theNeurons)     
         Bl_Pop_Corrs(c1,c2)=corr(signals.Fraw(theNeurons(c1),BL_frames)',signals.Fraw(theNeurons(c2),BL_frames)');
     end
end
 
%compute pairwise pearsons p for all cells during post
for c1=1:numel(theNeurons)
    for c2=1:numel(theNeurons)     
       Post_Pop_Corrs(c1,c2)=corr(signals.Fraw(theNeurons(c1),Post_frames)',signals.Fraw(theNeurons(c2),Post_frames)');
    end
end
  
[L,idx] = luczak_scrunch(Bl_Pop_Corrs);
 
figure();
subplot(1,2,1);
imagesc(L); caxis([-.5 .5]);
axis square;
subplot(1,2,2);
imagesc(Post_Pop_Corrs(idx,idx)); caxis([-.5 .5]);
axis square;

%% Checking if there is a decrease in baseline over time

smoothFraw = zeros(size(signals.Fraw));
p = zeros(size(signals.Fraw,1),1);
figure
hold on

for ii=1:size(signals.Fraw,1)
    smoothFraw(ii, :) = smooth(signals.Fraw(ii,:), 1000);
    plot(smoothFraw(ii, :), 'color',[0.5, 0.5, 0.5], 'linewidth', 0.5)
    paux = polyfit(1:size(signals.Fraw,2), signals.Fraw(1,:), 1);
    p(ii) = paux(1);
end

angle = round(rad2deg(atan(mean(p)/size(signals.Fraw,2))));
plot(mean(smoothFraw,1), 'k','linewidth',2)
txt = ['mean gradient (deg)= ', num2str(angle)];
text(1e4, -25, txt )

%% in case we need to add JAVA change here
javaaddpath('C:/Users/senis/Dropbox/HoloBMI/Analysis functions/jidt/infoDynamics.jar');
%% Effective connectivity transfer of entropy

if ~ exist('vtaStimHistory','var')
    [vtaStimHistory, dffrecord] = BMI2_offline(signals.F(S2P_ID,:),Exp,.25,E1,E2);
end


%TePFholo = TePF; TeBLFholo = TeBLF;
clear TeBLF TePF TeCalc
% actfull_frames = VTAholo;  %change to holo whenever
% actfull_frames = VTAreal;  %change to holo whenever
actfull_frames = rewardFrames;  %change to holo whenever

sourceMVArray = signals.spikes(S2P_ID(E1),:);
destMVArray = signals.spikes(S2P_ID(E1),:);
% sourceMVArray = signals.spikes(S2P_ID(E2),:);
% destMVArray = signals.spikes(S2P_ID(E2),:);

sourceDim = size(sourceMVArray, 1);
destDim = size(destMVArray, 1);

sizePF = Post_frames(end)-Post_frames(1);
sizeBLF = BL_frames(end)- BL_frames(1);

trialsize=20;

TeBLF = zeros(sourceDim, destDim, round((sizeBLF-trialsize)/trialsize)-1)+nan;
TePF = zeros(sourceDim, destDim, size(actfull_frames,1)-1)+nan;

%teCalc=javaObject('infodynamics.measures.continuous.kraskov.TransferEntropyCalculatorMultiVariateKraskov');
teCalc=javaObject('infodynamics.measures.continuous.kraskov.TransferEntropyCalculatorKraskov');
teCalc.initialise(1); % Use history length 1 (Schreiber k=1)
teCalc.setProperty('k', '4'); % Use Kraskov parameter K=4 for 4 nearest points


for n1 = 1: sourceDim
    for n2 = 1: destDim
        if n1~=n2
            for frames=1:size(TeBLF,3)
                teCalc.setObservations(sourceMVArray(n1, BL_frames(1)+frames*trialsize:BL_frames(1)+frames*trialsize+trialsize),destMVArray(n2, BL_frames(1)+frames*trialsize:BL_frames(1)+frames*trialsize+trialsize))
                TeBLF(n1,n2,frames) = teCalc.computeAverageLocalOfObservations()*Exp.FPS;
            end

            for frames=1:size(actfull_frames)
                teCalc.setObservations(sourceMVArray(n1, actfull_frames(frames)-trialsize:actfull_frames(frames)),destMVArray(n2, actfull_frames(frames)-trialsize:actfull_frames(frames)))
                TePF(n1,n2,frames) = teCalc.computeAverageLocalOfObservations()*Exp.FPS;
            end 
        end
    end
end

%to check average TE in different moments of time
aveTePFneuron = reshape(nanmean(TePF, 3), [1, size(TePF,1)*size(TePF,2)]);
aveTePFneuron(aveTePFneuron<=0) = []; %remove n1=n2 and negative values
aveTeBLFneuron = reshape(nanmean(TeBLF, 3), [1, size(TeBLF,1)*size(TeBLF,2)]);
aveTeBLFneuron(aveTeBLFneuron<=0) = [];

clear ylim
figure
plotSpread({aveTeBLFneuron, aveTePFneuron},'showMM',4, 'distributionMarkers', '*', 'distributionColor', 'k');
[~, p]=ttest2(aveTeBLFneuron,aveTePFneuron);
text(1.5, -0.05, ['p = ', num2str(p)])
ylim([-0.1,0.5])
xticklabels({'baseline', 'VTA'})

% choosing only the maximum value of each n1/n2 pair... not really a nice
% thing to do anyway
% TeBLFunique = zeros(size(TeBLF,1),size(TeBLF,2))+nan;
% TePFunique = zeros(size(TePF,2), size(TePF,2))+nan;
% 
% for n1=1:4
%     for n2=1:4
%         if isnan(TeBLFunique(n1,n2))
%             TeBLFunique(n1,n2) = max(nanmean(TeBLF(n1,n2,:)), nanmean(TeBLF(n2,n1,:)));
%             TeBLFunique(n2,n1) = -1;
%         end
%         if isnan(TePFunique(n1,n2))
%             TePFunique(n1,n2) = max(nanmean(TePF(n1,n2,:)), nanmean(TePF(n2,n1,:)));
%             TePFunique(n2,n1) = -1;
%         end
%     end
% end
% 
% TeBLFunique = reshape(TeBLFunique,[1, size(TeBLFunique,1)*size(TeBLFunique,2)]);
% TePFunique = reshape(TePFunique,[1, size(TePFunique,1)*size(TePFunique,2)]);
% TeBLFunique(TeBLFunique==-1) = [];
% TePFunique(TePFunique==-1) = [];

%% For this part I need to run it twice the previous one, one with VTAholo one with VTAreal

%to check average TE in different moments of time
aveTePFneuronreal = reshape(nanmean(TePFreal, 3), [1, size(TePFreal,1)*size(TePFreal,2)]);
aveTePFneuronreal(aveTePFneuronreal<=0) = []; %remove n1=n2 and negative values
aveTeBLFneuronreal = reshape(nanmean(TeBLFreal, 3), [1, size(TeBLFreal,1)*size(TeBLFreal,2)]);
aveTeBLFneuronreal(aveTeBLFneuronreal<=0) = [];
aveTePFneuronholo = reshape(nanmean(TePFholo, 3), [1, size(TePFholo,1)*size(TePFholo,2)]);
aveTePFneuronholo(aveTePFneuronholo<=0) = [];


clear ylim

figure()
subplot(1,3,1);
plot(1:1:length(aveTeBLFneuronreal), aveTeBLFneuronreal, 'k*')
ylim([-0.1,0.5])

subplot(1,3,2);
plot(1:1:length(aveTePFneuronholo), aveTePFneuronholo, 'k*')
ylim([-0.1,0.5])

subplot(1,3,3);
plot(1:1:length(aveTePFneuronreal), aveTePFneuronreal, 'k*')
ylim([-0.1,0.5])

figure
plotSpread({aveTeBLFneuronreal, aveTePFneuronholo, aveTePFneuronreal},'showMM',4, 'distributionMarkers', '*', 'distributionColor', 'k');
[~, p]=ttest2(aveTeBLFneuronreal,aveTePFneuronreal);
text(3, -0.05, ['p = ', num2str(p)])
[~, p]=ttest2(aveTeBLFneuronreal,aveTePFneuronholo);
text(2, -0.05, ['p = ', num2str(p)])
[~, p]=ttest2(aveTePFneuronholo,aveTePFneuronreal);
text(2, 0.4, ['p = ', num2str(p)]);
ylim([-0.1,0.5])
xticklabels({'baseline','Holo', 'Post'})

% figure
% % TeBLFuniquereal(TeBLFuniquereal<0)=[];
% % TePFuniquereal(TePFuniquereal<0)=[];
% % TePFuniqueholo(TePFuniqueholo<0)=[];
% plotSpread({TeBLFuniquereal, TePFuniqueholo, TePFuniquereal},'showMM',4, 'distributionMarkers', '*', 'distributionColor', 'k');
% [~, p]=ttest(TeBLFuniquereal,TePFuniquereal);
% text(3, -0.05, ['p = ', num2str(p)])
% [~, p]=ttest(TeBLFuniquereal,TePFuniqueholo);
% text(2, -0.05, ['p = ', num2str(p)])
% [~, p]=ttest(TePFuniqueholo,TePFuniquereal);
% text(2, 0.4, ['p = ', num2str(p)]);
% ylim([-0.1,0.5])

%% Granger causality
% get the data by trial being each trial trialsize defined by
trialsize = 15; %frames
% actfull_frames = VTAholo;  %change to holo whenever
% actfull_frames = VTAreal;  %change to holo whenever
actfull_frames = rewardFrames;  %change to holo whenever

reorder_neurons = 1:size(signals.spikes,1)-1;
reorder_neurons(S2P_ID)=[];
reorder_neurons = [S2P_ID(E1)', S2P_ID(E2)', reorder_neurons];

neurons = reorder_neurons;
% neurons = S2P_ID(E1);

sizePF = Post_frames(end)-Post_frames(1);
sizeBLF = BL_frames(end)- BL_frames(1);

fullData = signals.spikes(neurons,:);
fullBLData = reshape(signals.spikes(neurons,1:BL_frames(end)), [length(neurons), 1, length(BL_frames)]);
fullPostData = reshape(signals.spikes(neurons,Post_frames(1):end), [length(neurons), 1, length(Post_frames)]);
trialData = zeros(size(fullData,1), round((size(fullData,2)-trialsize)/trialsize)-1, trialsize+1)+nan;
trialVTAData = zeros(length(neurons), length(actfull_frames), trialsize+1)+nan;
trialBLData = zeros(length(neurons), round((sizeBLF-trialsize)/trialsize)-1, trialsize+1)+nan;
trialPostData = zeros(length(neurons), round((sizePF-trialsize)/trialsize)-1, trialsize+1)+nan;
trialVTAholoData = zeros(length(neurons), length(VTAholo), trialsize+1)+nan;
trialVTArealData = zeros(length(neurons), length(VTAreal), trialsize+1)+nan;

for n = 1:length(neurons)
    for frames=1:size(trialVTAData,2)
        trialVTAData(n,frames, :) = fullData(n, actfull_frames(frames)-trialsize:actfull_frames(frames));
    end
    for frames=1:size(trialVTAData,2)
        trialVTAData(n,frames, :) = fullData(n, actfull_frames(frames)-trialsize:actfull_frames(frames));
    end
    for frames=1:size(trialVTAholoData,2)
        trialVTAholoData(n,frames, :) = fullData(n, VTAholo(frames)-trialsize:VTAholo(frames));
    end
    for frames=1:size(trialVTArealData,2)
        trialVTArealData(n,frames, :) = fullData(n, VTAreal(frames)-trialsize:VTAreal(frames));
    end
    for frames=1:size(trialBLData,2)
        trialBLData(n,frames, :) = fullData(n, BL_frames(1)+frames*trialsize:BL_frames(1)+frames*trialsize+trialsize);
    end
    for frames=1:size(trialPostData,2)
        trialPostData(n,frames, :) = fullData(n, Post_frames(1)+frames*trialsize:Post_frames(1)+frames*trialsize+trialsize);
    end
end

%% Evaluate Granger Causality
% Variables

regmode   = 'OLS';  % VAR model estimation regression mode ('OLS', 'LWR' or empty for default)
icregmode = 'LWR';  % information criteria regression mode ('OLS', 'LWR' or empty for default)

%morder    = 'AIC';  % model order to use ('actual', 'AIC', 'BIC' or supplied numerical value)
momax     = 20;     % maximum model order for model order estimation

acmaxlags = 5;   % maximum autocovariance lags (empty for automatic calculation)

tstat     = '';     % statistical test for MVGC:  'F' for Granger's F-test (default) or 'chi2' for Geweke's chi2 test
alpha     = 0.05;   % significance level for significance test
mhtc      = 'FDR';  % multiple hypothesis test correction (see routine 'significance')

fs        = 15.2;    % sample rate (Hz)
fres      = [];     % frequency resolution (empty for automatic calculation)

seed      = 0;      % random seed (0 for unseeded)



X=trialBLData([1:3,6:8],:,:);
ntrials = size(X,2);
nobs = size(X,1);

[~,~,moAIC,moBIC] = tsdata_to_infocrit(X,momax,icregmode);
if moAIC < 3
    morder = 3;
else
    morder = moAIC;
end
[A,SIG] = tsdata_to_var(X,morder);
assert(~isbad(A),'VAR estimation failed');
[G,info] = var_to_autocov(A,SIG,acmaxlags);
var_info(info,true); %

F = autocov_to_pwcgc(G);
assert(~isbad(F,false),'GC calculation failed');

pval = mvgc_pval(F,morder,nobs,ntrials,4,4); % take careful note of arguments!
sig  = significance(pval,alpha,mhtc);

figure; clf;
subplot(1,3,1);
plot_pw(F);
caxis([0,0.01])
title('Pairwise-conditional GC');
subplot(1,3,2);
plot_pw(pval);
title('p-values');
subplot(1,3,3);
plot_pw(sig);
title(['Significant at p = ' num2str(alpha)])

cd = mean(F(~isnan(F)));
fprintf('\ncausal density = %f\n',cd);
