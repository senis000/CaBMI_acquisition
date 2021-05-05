function [miceSettings] = define_mice_settings(miceName)
% this function keeps info that is specific of each mouse
%   miceName can be ()

if miceName == 'IT01'
    miceSettings.target_low = true;
else
    miceSettings.target_low = true;
end

