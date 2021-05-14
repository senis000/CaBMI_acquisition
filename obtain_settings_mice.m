function [mice_settings] = obtain_settings_mice(mice_name)
    %{
    this function keeps info that is specific of each mouse mice_name can be () 
    %}
    mice_settings.mice_name = mice_name;
    if mice_name == 'IT01'
        mice_settings.target_low = true;
    else
        mice_settings.target_low = true;
    end
    
    

