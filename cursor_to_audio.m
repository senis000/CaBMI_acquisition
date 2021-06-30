function freq = cursor_to_audio(cursor, fb_mapping, target_low_freq)

%{
    Cursor is the neural control signal.  From old work, this would be: 
    sumE2-sumE1

    Auditory feedback calculation
    freq = a*exp(b*(cursor_trunc-cursor_min))
    freq_min = a =  a*exp(0)
    freq_max = a*exp(b*(cursor_max-cursor_min))
    b = log(freq_max/a)/(cursor_max-cursor_min)
    param: cursor_min, cursor_max
    %
    If target_low_freq == 1, then we negate cursor and cursor_min,
    cursor_max

    This is because cursor is E2-E1.  Our target is gonna be positive. 
 %}

    % obtain the max/min
    if target_low_freq == 1
        %This means cursor up makes auditory freq go down:
        cursor      = -cursor; 
        cursor_min  = -fb_mapping.cursor_max;
        cursor_max  = -fb_mapping.cursor_min;
    else
        %This means cursor up makes auditory freq go up:
        cursor_min  = fb_mapping.cursor_min;
        cursor_max  = fb_mapping.cursor_max;
    end

    % calculate the frequency
    cursor_trunc    = max(cursor, cursor_min); 
    cursor_trunc    = min(cursor_trunc, cursor_max); 
    freq = fb_mapping.a*exp(fb_mapping.b*(cursor_trunc-cursor_min));
    freq = double(freq); 
    % h = figure;
    % plot(cursor, freq); 