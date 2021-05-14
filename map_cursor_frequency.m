function freq = map_cursor_frequency(cursor, fb_mapping, low_target)
%{
    function to map a value of cursor to a frequency
    Auditory feedback calculation
    freq = a*exp(b*(cursor_trunc-cursor_min))
    freq_min = a =  a*exp(0)
    freq_max = a*exp(b*(cursor_max-cursor_min))
    b = log(freq_max/a)/(cursor_max-cursor_min)
    param: cursor_min, cursor_max
    %}

    %%
    %Handle target -> freq:
    if low_target
        %This means cursor up makes auditory freq go down:
        cursor      = -cursor; 
        cursor_min  = -fb_mapping.cursor_max;
        cursor_max  = -fb_mapping.cursor_min;
    else
        %This means cursor up makes auditory freq go up:
        cursor_min  = fb_mapping.cursor_min;
        cursor_max  = fb_mapping.cursor_max;
    end

    %%
    cursor_trunc    = max(cursor, cursor_min); 
    cursor_trunc    = min(cursor_trunc, cursor_max); 
    freq = fb_mapping.a*exp(fb_mapping.b*(cursor_trunc-cursor_min));
    freq = double(freq); 
    % h = figure;
    % plot(cursor, freq); 