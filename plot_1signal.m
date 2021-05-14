function plot_1signal(neuronal_activity, E_id, chosen_colors, ...
    x_l, y_l, titl, folder_plot, file_name, offset)
%{
    function to plot neuronal activity
%}
    if nargin < 9
        offset = 0;
    end
 
    h = figure; hold on;
    for i=1:size(neuronal_activity, 1)
        y_plot = neuronal_activity(i, :);
        y_plot = y_plot-min(y_plot); 
        y_amp = max(y_plot); 
        if i>1
            offset = offset + y_amp;
        end
        y_plot = y_plot-offset;
        plot(1:size(neuronal_activity, 2), y_plot, 'Color', chosen_colors{E_id(i)}); 
    end
    plot_saving(h, x_l, y_l, titl, folder_plot, file_name)