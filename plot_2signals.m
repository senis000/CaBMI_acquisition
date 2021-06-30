function plot_2signals(signal_1, signal_2, E_id, chosen_colors, ...
    x_l, y_l, titl, folder_plot, file_name, ind, offset, offset_2)
%{
    function to plot 2 signals in top of each other
%}
    if nargin < 10
        ind = 1:length(E_id);
    end
    if nargin < 11
        offset = 0;
    end
    if nargin < 12
        offset_2 = zeros(size(signal_2, 1));
    end
    
    
    h = figure; hold on;
    for i=1:size(signal_1, 1)
        y_a_plot = signal_1(ind(i), :);
        y_b_plot = signal_2(ind(i), :);
        y_a_plot = y_a_plot-min(y_a_plot); 
        y_b_plot = y_b_plot-min(y_b_plot); 
        y_amp = max([y_a_plot, y_b_plot]); 
        if i>1
            offset = offset + y_amp;
        end
        y_b_plot = y_b_plot-offset - offset_2(ind(i));
        y_a_plot = y_a_plot-offset;
        plot(1:size(signal_1, 2), y_a_plot, 'Color', chosen_colors{E_id(ind(i))}); 
        plot(1:size(signal_2, 2), y_b_plot, 'Color', 'k'); 
    end
    
    plot_saving(h, x_l, y_l, titl, folder_plot, file_name)