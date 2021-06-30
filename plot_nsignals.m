function h = plot_nsignals(signal_n, x_l, y_l, titl, folder_plot, file_name, offset, legend_str, save_flag)
%{
    function to plot n signals in top of each other with no further ado
%}
    if nargin < 7
        offset = 0;
    end
    
    if nargin < 8
        legend_str =[];
    end
    
    h = figure;
    hold on;
    for i=1:size(signal_n, 1)
        y_a_plot = signal_n(i, :);
        y_a_plot = y_a_plot-min(y_a_plot);  
        if i>1 
            offset = offset + max(y_a_plot);
        end
        y_a_plot = y_a_plot-offset;
        plot(1:size(signal_n, 2), y_a_plot, 'linewidth',1.5);  
    end
    legend(legend_str)
    set(gca,'ytick',[])
    
    if save_flag
        plot_saving(h, x_l, y_l, titl, folder_plot, file_name)
    end