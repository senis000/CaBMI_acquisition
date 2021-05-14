function plot_saving(h, x_l, y_l, titl, folder_plot, file_name)
%{
    function to save plots
%}

    xlabel(x_l); 
    ylabel(y_l); 
    title(titl); 
    im_path_png = fullfile(folder_plot, strcat(file_name, '.png')); 
    im_path_eps = fullfile(folder_plot, strcat(file_name, '.eps')); 
    saveas(h, im_path_png); 
    saveas(h, im_path_eps);
    close('all')