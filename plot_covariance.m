function plot_covariance(neuronal_activity, folder_plot)
%{
    function to plot covariance of neuronal activity
%}
    h = figure;
    imagesc(neuronal_activity); 
    colorbar
    axis square
    colormap;
    caxis([-0.2 0.5]); 
    plot_saving(h, 'roi', 'roi', 'neural cov', folder_plot, 'neural_cov_mat')

    [~,s,~] = svd(neuronal_activity); 
    s_cumsum = cumsum(diag(s))/sum(diag(s)); 
    h = figure;
    plot(s_cumsum, '.-', 'MarkerSize', 20); 
    axis square  
    plot_saving(h, 'PC', 'Frac Var Explained', 'DFF Smooth PCA Covariance', folder_plot, 'cov_pca')