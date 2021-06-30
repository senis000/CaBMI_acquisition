function [E1, E2] = select_ensemble_neurons(baseline_data)
%{
    Function to obtain the ensemble neurons
    baseline_data -> baseline data
    %}
    
    % plot neurons
    
    S = nanstd(baseline_data');
    [~, ind] = sort(S, 'descend');

    disp('Displaying the best 20 neurons, please select ensemble neurons when promted')
    figure()

    for idx=1:20
        subplot(4,5,idx)
        plot(baseline_data(ind(idx), :)');
        title(['ROI ' int2str(ind(idx))]);
    end

    while true
        E1 = get_ensemble_neur(1);
        E2 = get_ensemble_neur(2);
        fprintf('\n')
        fprintf(' E1: %.d  ', E1)
        fprintf(' E2: %.d  ', E2)
        satisfied = input('are you happy with this selection Y/N [Y]:', 's');
        if isempty(satisfied) || strcmp(satisfied, 'Y') || strcmp(satisfied, 'y')
            break
        end
    end

end

function neurons = get_ensemble_neur(num_ensemble)
    neurons = [];
    while true
        x = input(sprintf('select a neuron for ensemble E %d. If enough neurons press enter: ', num_ensemble));
        if isempty(x)
            break
        else
            neurons(end + 1) = x;
        end
    end
end
