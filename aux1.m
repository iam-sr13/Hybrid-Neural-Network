%% Train SNN
% Setup
rand('seed', 42);
clear edbn opts;
edbn.sizes = [9216 100 10];
opts.numepochs = 6;
opts.batchsize = 73;

[edbn, opts] = edbnsetup(edbn, opts);

% Train
fprintf('Beginning SNN training:\n');
% Use unsupervised training on the hidden layer
edbn = edbntrain(edbn, train_x, opts);
% Use supervised training on the top layer
edbn = edbntoptrain(edbn, train_x, opts, train_y);

fprintf('Hybrid Transfer Learning complete successfully\n');

% Show results
figure;
visualize(edbn.erbm{1}.W');   %  Visualize the RBM weights
er = edbntest (edbn, test_x, test_y);
fprintf('Scored: %2.2f\n', (1-er)*100);

%% Show the EDBN in action
spike_list = live_edbn(edbn, test_x(1, :), opts);
output_idxs = (spike_list.layers == numel(edbn.sizes));

figure(2); clf;
hist(spike_list.addrs(output_idxs) - 1, 0:edbn.sizes(end));
xlabel('Digit Guessed');
ylabel('Histogram Spike Count');
title('Label Layer Classification Spikes');

%% Export to xml
edbntoxml(edbn, opts, 'mnist_edbn');
