%cuda2.cims.nyu.edu
%total_duration = [67.170169, 449.028698, 480.411388, 560.280800, 154.483718, 165.991442];
%duration_per_iteration = [0.000671, 0.000449, 0.000480, 0.000560, 0.000772, 0.000830];
%blocks = [2, 2, 2, 4, 8, 16];
%threads = [1024, 1024, 1024, 1024, 1024, 1024];
%iterations = [100000, 1000000, 1000000, 1000000, 200000, 200000];
%test_accuracy = [0.330100, 0.349900, 0.350000, 0.351900, 0.349000	, 0.352700];


%cuda3.cims.nyu.edu
total_duration = [118.767677, 123.282203, 137.629988, 141.911966, 155.535668, 230.499940, 451.138942];
duration_per_iteration = [0.000594, 0.000616, 0.000688, 0.000710, 0.000778, 0.001152, 0.002256];
blocks = [1, 2, 4, 8, 16, 32, 64];
threads = [1024, 1024, 1024, 1024, 1024, 1024, 1024];
iterations = [200000, 200000, 200000, 200000, 200000, 200000, 200000];
test_accuracy = [0.336200, 0.345400, 0.349200, 0.349000, 0.350000	, 0.353000, 0.353900];


figure(2)
plotspec = 'r-';
plotspec2 = 'b-';
plot(blocks, duration_per_iteration, plotspec2);
str_title = sprintf('Duration per iteration vs Number of Blocks');
title(str_title)
xlabel('Number of Blocks')
ylabel('Duration')
legend('Duration/Iteration')

figure(1)
plotspec = 'r-';
plotspec2 = 'b-';
plot(blocks, test_accuracy, plotspec2);
str_title = sprintf('Accuracy after 200000 iterations');
title(str_title)
xlabel('Number of Blocks')
ylabel('Accuracy on test data')
legend('Accuracy')


%For 0.345 test accuracy required time (same block size) for different
%number of blocks
blocks = [1, 2, 4, 8, 16, 32, 64];
iterations = [400000, 200000, 100000, 80000, 40000, 40000, 40000];
one_block_iterations = max(iterations);
ideal_iterations = blocks .^ (-1) * one_block_iterations;



figure(3)
plotspec = 'r-';
plotspec2 = 'b-';
plot(blocks, iterations, plotspec, blocks, ideal_iterations, plotspec2);
str_title = sprintf('Iterations for 0.345 accuracy vs Blocks');
title(str_title)
xlabel('Number of Blocks')
ylabel('Iterations')
legend('Iterations', 'Ideal Iterations')