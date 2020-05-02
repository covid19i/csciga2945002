processors1 = [4, 16, 64, 256];
processors2 = [1, 4, 16, 64];

processors1_run2 = [4, 16, 64];
processors2_run2 = [1, 4, 16, 64];

time1 = [0.160852, 0.186441, 1.107792, 2.327834];
time2 = [8.982452, 2.376736, 0.766782, 1.199778];
speedup = max(time2) * time2.^-1;
idealspeedup = processors2 / min(processors2);


time1_run2 = [0.139245, 1.020928, 1.336720];
time2_run2 = [9.032821, 2.374013, 1.623890, 1.382160];
speedup_run2 = max(time2_run2) * time2_run2.^-1;
idealspeedup_run2 = processors2_run2 / min(processors2_run2);

figure(1)

str_title = sprintf('Weak Scaling study of 2D Jacobian Smoother with MPI');
title(str_title);
plotspec = 'b-';
plot(processors1, time1, plotspec);
hold on;
plotspec = 'r-';
plot(processors1_run2, time1_run2, plotspec);
xlabel('No of processes')
ylabel('Time')
legend('Run 1', 'Run 2')

figure(2)

str_title = sprintf('Strong Scaling study of 2D Jacobian Smoother with MPI');
title(str_title);
plotspec = 'b-';
plotspec2 = 'b--';
plot(processors2, speedup, plotspec, processors2, idealspeedup, plotspec2);
hold on;
plotspec3 = 'r-';
plotspec4 = 'r--';
plot(processors2_run2, speedup_run2, plotspec3);
xlabel('No of processes')
ylabel('Speedup')
legend('Speedup Run 1', 'Ideal Speedup Run', 'Speedup Run 2')