function [best_permutation, min_distance,time_elapsed] = BruteForce(Inputmatrix)
% We  fixed  the first element as initial point. Sicne we update the
% min_distace, at first we give it a large value so 
x = Inputmatrix(:, 1);
y = Inputmatrix(:, 2);
n = length(x);
% we start the timer.
strat_time = tic;
initial_point = 1;
all_permutation = perms(2:n);
min_distance = 1e+100;
best_permutation = [];
for i = 1:size(all_permutation, 1)
    % We need to compute all the route start from initial point and we end
    % up at initial point, and record all the distance
    route = [initial_point, all_permutation(i, :), initial_point];
    total_distance = 0;
    for j = 1:(length(route) - 1)
        total_distance = total_distance + sqrt((x(route(j+1)) - x(route(j)))^2 + (y(route(j+1)) - y(route(j)))^2);
    end
    if total_distance < min_distance
        min_distance = total_distance;
        best_permutation = route;
    end
end
time_elapsed = toc(strat_time);
fprintf('The total time is %.4f\n',time_elapsed);
fprintf('The optimal route is: %s\n', num2str(best_permutation));
fprintf('The optimal total distance is: %.6f\n', min_distance);
% Plot the optimal route
figure;
labels = string(best_permutation);
text(x(best_permutation), y(best_permutation),labels,'FontSize',14)
plot(x(best_permutation), y(best_permutation),'-o');
hold on
plot(x(1), y(1), 'r.', 'LineWidth', 2, 'MarkerSize', 25)
title('Brute-force Method');
xlabel('X Coordinate');
ylabel('Y Coordinate');
legend('Best route','initial point')
grid on;
end
Inputmatrix= readmatrix("C:\Users\wufan\OneDrive\Desktop\tiny.csv");
[a,b,c]=BruteForce(Inputmatrix);