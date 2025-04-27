function [route, total_distance, time_elapsed] = NearestNeighbor(Inputmatrix)
% We  fixed  the first element as initial point.
start_time = tic;
x = Inputmatrix(:, 1);
y = Inputmatrix(:, 2);
n = length(x);
start_time = tic;
%We need to keep track whether a cooridinate been visited or not.

visited = 1;
route = 1; 
current_point = 1;
total_distance = 0;

for k = 2:n
    nearest_distance = 1e+100;
    nearest_point = -1;
    for j = 2:n
        if ~ismember(j, visited)
            % Calculate distance to point j
            dist = sqrt((x(j) - x(current_point))^2 + (y(j) - y(current_point))^2);
            if dist < nearest_distance
                nearest_distance = dist;
                nearest_point = j;
            end
        end
    end
    
    % We need to update our visited list 
    total_distance = total_distance + nearest_distance;
    visited =[visited, nearest_point];
    route = [route, nearest_point];
    current_point = nearest_point;
end

% Then we need to return to our initial point
return_distance = sqrt((x(current_point) - x(1))^2 + (y(current_point) - y(1))^2);
total_distance = total_distance + return_distance;
route = [route, 1];

time_elapsed = toc(start_time);
fprintf('The total time is %.4f\n',time_elapsed);
fprintf('The optimal route is: %s\n', num2str(route));
fprintf('The optimal total distance is: %.6f\n', total_distance);
% Plot the route
figure;
plot(x(route), y(route), '-o');
hold on

plot(x(1), y(1), 'r.', 'LineWidth', 2, 'MarkerSize', 25)
title('Nearest Neighbor Method');
xlabel('X Coordinate');
ylabel('Y Coordinate');
legend('The route','initial point')
grid on;
end
Inputmatrix= readmatrix("C:\Users\wufan\OneDrive\Desktop\tiny.csv");
[a,b,c]=NearestNeighbor(Inputmatrix);