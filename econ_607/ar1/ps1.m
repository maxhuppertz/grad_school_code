% This is the code for problem set 1 of econ 607
% Set random number generator's seed
rng(607)

% Specify model parameters
a = 0;  % alpha
R = [.9, .5];  % Vector of different values for rho
s2 = .1^2;  % Shock process variance
T = 100;  % Length of time series

% Specify parameters for impulse response function (i.e implicit shock
% size)
y0_imp = .01;

% Use LaTeX for axis labels
set(groot, 'defaultAxesTickLabelInterpreter','latex');

% Go through all values of rho
for i = 1:length(R)
    % Get and display current rho
    r = R(1, i);
    disp([newline, 'rho = ', num2str(r)])
    
    % Calculate unconditional mean and standard deviation of AR(1) process
    % and display them
    ybar = a / (1 - r);
    sigma2 = s2 / (1 - r^2);
    disp(['ybar = ', num2str(ybar), ', sigma = ', num2str(sqrt(sigma2))])
    y0 = ybar;

    % Generate AR(1) data
    y = ar1(a, r, s2, T, y0);

    % Generate AR(1) data for impulse response function
    y_imp = ar1(a, r, 0, T, y0_imp);

    % Calculate mean and standard deviation
    disp(['Sample mean: ', num2str(mean(y)), ', SD: ', num2str(std(y))])

    % Scatter plot of y_t and y_t+1
    subplot(length(R), 2, 2*i-1);
    scatter(y(1:end-1, 1), y(2:end, 1));
    
    % Add a title and labels
    title(['\textbf{Scatter plot}: $$\rho = ' num2str(r), ...
        ', \: \bar{y} = ', num2str(ybar), ', \: \sigma = ', ...
        num2str(sqrt(sigma2)), ', \: \bar{y}_n = ', num2str(mean(y)), ...
        ', \: S_n = ', num2str(std(y)), '$$'], ...
        'Interpreter', 'latex');
    xlabel('$$y_{t-1}$$', 'Interpreter', 'latex');
    ylabel('$$y_t \qquad$$', 'Interpreter', 'latex', 'rotation', 0);
    
    % Plot impulse response function
    subplot(length(R), 2, 2*i);
    plot(y_imp);
    title('\textbf{Impulse response}', 'Interpreter', 'latex');
    xlabel('$$t$$', 'Interpreter', 'latex');
    ylabel('$$\Delta y \qquad$$', 'Interpreter', 'latex', 'rotation', 0);
end

% Adjust figure dimensions and save
set(gcf, 'Units', 'inches')
set(gcf, 'Position', [0 0 16 9])
saveas(gcf, 'ar1.svg')

% Reset interpreter to factory settings
set(groot, 'defaultAxesTickLabelInterpreter','factory');
set(groot, 'defaultLegendInterpreter','factory');