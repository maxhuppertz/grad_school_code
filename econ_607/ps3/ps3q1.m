% How to clean everything
clear

% Use LaTeX for plot axis labels
set(groot, 'defaultAxesTickLabelInterpreter', 'latex');  % I am Groot

% Specify graph format
gform = '.svg';

% Specify name of figures directory
fdir = './fig_ps3q1';

% Create it if it doesn't exist
if ~exist(fdir, 'dir')
    mkdir(fdir)
end

% Choose start date for time series to be retrieved (YYYY-MM-DD)
sdate = '1999-01-01';

% Make a string containing today's date (to be able to get most recent
% data when entering other optional arguments)
curdate = clock;
curdate = strcat(num2str(curdate(1)), '-', ...
    num2str(curdate(2), '%02d'), '-', ...
    num2str(curdate(3), '%02d'));

% Time series to be retrieved from FRED (in order: real GDP, non-durable
% consumption, non-residential investment, residential investment)
% fredseries = {'GDPC1', 'PCND', 'PNFI', 'PRFI'};  % Nominal data
fredseries = {'GDPC1', 'PCNDGC96', 'PNFIC1', 'PRFIC1'};  % Real data

% Set tuning parameter for Hodrick-Prescott filter
mu = 1600;

% Set number of lags/leads for cross correlations between real GDP and
% other time series
J = 9;

% Set up matrix of cross correlations
C = zeros(length(fredseries) - 1, 2*J + 1);

% Go through all time series
for k = 1:length(fredseries)
    % Retrieve data on current series from FRED; note that this requires
    % getFredData by Robert Kirby to be added to Matlab, otherwise it won't
    % work; there's also an official toolbox for receiving FRED data, but I
    % don't think it's included in my university account, and I'm not gonna
    % pay money for it
    series = getFredData(fredseries{k}, sdate, curdate, 'lin');
    
    % Get data from series object, take logs
    y = log(series.Data(:, 2));
    
    % Get time dimension from series object
    t = datetime(series.Data(:, 1), 'ConvertFrom', 'datenum');
    
    % Remove not-a-number values (this is super crude, and for a proper
    % project I'd probably make sure I'm only removing NaN's at the
    % beginning or end of the series)
    t(isnan(y)) = [];
    y(isnan(y)) = [];
    
    % Get detrended series and trend estimate 
    [y_dt, T] = hp_filter(y, mu);
    
    % If this is not real GDP, calculate correlation coefficients between
    % this series and GDP for J lags/leads
    if k == 1
        y_gdp = y_dt;
    else
        r = 1;
        for s = -J:J
            C(k-1,r) = ...
                corr2(y_dt(1 + (s > 0)*s : length(y) + (s < 0)*s, 1), ...
                         y_gdp(1 - (s < 0)*s : length(y) - (s > 0)*s, 1));
            r = r + 1;
        end
    end
    
    % Plot data and trend estimate
    subplot(length(fredseries), 2, 2*k-1);
    plot(t, y);
    hold on;
    plot(t, T);
    hold off;
    
    % Add a title and labels
    % Bold face heading if this is the first row
    if k == 1
        title_pref = '\textbf{Actual data (log) and trend estimate}';
    elseif k == 2
        title_pref = '';
    end
    
    % Plus series title (not bold face)
    title(cat(1, title_pref, series.Title), 'Interpreter', 'latex');
    
    % If it's the last row, add an axis label for the time dimension
    if k == length(fredseries)    
        xlabel('Year', 'Interpreter', 'latex');
    end
    
    % Add a label to the vertical axis
    ylabel('$$\log(y_t) \qquad \qquad$$', ...
        'Interpreter', 'latex', 'rotation', 0);
    
    % Plot trend deviation
    subplot(length(fredseries), 2, 2*k);
    plot(t, y_dt);
    
    % Add a title and labels
    if k == 1
        title_pref = '\textbf{Log deviation from trend}';
    elseif k == 2
        title_pref = '';
    end
    
    % Calculate standard deviation of detrended series
    sd = std(y_dt);
    
    % Plus standard deviation of detrended series
    title({title_pref, ['Volatility (SD): ', num2str(sd)]}, ...
        'Interpreter', 'latex');
    
    % Only the last row gets a time axis label
    if k == length(fredseries)    
        xlabel('Year', 'Interpreter', 'latex');
    end
    
    % Label the vertical axis
    ylabel('$$\Delta \log(y_t) \qquad \qquad$$', ...
        'Interpreter', 'latex', 'rotation', 0);
end

% Adjust figure dimensions and save
set(gcf, 'Units', 'inches')
set(gcf, 'Position', [0 0 8*length(fredseries) 4.5*2])
saveas(gcf, fullfile(fdir, strcat('fredseries', gform)))

% Reset interpreter to factory settings
set(groot, 'defaultAxesTickLabelInterpreter','factory');
set(groot, 'defaultLegendInterpreter','factory');