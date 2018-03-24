% How to clean everything
clear

% Use LaTeX for plot axis labels
set(groot, 'defaultAxesTickLabelInterpreter', 'latex');  % I am Groot

% Specify graph format
gform = '.svg';

% Save current working directory
dir_orig = pwd;

% Specify name of figures directory
fdir = './fig_ps5q1';

% Specify name of data directory
ddir = './dat_ps5q1';

% Put them in a string array
dirs = {fdir, ddir};

% Create them if they don't exist
for i = 1:length(dirs)
    if ~exist(dirs{i}, 'dir')
        mkdir(dirs{i})
    end
end

% Choose start date for time series to be retrieved (YYYY-MM-DD)
sdate = '1948-01-01';

% Make a string containing today's date (to be able to get most recent
% data when entering other optional arguments)
curdate = clock;
curdate = strcat(num2str(curdate(1)), '-', ...
    num2str(curdate(2), '%02d'), '-', ...
    num2str(curdate(3), '%02d'));

% Get data on TFP from San Francisco FED
% Specify URL and file name
tfpurl = ...
    'https://www.frbsf.org/economic-research/files/quarterly_tfp.xlsx';
tfpfnm = 'sffed_tfp.xlsx';

% Change to data directory and save data
cd(ddir)
websave(tfpfnm, tfpurl, weboptions('Timeout',Inf));

% Read data
tfpset = xlsread(tfpfnm, 'quarterly');

% Extract TFP data (okay, so the date is hard-coded here, and this better
% match up with your start date, but basically it's quarters since 1947:Q1
% plus one; the eleventh column is the TFP data I need) 
tfp = tfpset(5:end, 11);

% Convert it to levels
tfp(1) = 1;

% The architecture of the spreadsheet is such that there's a NaN row before
% some summary statistics at the end, so cut out before that happens
cut = find(isnan(tfp));
tfp = tfp(1:cut-1);

% Go through all TFP values
for i = 2:length(tfp)
    % Iteratively convert them back to levels
    tfp(i) = tfp(i-1) * exp(tfp(i)/400);
end

% Make it into a structure including a series title, so it's conformable
% with the FRED data
% First of all, make a datetime vector dating it (but then convert it
% back to datenum format to be able to store it in the structure)
t = datetime(sdate);
t = t:calmonths(3):curdate;
t = t(1:length(tfp));
t = datenum(t);

% Put that and the data into a structure
tfpstrc.Data(:, 1) = t;
tfpstrc.Data(:, 2) = tfp;
tfpstrc.Title = 'Total Factor Productivity';
tfpstrc.Units = '';

% Time series to be retrieved from FRED (in order: nominal GDP, private
% fixed investment, real compensation per hour worked in the non-farm
% business sector, index of total hours worked)
fredseries = {'GDP', 'PNFI', 'COMPRNFB', 'HOANBS'};

% Manually construct sum of services and non-durable consumption
cp_svc = getFredData('PCESV', sdate, curdate, 'lin');
cp_ndr = getFredData('PCDG', sdate, curdate, 'lin');

% Add both series together and fill in other fields
cpstrc.Data(:, 2) = cp_svc.Data(:, 2) + cp_ndr.Data(:, 2);
cpstrc.Data(:, 1) = cp_svc.Data(:, 1);
cpstrc.Title = 'Consumption: Durable Goods and Services';
cpstrc.Units = 'Billions';

% Get data on population and GDP deflator
pop = getFredData('CNP16OV', sdate, curdate, 'lin');
ydef = getFredData('GDPDEF', sdate, curdate, 'lin');

% Population needs to be converted to quarterly data
pop.Data = pop.Data(1:3:end, :);

% Select which series to convert to per-capita and/or real values (it's not
% really per capita, obviously, it's per person above 16 years of age who's
% not currently in the armed forces or "inmates of institutions", which btw
% includes "homes for the aged")
toreal = {'GDP', 'PNFI', cpstrc};
topc = {'GDP', 'PNFI', 'HOANBS', cpstrc};

% Set tuning parameter for Hodrick-Prescott filter
mu = 1600;

% Change back to original directory, where the HP filter code lives
cd(dir_orig)

% Specify which series to plot
plotseries = {'GDP', cpstrc, 'PNFI', 'HOANBS', 'COMPRNFB', tfpstrc};

% Set number of lags/leads for cross correlations between real GDP and
% other time series
J = 9;

% Go through all time series
for k = 1:length(plotseries)
    % Retrieve data on current series from FRED; note that this requires
    % getFredData by Robert Kirby to be added to Matlab, otherwise it won't
    % work; there's also an official toolbox for receiving FRED data, but I
    % don't think it's included in my university account, and I'm not gonna
    % pay money for it
    if any(strcmp(fredseries, plotseries{k}))
        series = getFredData(plotseries{k}, sdate, curdate, 'lin');
    else
        series = plotseries{k};
    end
    
    % Figure out units of the current series (FRED is pretty consistent
    % with their units, so it's easy to just pick out the first word of the
    % string based on breaking it up after the first space, but obviously
    % you'd really want to use a super comprehensive library here)
    units = strtok(series.Units, ' ');
    if ~strcmp(units, '')
        units = units(1);
    end
    
    % Convert series based on units
    if strcmp(units, 'Billions')
        series.Data(:, 2) = series.Data(:, 2) * 10^9;
    elseif strcmp(units, 'Index')
        series.Data(:, 2) = series.Data(:, 2) / 100;
    end
    
    % Convert to real values if desired
    if any(strcmp(plotseries{k}, toreal))
        series.Data(:, 2) = series.Data(:, 2) .* (ydef.Data(:, 2)/100);
    end
    
    % Convert to per-capita terms if desired
    if any(strcmp(plotseries{k}, topc))
        series.Data(:, 2) = series.Data(:, 2) ./ ...
            (pop.Data(1:length(series.Data(:, 2)), 2) * 1000);
    end
        
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
    
    % Save real GDP series
    if k == 1
        y_gdp = y_dt;
    end
        
    % Plot data and trend estimate
    subplot(length(plotseries), 2, 2*k-1);
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
    if k == length(plotseries)    
        xlabel('Year', 'Interpreter', 'latex');
    end
    
    % Add a label to the vertical axis
    ylabel('$$\log(y_t) \qquad \qquad$$', ...
        'Interpreter', 'latex', 'rotation', 0);
    
    % Plot trend deviation
    subplot(length(plotseries), 2, 2*k);
    plot(t, y_dt);
    
    % Add a title and labels
    if k == 1
        title_pref = '\textbf{Log deviation from trend}';
    elseif k == 2
        title_pref = '';
    end
    
    % Calculate standard deviation of detrended series
    sd = std(y_dt);
    
    % Save the standard deviation of real per capita GDP
    if k == 1
        sd_y = sd;
    end
    
    % Calculate first order autocorrelation
    acor = corr2(y_dt(1:end-1), y_dt(2:end));
    
    % Calculate contemporaneous correlation with GDP
    ccor = corr2(y_dt, y_gdp);
    
    % Plus standard deviation of detrended series
    title({title_pref, ['$$\sigma_y$$: ', num2str(sd), ...
        '; $$\sigma_y / \sigma_{GDP}$$: ', num2str(sd/sd_y), ...
        '; $$\rho_1$$: ', num2str(acor), '; $$\rho_{GDP}$$: ', ...
        num2str(ccor)]}, 'Interpreter', 'latex');
    
    % Only the last row gets a time axis label
    if k == length(plotseries)    
        xlabel('Year', 'Interpreter', 'latex');
    end
    
    % Label the vertical axis
    ylabel('$$\Delta \log(y_t) \qquad \qquad$$', ...
        'Interpreter', 'latex', 'rotation', 0);
end

% Adjust figure dimensions and save
set(gcf, 'Units', 'inches')
set(gcf, 'Position', [0 0 8*length(plotseries) 4.5*2])
saveas(gcf, fullfile(fdir, strcat('plotseries', gform)))

% Reset interpreter to factory settings
set(groot, 'defaultAxesTickLabelInterpreter','factory');
set(groot, 'defaultLegendInterpreter','factory');