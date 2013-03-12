function [prediction_error, significant_error] = penguin_kf(output, input, p)
%
%prediction_error = penguin_kf(output, input, p)
%
%Uses the Kalman Filter to predict the output values using the past
%p-timesteps of the input values (default p = 10). Inputs can have many dimensions and can
%include the output stream (autocorrelation). Function returns the
%prediction error. Peaks in predictive error correspond to behaviour
%changes. Also returns a binary string indicating where the rediction error
%is signigficantly greater than average at the 5% level.

%This file: Richard Mann (2010). All other included code is (c) Stephen J.
%Roberts (1997)

%If number of timesteps is unspecified, use 10
if nargin < 3
    p = 10;
end


%No chweck for common problems.
%Lets assume we have more data points than dimensions
if size(output, 2) > size(output, 1)
    output = output';
end
if size(input, 2) > size(input, 1)
    input = input';
end
%Check inputs match outputs
if size(input,1) ~= size(output, 1)
    disp('Number of input points does not match number of outputs. Aborting')
    prediction_error = NaN;
    significant_error = NaN;
    return;
end


%Just collect what we need from kfManyto1. Use 0.1 as a rough estimate of
%observation noise: implies observation noise is ~10% of observed
%variation, but will be adjusted as the algorihtm learns.

[~, y_pred] = kfManyTo1(output, input, p, 0.1);

prediction_error = y_pred-output;

%Rough gauge of significance, assuming that th majority of the data is
%steady state (we use the variance of the whole prediction_error to assign
%significance to the peaks). 2 tailed Z test (assuming large data size). We
%also assume that the mean error should be zero.

significant_error = abs(prediction_error) > 1.96*std(prediction_error);













