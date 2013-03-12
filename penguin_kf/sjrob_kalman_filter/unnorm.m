%[Y] = unnorm(X,D)
% Y : unnormalised data set
% X : input data set
% D : data set providing normal stats

function [Y] = unnorm(X,D)

c = diag(cov(D))';
m = mean(D);

Y = X.*(ones(size(X,1),1)*sqrt(c));	% restores variance
Y = Y + ones(size(X,1),1)*m;	% adds mean

return;
