%[Y] = normalis(X,D,mD,covD)
% Y : normalised data set
% X : input data set
% D : data set providing normal stats

function [Y] = normalis(X,D,mD,covD)

if (nargin <4),
    c = diag(cov(D))';
    m = mean(D);
else
    c = diag(covD)';
    m = mD;
end;

f = find(c==0);
c(f) = 1;

X = X - ones(size(X,1),1)*m;	% removes mean
Y = X./(ones(size(X,1),1)*sqrt(c));	% unit variance

return;
