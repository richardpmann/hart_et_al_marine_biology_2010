% [A,y_pred,Ea,Ey,P,Vt,Tt,Ct] = kfManyTo1(Z,X,p,V,constrain);
%	runs Kalman-Bucy filter over observations matrices Z = [Z1,Z2,...Zn]
%	for 1-step prediction onto matrix X with elements y_t
%	with model order p
%	V = initial covariance of observation sequence noise
%		put in -ve V to stop adaption of V: uses abs(V)
%	returns model parameter estimation sequence A,
%	sequence of predicted outcomes y_pred
%	and error matrix Ey (reshaped) for y and Ea for a
%	along with inovation prob P = P(y_t | D_t-1) = evidence
%   and Vt the variance of the observation noise process
%   and Tt the variance of the state noise process.
%   Interleaving of Y,Z means that same kf eqns can be used
%	(c) S.J.Roberts Nov 1997, modified August 1998, June 2006, June 2007

function [A,y_pred,Ea,Ey,P,Vt,Tt,Ct] = kfManyTo1(Z,X,p,V,constrain)

if (nargin < 5)
  constrain = 0; % default is not to constrain system to stability i.e. |a| < 1
end;

if ((constrain == 1) & (p == 1))
  disp('Constrained model, |a| < 1');
end;
  
[LZ,DZ] = size(Z);

if (LZ < DZ)
  error('Length < dimension of Z');
end;

ALPHA = 0.9;      % smoother for W, V estimates
SEG = p;                    % for moving along the time series in taps
p = p*DZ;                    % as we will create a tap line from 2 time series
SHIFT = 1;				% move on one timestep at a time

W = eye(p);				% unit indep to start
y_pred = zeros(size(X));			% output same size
E = zeros(size(X));
A = ones(size(X,1),p)/p;			% average first p points to start with
N = round(size(X,1)/SHIFT - SEG/SHIFT);                         % number of time windows
D = size(X,2);				% dimension of prediction signals
a = ones(p,1)/p;				% model parameters - prior mean for a(0)
C = eye(p);					% initial covariance for a(0)
Ct = zeros(size(X,1),p,p);
Vt = zeros(size(X,1),1);
Tt = zeros(size(X,1),1);
T = 0;					% test statistic on residual
sigma_wu = 0;                           % weight uncertainty

if (V < 0), V = abs(V); V_ADAPT = 0;
else V_ADAPT = 1;
end;

Vt(1,1) = V; % first point

% now normalise the data
Zorig = Z;
Z = normalis(Z,Z);
Xorig = X;
X = normalis(X,X);

for t = 1:N
  n = (t-1)*SHIFT + SEG + 1;			% one-step predictor
  F = [];
  for dz = 1:DZ
    F = [F;Z(1+(t-1)*SHIFT:(t-1)*SHIFT + SEG,dz)];
                                % p-past samples - relates obsn. to state
  end;
  y = X(n,:);					% one step predictor
  R = C + W;				% covariance of prior P(a|D_t-1)
  y_pred(n,:) = -a'*F;				% make one-step forecast, mean y_pred, cov E 
  Q = F'*R*F + V;				% covariance of posterior P(Y_t|D_t-1)
  Q_W0 = F'*C*F + V;				% assuming there is no state noise added
  K = R*F/Q;				% Kalman gain factor
  a = a + K.*(y_pred(n,:)-y)';			% posterior mean of P(a|D_t)
  C = R - K*F'*R;				% posterior cov of P(a|D_t)
  e = (y_pred(n,:)-y)'*(y_pred(n,:)-y);		% error residual
  if (F'*F > 0)
    sigma_wu = F'*R*F;				% pred error due to
                                                % weight uncertainty
    T = T*ALPHA + (1-ALPHA)*(e - Q_W0)/(F'*F);	% infer the system (parameter) noise level
  end;
  T = max(T,0);
  W = T*eye(p);
  if ( ((e - sigma_wu) > 0) & (V_ADAPT == 1) )
    V = ALPHA*V + (1-ALPHA) * (e - sigma_wu);	% infer observation noise level
  end;
  Vt(n,1) = V;
  Tt(n,1) = T;
  
  if ((constrain == 1) & (SEG == 1)) % constrain  models to stable system
    fp = find(a>1); a(fp) = 2-a(fp); fm = find(a<-1); a(fm) = -2-a(fm);
  end;
  A(n,:) = a';
  Ey(n,:) = reshape(Q,1,size(Q,1)*size(Q,2));
  Ea(n,:) = reshape(diag(C),1,p);
  Ct(n,:,:) = C;
  P(n) = gaussres(y,y_pred(n,:),Q) * sqrt(det(C)/det(R));
  if (rem(t,100)==0)
    fprintf('.');
  end;
end;

P = P';
fprintf('\n');

% now un-normalise the output
y_pred = unnorm(y_pred,Xorig);
Ey = sqrt(Ey*var(Xorig));

Ea = sqrt(Ea);

