%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% K-S98 & B-K-M18, QMM II PS1
% Yifan Lyu
% Stockholm School of Economics
% 2nd Nov, 2021
% solved with partially vectorised EGM
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear; clc; tic; warning off;
cd '/Users/frankdemacbookpro/Dropbox/SSE_yr2/QMMII/HW1'
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% set environment
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% utility
gamma = 2;                              % risk aversion
ut   = @(c) (c.^(1-gamma))/(1-gamma);   % flow utility: CRRA
marg_ut = @(c) c.^(-gamma);             % marginal utility: CRRA
inv_marg_ut = @(u) u.^(-1/gamma);       % inverse marginal utility: CRRA

% household
a_lbar  = 0;      % borrowing constriant
a_bar   = 20;     % maximum asset
num_grid  = 50;  % number of grid for asset
%agrid = getGrid(a_lbar, a_bar, num_grid, 'logsteps'); % log spaced asset grid
agrid = linspace(a_lbar, a_bar, num_grid); % log spaced asset grid
tol_v   = 1e-06;  % value function tolerance
clear a_bar

% income process
rho   = 0.95;         % AR1 persistence
ny    = 2;            % size of income grid
sigma = sqrt(0.015);  % AR1 standard deviation
w = 0.5 + rho/4;      % weight
sigma_Z = sigma/sqrt(1-rho^2);
base_sigma = w*sigma + (1-w)*sigma_Z;
[sj,P_S] = tauchenHussey(ny,1,rho,sigma,base_sigma); % Markov appox, assume mean zero
y = exp(sj)';         % permanent income process: 1*5
% replication
y = [0.05,1];

clear w sigma_Z base_sigma sj

% firm side with technology shock
delta = 0.06;         % depreciation
alpha = 0.36;         % factor share of capital
rho_A = 0.90;         % AR1 persistence

nA    = 2;            % size of technology grid
sigma_A = 0.012;        % std of firm AR1, can be 0.012 or 0
w = 0.5 + rho_A/4;    % weight
sigma_Z = sigma_A/sqrt(1-rho_A^2);
base_sigma = w*sigma_A + (1-w)*sigma_Z;
[logA,P_A] = tauchenHussey(nA,-sigma_A^2/2,rho_A,sigma_A,base_sigma); % Markov appox, assume mean zero
At = exp(logA);      % productivity process
% replication
At = [0.08; 1.02];
P_A = [1-1/8, 1/8; 1/8, 1-1/8];

clear w sigma_Z base_sigma sj logA

if unique(At) == 1
    fprintf("no technology shock...");
    P_A = eye(nA);
else        
    fprintf("add technology shock...");
end
%F = @(K,L) At*K.^alpha.*L.^(1-alpha); % production function

% joint transition matrix of A and S
PP = kron(P_A, P_S);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Q1: find beta
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%fprintf("\n %.2f",y);

syms beta % LHS is k_ss, RHS from firms and hh stead state condition
eqn = 2.5^(1/(1-alpha)) == ((1/beta - 1 + delta) / alpha)^(1/(alpha-1));
beta = double(solve(eqn, beta)); % solve the expression as beta = ...

fprintf("\n discount factor = %.2f \n",beta);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Q2: gini coefficient (under A shock) - household problem
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% guess a capital level: since time endowment is normalised to 1, K/L = K
nK = 10; % number of total capital grid
Kgrid = linspace(150,250,nK); % capital grid
L = 1; % time endowment
%w = (1-alpha)*At.*(Kgrid/L).^(alpha);
%r = (alpha)*At.*(Kgrid/L).^(alpha-1);
    % r = MPK - delta ??

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% start infinite period EGM by specifying next period asset grid
a1 = repmat(agrid,[nA*ny*nK,1]); % state is A*S*K

% iteration
distance  = inf;
tol_iter = 0;


while distance > tol_v

tol_iter = tol_iter + 1;
% guess next period Kprime by guessing regression coefficient B first
B = [zeros(nA,1), ones(nA,1)]; % nA*2 matrix of regressio coefficient
Kprime = exp(B(:,1) + B(:,2).*log(  repmat(Kgrid,[nA,1])  )); % na*length(Kgrid)

% obtain a new set of w(Kprime,At) and r(Kprice, At)
w1 = (1-alpha)*At.*(Kprime/L).^(alpha);
r1 = (alpha)*At.*(Kprime/L).^(alpha-1);



a0 = a1;
% for each of the capital level K_t, A, S, calculate c and a_t, and compare
% with the initial guess until distance is small enough
iter = 0;
marg_u_plus = nan(ny*nA*nK,num_grid);
for j = 1:nA
    for i = 1:ny
        for k = 1:nK
            iter = iter+1;
            % create interpolant (policy function) ,g
            % it is a mapping from a_t (which is a1) to a_t+1, given state A and S today
            [x,ia] = unique(a1(iter,:));
            g{j,i,k} = griddedInterpolant(x,agrid(ia),'linear','pchip');

            % w and r is from period t, w1 r1 from period t+1, (both scalar)
            %w = (1-alpha)*At(j)*(Kgrid(k)/L).^(alpha);
            %r = (alpha)*At(j)*(Kgrid(k)/L).^(alpha-1);
    
            % calculate marginal utility
            % note a_t+1 is agrid, a_t+2 is just g{i}(agrid)
            % the row of marg_u_plus is state A,S,K combination, column is
            % a grid
            marg_u_plus(iter,:) = marg_ut( agrid*(1+r1(j,k))...
                + y(i)*w1(j,k) - max(g{j,i,k}(agrid), a_lbar) );
            % A = 1, S = 1, K = 1
            % A = 1, S = 1, K = 2
            % ...
            % A = 1, S = 2, K = 1
            % A = 1, S = 2, K = 2
            % ...
            
        end
    end
end

% calculate ave_marg_plus by averaging transition prob
ave_marg_u_plus = nan(nA*ny,num_grid);
for l = 1:nA*ny
    ave_marg_u_plus(l,:) = kron(PP(l,:),ones(1,nK))*marg_u_plus;
end

iter = 0;
c = nan(nA*ny*nK,num_grid);
for j = 1:nA
    for i = 1:ny
        for k = 1:nK
            iter = iter + 1;
            c(iter,:) = inv_marg_ut(beta*(1+r1(j,k))*ave_marg_u_plus(i+ny*(j-1),:));
            
            % back out a_t using this period budget constraint
            % w and r is from period t, w1 r1 from period t+1, (both scalar)
            w = (1-alpha)*At(j)*(Kgrid(k)/L).^(alpha); % this is inefficient
            r = (alpha)*At(j)*(Kgrid(k)/L).^(alpha-1);

            a_temp = (c(iter,:)+agrid - y(i)*w)/(1+r);
            % no negative asset
            a_temp(a_temp<0) = 0;
            a1(iter,:) = a_temp;

        end
    end
end

distance = max( abs(a0 - a1),[],'all');
end

fprintf('convergence takes %.0f iterations, distance = %.9f',tol_iter, distance);


%plot
%plot(a1(11,:),c(11,:),'linewidth',1.5);

model.figplot(agrid, max(g{1,1,5}(agrid),a_lbar));
hold on;
model.figplot(agrid, max(g{1,2,5}(agrid),a_lbar));
model.figplot(agrid, max(g{2,1,5}(agrid),a_lbar));
model.figplot(agrid, max(g{2,2,5}(agrid),a_lbar));

%{
% possible combination of asset and income
[Amat,Ymat] = ndgrid(agrid,y);               % state space
Amat = Amat(:);  % compress Amat and Ymat into long vector, 1000*3 -> 3000*1
Ymat = Ymat(:);

% all possible consumption
c  = nan(numel(Amat),num_grid);

% generate all possible values of c, column is possible choice of a_t+1
% row is the current asset and income grid
for j = 1:numel(agrid)
c(:,j) = -agrid(j) + (1+r).*Amat + w*Ymat; 
end

% to negate the choice of negative consumption, add large panelty
neg_ind = (c<=0);

u = ut(c) - 10^10*neg_ind;

% initial guess of value function: 3000*1 vector
if isempty(V)
V       = log(Ymat);
end
%}





