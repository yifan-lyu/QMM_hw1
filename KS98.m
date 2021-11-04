%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% K-S98 & B-K-M18, QMM II PS1
% Yifan Lyu
% Stockholm School of Economics
% 2nd Nov, 2021
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
a_bar   = 24;     % maximum asset
num_grid  = 100;  % number of grid for asset
agrid = getGrid(a_lbar, a_bar, num_grid, 'logsteps'); % log spaced asset grid
tol_v   = 1e-06;  % value function tolerance
clear a_lbar a_bar

% income process
rho   = 0.95;         % AR1 persistence
ny    = 5;            % size of income grid
sigma = sqrt(0.015);  % AR1 standard deviation
w = 0.5 + rho/4;      % weight
sigma_Z = sigma/sqrt(1-rho^2);
base_sigma = w*sigma + (1-w)*sigma_Z;
[sj,P] = tauchenHussey(ny,1,rho,sigma,base_sigma); % Markov appox, assume mean zero
y = exp(sj)';         % permanent income process: 1*5
clear w sigma_Z base_sigma sj

% firm side with technology shock
delta = 0.06;         % depreciation
alpha = 0.36;         % factor share of capital
rho_A = 0.90;         % AR1 persistence

nA    = 2;            % size of technology grid
sigma_A = 0.0;        % std of firm AR1, can be 0.012 or 0
w = 0.5 + rho_A/4;    % weight
sigma_Z = sigma_A/sqrt(1-rho_A^2);
base_sigma = w*sigma_A + (1-w)*sigma_Z;
[logA,P_A] = tauchenHussey(nA,-sigma_A^2/2,rho_A,sigma_A,base_sigma); % Markov appox, assume mean zero
At = exp(logA)';      % productivity process
clear w sigma_Z base_sigma sj logA
if unique(At) == 1
At = At(1); fprintf("no technology shock...");
else        
            fprintf("add technology shock...");
end
F = @(K,L) At*K.^alpha.*L.^(1-alpha); % production function


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Q1: find beta
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf("\n %.2f",y);

syms beta % LHS is k_ss, RHS from firms and hh stead state condition
eqn = 2.5^(1/(1-alpha)) == ((1/beta - 1 + delta) / alpha)^(1/(alpha-1));
beta = double(solve(eqn, beta)); % solve the expression as beta = ...

fprintf("\n discount factor = %.2f \n",beta);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Q2: gini coefficient
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% guess a capital level: since time endowment is normalised to 1, K/L = K
K = 30;
L = 1;
w = (1-alpha)*At*(K/L).^(alpha);
r = (alpha)*At*(K/L).^(alpha-1);

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



% nested value function iteration
iter = 0;
distance  = inf;


