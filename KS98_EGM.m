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
global alpha delta At agrid y num_grid ut tol_v beta PP ny marg_ut inv_marg_ut

% utility
gamma = 2;                              % risk aversion
ut   = @(c) (c.^(1-gamma))/(1-gamma);   % flow utility: CRRA
marg_ut = @(c) c.^(-gamma);             % marginal utility: CRRA
inv_marg_ut = @(u) u.^(-1/gamma);       % inverse marginal utility: CRRA

% household
a_lbar  = 0;      % borrowing constriant
a_bar   = 25;     % maximum asset
num_grid  = 100;  % number of grid for asset
%agrid = getGrid(a_lbar, a_bar, num_grid, 'logsteps'); % log spaced asset grid
agrid = linspace(a_lbar, a_bar, num_grid); % log spaced asset grid
tol_v   = 1e-06;  % value function tolerance
clear a_bar

% income process
rho   = 0.95;         % AR1 persistence
ny    = 3;            % size of income grid
sigma = sqrt(0.015);  % AR1 standard deviation
w = 0.5 + rho/4;      % weight
sigma_Z = sigma/sqrt(1-rho^2);
base_sigma = w*sigma + (1-w)*sigma_Z;
[sj,P_S] = tauchenHussey(ny,1,rho,sigma,base_sigma); % Markov appox, assume mean zero
y = exp(sj)';         % permanent income process: 1*5

clear w sigma_Z base_sigma

% firm side with technology shock
delta = 0.06;         % depreciation
alpha = 0.36;         % factor share of capital
rho_A = 0.90;         % AR1 persistence

nA    = 2;            % size of technology grid
sigma_A = 0.00;        % std of firm AR1, can be 0.012 or 0
w = 0.5 + rho_A/4;    % weight
sigma_Z = sigma_A/sqrt(1-rho_A^2);
base_sigma = w*sigma_A + (1-w)*sigma_Z;
[logA,P_A] = tauchenHussey(nA,-sigma_A^2/2,rho_A,sigma_A,base_sigma); % Markov appox, assume mean zero
At = exp(logA);      % productivity process
clear w sigma_Z base_sigma sj logA

if unique(At) == 1
    fprintf("no technology shock...");
    P_A = 1;
    At = 1;
else        
    fprintf("add technology shock...");
end

% joint transition matrix of A and S
PP = kron(P_A, P_S);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Q1: find beta
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%fprintf("\n %.2f",y);
tic;
syms beta % LHS is k_ss, RHS from firms and hh stead state condition
eqn = 2.5^(1/(1-alpha)) == ((1/beta - 1 + delta) / alpha)^(1/(alpha-1));
beta_single = double(solve(eqn, beta)); % solve the expression as beta = ...
r_single = 1/beta_single - 1 + delta;
fprintf("\n single agent economy: discount factor = %.2f, interest rate = %.3f \n",beta_single, r_single);

% given steady state captial, find discount factor
K_ss = 2.5^(1/(1-alpha));
options = optimset('TolX',1e-06);

% before we move to optimze, try some value first:
findK(K_ss,0.91)

%beta_final = fzero(@(beta) findK(K_ss,beta), beta_single, options);
%[~,lambda,a_plus,r_final] =findK(K_ss,beta_final);
%fprintf("\n heterogenuous agent economy: discount factor = %.2f, interest rate = %.3f \n",beta_final, r_final);

toc;
% state variables: asset and income state

function [NetAsset, lambda, a_plus, r] = findK(K, beta)
global alpha delta At agrid y num_grid marg_ut inv_marg_ut tol_v PP ny

% set capital level and time endowment
L = 1;

w = (1-alpha)*At.*(K/L).^(alpha);
r = (alpha)*At.*(K/L).^(alpha-1) - delta;

% household problem

[Amat,Ymat] = ndgrid(agrid,y);               % state space
% Amat is next period choice of asset at 5 different states

% solve from backward: initial guess of consumption
sol.c =  Amat + .01;
sol.x =  Amat + .01;

distance = inf;
iter = 0;
while distance > tol_v
    iter = iter+1;
    %Amat_old = Amat;
    x0 = sol.x;
    % next period cash on hand at 5 different states at t+1
    x_plus = Amat*(1+r) + w*Ymat;
    
    c_plus_interp = cell(1,ny);
    for s = 1:ny
    % create interpolant across income states
    c_plus_interp{s} = griddedInterpolant([0;sol.x(:,s)],[0;sol.c(:,s)],'linear','linear');
    
    % check life cycle model here
    % next period marginal utility at state s_t+1
    marg_u_plus = marg_ut(c_plus_interp{s}(x_plus(:))); %(1000*10) reshape to column vector
    marg_u_plus = reshape(marg_u_plus,size(x_plus)); %reshape back
    
    % take expectation over shocks
    avg_marg_u_plus = marg_u_plus*PP(s,:)'; %take expectation over tomorrow's states, column became today's state
 
    % Euler -> c_t, x_t, a_t
    assert(all(sol.c,'all'),'negative consumption!')
    sol.c(:,s) = inv_marg_ut(beta*(1+r)*avg_marg_u_plus);
    sol.x(:,s) = agrid' + sol.c(:,s);
    sol.Amat0(:,s) = (sol.x(:,s)-w*y(s))/(1+r); % can take outside

    end
    distance = max( abs(x0 - sol.x),[],"all");
    %distance = max( abs(Amat_old - Amat),[],"all")
end
fprintf('convergence at iteration %.0f', iter);

% policy function is sol.Amat0 -> sol.c
% add zero point
sol.c = [zeros(1,ny); sol.c];
sol.Amat0 = max(sol.Amat0,0); %max([zeros(1,ny); sol.Amat0],0);
sol.Amat1 = Amat; %[zeros(1,ny);Amat]; % extended asset grid


%create interpolents - maps a_t to a_t+1 and new policy function
for s = 1:ny
% unique Amat
[Amat00,ind] = unique(sol.Amat0(:,s));
Amat11 = sol.Amat1(ind,s);

g{s} = griddedInterpolant(Amat00,Amat11,'linear','linear');
sol.Amat1_new(:,s) = max(g{s}(agrid),0);
% dsearchn find nearest grid index in the first argument
index(:,s) = dsearchn(agrid',sol.Amat1_new(:,s));
a_plus(:,s) = agrid(index(:,s)); % a_plus becomes new, nearest grided policy function
end

%plot(sol.Amat0(2:end,1),Amat(:,1))

a_plus = a_plus(:); % reshape to 500*1
% find stationary distribution of (a,s)

ns  = numel(Amat); % 500, total state number
A_aug  = repmat(agrid,[ns,ny]); % augmented asset grid, 500*500
A   = (A_aug == a_plus([1:ns])); % 500*500, augmented income transition matrix
PPP = kron(PP,ones(num_grid));  % 500*500, augmented prob transition matrix
Q   = A.*PPP; % full transition matrix

[eigV,eigD] = eig(Q'); % Find eigenvectors and eigenvalues
i_eig1 = dsearchn(diag(eigD),1); % Find first unit eigenvalue
lambda = eigV(:,i_eig1);
lambda = lambda/sum(lambda); % Check distribution is appropriately normalised.


% check market clearing
Asset = sum(lambda.*a_plus);
NetAsset = K - Asset;

fprintf('value function iterations = %d,  ', iter)
fprintf('Net asset = %f\n', NetAsset)
end
