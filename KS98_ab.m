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
global alpha delta At agrid y num_grid ut tol_v beta PP ny
% utility
gamma = 2;                              % risk aversion
ut   = @(c) (c.^(1-gamma))/(1-gamma);   % flow utility: CRRA
marg_ut = @(c) c.^(-gamma);             % marginal utility: CRRA
inv_marg_ut = @(u) u.^(-1/gamma);       % inverse marginal utility: CRRA

% household
a_lbar  = 0;      % borrowing constriant
a_bar   = 24;     % maximum asset
num_grid  = 300;  % number of grid for asset
%agrid = getGrid(a_lbar, a_bar, num_grid, 'logsteps'); % log spaced asset grid
agrid = linspace(a_lbar, a_bar, num_grid); % log spaced asset grid
tol_v   = 1e-08;  % value function tolerance
clear a_bar

% income process
rho   = 0.95;         % AR1 persistence
ny    = 5;            % size of income grid
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
%F = @(K,L) At*K.^alpha.*L.^(1-alpha); % production function

% joint transition matrix of A and S
PP = kron(P_A, P_S);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Q1: find beta
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% first, we find beta and r in a representative agent economy,
% and compare it with heterogenuous agent, make sure that r_rep > r_het
tic;
syms beta % LHS is k_ss, RHS from firms and hh stead state condition
eqn = 2.5^(1/(1-alpha)) == ((1/beta - 1 + delta) / alpha)^(1/(alpha-1));
beta_single = double(solve(eqn, beta)); % solve the expression as beta = ...
r_single = 1/beta_single - 1 + delta;
fprintf("\n single agent economy: discount factor = %.2f, interest rate = %.3f \n",beta_single, r_single);

% given steady state captial, find discount factor
K_ss = 2.5^(1/(1-alpha));
options = optimset('TolX',1e-06);
beta_final = fzero(@(beta) findK(K_ss,beta), beta_single, options);
[~,lambda,a_plus,r_final] =findK(K_ss,beta_final);
fprintf("\n heterogenuous agent economy: discount factor = %.2f, interest rate = %.3f \n",beta_final, r_final);

toc;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Q2: gini coefficient in Aiyagari model
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Gini(a_plus',lambda')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Q3: AKM 18
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function [NetAsset, lambda, a_plus, r] = findK(K, beta)
global alpha delta At agrid y num_grid ut tol_v PP ny

L = 1;
w = (1-alpha)*At.*(K/L).^(alpha);
r = (alpha)*At.*(K/L).^(alpha-1) - delta;

% household problem

[Amat,Ymat] = ndgrid(agrid,y);               % state space
% Amat is next period choice of asset at 5 different states
Amat = Amat(:);
Ymat = Ymat(:);

% all possible consumption
c  = nan(numel(Amat),num_grid);
% generate all possible values of c, column is possible choice of a_t+1

for j = 1:numel(agrid)
c(:,j) = -agrid(j) + (1+r).*Amat + Ymat; 
end
% assert(c>=0,'consumption is negeative!')
neg_ind = (c<=0);

% to negate the choice of negative consumption, add large panelty
u = ut(c) - 10^10*neg_ind;

% initial guess of value function:
if ~exist('V','var')
V       = log(Ymat);
end

% nested value function iteration
iter = 0;
distance  = inf;

while (distance > tol_v)

% kron is a 3000*3 matrix, with 1000 repetition of P, 
% reshape()' is a 3 * 1000 matrix
RHS   = u + beta*kron(PP,ones(num_grid,1))*reshape(V,num_grid,[])';
[V_plus,argmax] = max(RHS,[],2); % find max in second dimention, i.e. a'
distance  = max( abs(V - V_plus) ); % note: use norm would be computationally costly
iter = iter + 1;
V = V_plus;
end

% solve from backward: initial guess of consumption

% since argmax is an index, we can find policy functions:
a_plus = Amat(argmax); % has negative numbers
c_opt  = (1+r)*Amat + Ymat - a_plus;

% given optimal policy, compute stationary distribution: lambda = lambda*P

method = 1;
% method 1 (from Kieran)
if method == 1
Q = zeros(ny*num_grid);
a_eye = eye(num_grid);
for j=1:ny
    for l=1:ny
    index1 = (j-1)*num_grid+1:j*num_grid;
    index2 = (l-1)*num_grid+1:l*num_grid;
    Q(index1,index2) = PP(j,l)*a_eye(argmax(index1),:);
    end
end


elseif method == 2
% method 2: 
ns = numel(Amat); % 300
A = zeros(ns,num_grid);
Q = zeros(ns,ns); %300*300
PPP = kron(PP,ones(num_grid,1)); % 300*3

for s=1:ns
    A(s,:) = (agrid==a_plus(s));  % puts a to 1 if a_plus =  future asset grid a
    Q(s,:) = kron(PPP(s,:),A(s,:)); % asset and income transition, 300*300
    % why? recall PP is 3*3000, A is 300*100, so kron(PP(s,:),A(s,:)) is
    % 1*300
end

end

[eig_vectors,eig_values] = eig(Q');
[~,arg] = min(abs(diag(eig_values)-1)); 
unit_eig_vector = eig_vectors(:,arg); 
lambda = unit_eig_vector/sum(unit_eig_vector); 

% method 2
%[eigV,eigD] = eig(Q'); % Find eigenvectors and eigenvalues
%i_eig1 = dsearchn(diag(eigD),1); % Find first unit eigenvalue
%lambda = eigV(:,i_eig1);
%lambda = lambda/sum(lambda); % Check distribution is appropriately normalised.

% check market clearing
Asset = sum(lambda.*a_plus);
NetAsset = K - Asset;

fprintf('value function iterations = %d,  ', iter)
fprintf('Net asset = %f\n', NetAsset)
end

function Gini(y,S)
%calculate gini coefficient given y (income/wealth) and S (stationary dist)
%step1: generate cumulative share of people from low to high income
% By Yifan Lyu, April, 2021

y = sort(y); % sort wealth from low to high
S = sort(S);

cumu_x = cumsum(S);

%step2: generate cumulative income, a scalar
cumu_y = y*S';

%step3: generate cumulative share of income
cumu_sy = y.*S;
cumu_sy = cumsum(cumu_sy)/cumu_y;

%step4: given we have continuum of agents, interpolate the fitted line
Lorenz = @(x) interp1([0,cumu_x],[0,cumu_sy],x,'linear'); % x is the query point

%step5: plot Lorenz curve, make sure correct
x = linspace(0,1,1000);

figure;
plot(x,Lorenz(x),'linewidth',1.5);
hold on
plot([0,1],[0,1],'linewidth',1.5)
legend('Lorenz Curve','45 degree line');
grid on;
title 'Lorenz Curve';
set(gcf,'PaperPosition',[0 0 30 20]);
set(gcf,'PaperSize',[30 20])
set(gca, 'FontName', 'times')
xlabel('people by percentile of weath holding');
ylabel('cumulative share of wealth');
%print('-dpdf',['/Users/frankdemacbookpro/Dropbox/SSE/Macro II/HW5/giniplot.pdf']);


%step6: calcualte gini coefficient, using intergation
gini = (0.5 - integral(Lorenz,0,1))/0.5; %triangular area is 0.5
fprintf('%8s = %5.3f \n','gini coefficient',gini);
%fprintf('%8s = %5.3f, %5.3f, %5.3f \n','share of wealth at 20th, 5th and 1st '...
%    ,Lorenz(1-0.2), Lorenz(1-0.05), Lorenz(1-0.01));
end

