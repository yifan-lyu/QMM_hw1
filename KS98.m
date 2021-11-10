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
a_bar   = 24;     % maximum asset
num_grid  = 50;  % number of grid for asset
%agrid = getGrid(a_lbar, a_bar, num_grid, 'logsteps'); % log spaced asset grid
agrid = linspace(a_lbar, a_bar, num_grid); % log spaced asset grid
tol_v   = 1e-06;  % value function tolerance
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
%fprintf("\n %.2f",y);

syms beta % LHS is k_ss, RHS from firms and hh stead state condition
eqn = 2.5^(1/(1-alpha)) == ((1/beta - 1 + delta) / alpha)^(1/(alpha-1));
beta = double(solve(eqn, beta)); % solve the expression as beta = ...

fprintf("\n discount factor = %.2f \n",beta);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Q2: gini coefficient in Aiyagari model
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% state variables: asset and income state


% set capital level and time endowment
K = 15;
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
sol.Amat0 = [zeros(1,ny); sol.Amat0];


% find stationary distribution - assume continuum of 1 agent
T = 10^4;  % number of periods
I = 10^4;  % number of individuals

% initialize variables - assume at t=1, income state = 1
sim.at      = [ones(I,1),zeros(I,T)];
sim.yt      = [y(1)*ones(I,1),zeros(I,T-1)];
sim.ct      = zeros(I,T);
rng(1234);
sim.rand    = rand(I,T);
sim.s   = [ones(I,1),nan(I,T-1)];

for t = 1:T
    
   if t == 1
        for i = 1:1
        AA(i,1) = indexfind(sim.rand(i,1),PP(1,:));
        end
   else
   %yt(:,t) = (s<=1/2).*y1+(s>1/2).*y2;  
   end
   %sim.at(:,t+1) = (1+r)*sim.at(:,t)+w*sim.yt(:,t)-sim.ct(:,t); 

end





%{
% stationary distribution
ns = numel(Amat); % 250
A = zeros(ns,num_grid); % row is combination of asset and income state
Q = zeros(ns,ns); % 250*250
PPP = kron(PP,ones(num_grid,1));
a_plus = Amat(:);
%%
for s=1:ns
    [~,index] = min( abs( sol.Amat0(:) - a_plus(s) ) ); % classify state
    A(s,:) = (sol.Amat0(:)==a_plus(s));  % puts a to 1 if a_plus =  future asset grid a
    Q(s,:) = kron(PPP(s,:),A(s,:)); % asset and income transition, 300*300
    % why? recall PP is 3*3000, A is 3000*100, so kron(PP(s,:),A(s,:)) is
    % 1*3000
end
%}

%{
for j = 1:numel(agrid)
c(:,j) = -agrid(j) + (1+r).*Amat + Ymat; 
end
% add panelty to negative consumption
neg_ind = (c<=0);
u = ut(c) - 10^10*neg_ind;
%}


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Q4: KS98 with productivity shock
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%{
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
            g{j,i,k} = griddedInterpolant(x,agrid(ia),'linear','linear');

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

%}
function index = indexfind(A,B)
B = cumsum(B);
if A < B(1)
    index = 1;
else
    find( A>B ,1) + 1;
    
end

end
