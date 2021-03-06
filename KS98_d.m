%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% K-S98 & B-K-M18, QMM II PS1 - Q4
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
tol_v   = 1e-04;  % value function tolerance
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


clear w sigma_Z base_sigma sj

% firm side with technology shock
delta = 0.06;         % depreciation
alpha = 0.36;         % factor share of capital
rho_A = 0.90;         % AR1 persistence

nA    = 2;            % size of technology grid
sigma_A = 0.012;      % std of firm AR1, can be 0.012 or 0
w = 0.5 + rho_A/4;    % weight
sigma_Z = sigma_A/sqrt(1-rho_A^2);
base_sigma = w*sigma_A + (1-w)*sigma_Z;
[logA,P_A] = tauchenHussey(nA,-sigma_A^2/2,rho_A,sigma_A,base_sigma); % Markov appox, assume mean zero
At = exp(logA);      % productivity process


clear w sigma_Z base_sigma sj logA

if unique(At) == 1
    fprintf("no technology shock...");
    P_A = eye(nA);
else        
    fprintf("add technology shock...");
end

% joint transition matrix of A and S
PP = kron(P_A, P_S);

L = 1; % time endowment


rep = 0;
% replication
if rep == 1
y = [0.05,1];
At = [0.98; 1.02];
P_A = [1-1/8, 1/8; 1/8, 1-1/8];
PP = [0.525, 0.35, 0.03125, 0.09375;
      0.038889, 0.836111, 0.002083, 0.122917;
      0.09375, 0.03125, 0.291667, 0.583333;
      0.009115, 0.115885, 0.024306, 0.850694];
L = 0.2;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Q4: AK98
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
beta = 0.91339;
% guess a capital level: since time endowment is normalised to 1, K/L = K
nK = 20; % number of total capital grid
%Kgrid = linspace(150,250,nK); % capital grid
Kgrid = linspace(5,10,nK); % capital grid
% ! for very low capital, it does not converge!


%w = (1-alpha)*At.*(Kgrid/L).^(alpha);
%r = (alpha)*At.*(Kgrid/L).^(alpha-1) - delta;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% start infinite period EGM by specifying next period asset grid
a1 = repmat(agrid,[nA*ny*nK,1]); % state is A*S*K

% guess next period Kprime by guessing regression coefficient B first
%B = [zeros(nA,1), ones(nA,1)]; % nA*2 matrix of regressio coefficient
%[alpha_b, beta_b]
%[alpha_g, beta_g]
%B = [0.1242,  0.9657
%     0.135,  0.9638];
B = [0.053    0.968
     0.358    0.945];
Kprime = exp(B(:,1) + B(:,2).*log(  repmat(Kgrid,[nA,1])  )); % na*length(Kgrid)

% obtain a new set of w(Kprime,At) and r(Kprice, At)
w1 = (1-alpha)*At.*(Kprime/L).^(alpha);
r1 = (alpha)*At.*(Kprime/L).^(alpha-1) - delta;


% iteration
distance  = inf;
distance0 = inf;
tol_iter = 0;

while distance > tol_v
tol_iter = tol_iter + 1;
a0 = a1;
% NOTE: a0 and a1 are both this period asset
% agrid is next period asset


% for each of the capital level K_t, A, S, calculate c and a_t, and compare
% with the initial guess until distance is small enough
iter = 0;
c = nan(nA*ny*nK,num_grid);
marg_u_plus = nan(ny*nA*nK,num_grid);
g = cell(ny,nA,nK);
for i = 1:ny % income state
    for j = 1:nA % technology state
        for k = 1:nK % total capital state
            iter = iter+1;
            % create interpolant (policy function) ,g, and keep unique grid
            % it is a mapping from a_t (which is a1) to a_t+1, given state A and S today
            [x,ia] = unique(a1(iter,:));
            g{i,j,k} = griddedInterpolant(x,agrid(ia),'linear','linear');

            %g{i,j,k} = griddedInterpolant(a1(iter,:),agrid,'linear','linear');
        end
    end
end

iter = 0;
for i = 1:ny
    for j = 1:nA
        for k = 1:nK
            iter = iter+1;
            % w and r is from period t, w1 r1 from period t+1, (both scalar)
            %w = (1-alpha)*At(j)*(Kgrid(k)/L).^(alpha);
            %r = (alpha)*At(j)*(Kgrid(k)/L).^(alpha-1);
    
            % calculate marginal utility
            % note a_t+1 is agrid, a_t+2 is just g{i}(agrid)
            % the row of marg_u_plus is state A,S,K combination, column is
            % a grid
            marg_u_plus(iter,:) = marg_ut( agrid*(1+r1(j,k))...
                + y(i)*w1(j,k) - max(g{i,j,k}(agrid), a_lbar) );
            % S = 1, A = 1, K = 1
            % S = 1, A = 1, K = 2
            % ...
            % S = 1, A = 2, K = 1
            % S = 1, A = 2, K = 2
            % ...
            % back out a_t using this period budget constraint
            % w and r is from period t, w1 r1 from period t+1, (both scalar)
            w = (1-alpha)*At(j)*(Kgrid(k)/L).^(alpha); % this is inefficient
            r = (alpha)*At(j)*(Kgrid(k)/L).^(alpha-1) - delta;
            
            ave_marg_u_plus=0;
            for o = 1:nA*ny % state in tomorrow: 2*5
                ind1 = floor((o-1)/ny)+1; % 1,1,1,1,1,2,2,2,2,2 -> A_t+1
                ind2 = rem((o-1),ny)+1;   % 1,2,3,4,5,1,2,3,4,5 -> S_t+1
                c_plus = agrid*(1+r1(ind1,k)) + y(ind2)*w1(ind1,k) - max(g{ind2,ind1,k}(agrid),a_lbar);
                % pay particular attention to PP, wherein S state change first, 
                % then A state change after S state is exhausted
                ave_marg_u_plus = ave_marg_u_plus + PP(i+(j-1)*2,o)*marg_ut(c_plus);
            end

            c(iter,:) = inv_marg_ut(beta*(1+r1(j,k))*ave_marg_u_plus);
            a_temp = (c(iter,:)+ agrid - y(i)*w)/(1+r);
            % no negative asset
            % assert(all(a_temp>=0),'check a_temp');
            %a_temp(a_temp<0) = 0;
            a1(iter,:) = a_temp;
        end
    end
end

distance = max( abs(a0 - a1),[],'all');
%assert(distance0>distance,'VFI DOES NOT CONVERGE!');
%distance0 = distance;

if rem(tol_iter,50) == 0
    % report at every 100 iteration
    fprintf('number of iteration reached %.0f, distance = %.9f \n',tol_iter, distance);
end

end % end of the while loop
fprintf('convergence takes %.0f iterations, distance = %.9f \n',tol_iter, distance);

% recover policy function: a'{y,s,K}(agrid)
r = (alpha)*At.*(Kgrid/L).^(alpha-1) - delta;
w = (1-alpha)*At.*(Kgrid/L).^(alpha);

sol.a_plus = cell(ny,nA,nK);
sol.c      = cell(ny,nA,nK);
for i = 1:ny
    for j = 1:nA
        for k = 1:nK
sol.a_plus{i,j,k} = max(g{i,j,k}(agrid),a_lbar);
sol.c{i,j,k}      = w(j,k)*y(i) + (1+r(j,k))*agrid - sol.a_plus{i,j,k};
assert(all(sol.c{i,j,k}>=0),'consumption is negative!')
        end
    end
end


%plot
%{
model.figplot(agrid, sol.a_plus{1,1,4});
hold on;
model.figplot(agrid, sol.a_plus{1,2,4});
model.figplot(agrid, sol.a_plus{2,1,4});
model.figplot(agrid, sol.a_plus{2,2,4});
plot(agrid,agrid,'--b','linewidth',1);
legend('low income, A_b','low income, A_g','high income, A_b','high income, A_g');
title('optimal policy function under mean total capital');
%}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% simulation:
% guess a initial capital level
K_ss = ((1/beta - 1 + delta) / alpha)^(1/(alpha-1));
[~,sim.K_index] = min(abs(K_ss - Kgrid)); 

% find stationary distribution
% note that results sensitive for small T and I

T = 10^3;  % number of periods
I = 10^3;  % number of individuals
rng(123); % set seed

% initialize variables
sim.at      = [K_ss*ones(I,1),nan(I,T)];
sim.yt      = [y(2)*ones(I,1),nan(I,T-1)];
sim.ct      = nan(I,T);
sim.rand_A  = rand(1,T);
sim.rand_S  = rand(I,T);
sim.w       = nan(1,T);
sim.A       = nan(1,T); % state index of technology
sim.A(1) = sum(( sim.rand_A(1) > cumsum(P_A(1,:)) )) + 1;
sim.S       = nan(I,T); % state index of income
sim.S(:,1)  = kron((1:ny)',ones(I/ny,1));

sim.K       = [K_ss, nan(1,T-1)];
sim.r       = nan(1,T);
sim.r(1)    = (alpha)*At(1)*(sim.K(1)/L).^(alpha-1) - delta;
sim.w(1)    = (1-alpha)*At(1)*(sim.K(1)/L).^(alpha);


% for t = 1, classify state in At and St
for t = 1:T
sim.A(t+1) = sum(( sim.rand_A(t) > cumsum(P_A(sim.A(t),:)) )) + 1;
    for i = 1:I
    sim.S(i,t+1) = sum(( sim.rand_S(i,t) > cumsum(P_S(sim.S(i,t),:)) )) + 1;
    end
end

fprintf('simulation starts')
for t = 1:T
    for i = 1:I
    % update next period total capital
    
    sim.at(i,t+1) = max( g{sim.S(i,t),sim.A(t),sim.K_index(t)}(sim.at(i,t)), a_lbar);
    sim.yt(i,t+1) = y(sim.S(i,t+1));
    sim.ct(i,t)   = sim.w(t)*sim.yt(i,t) + (1+sim.r(t))*sim.at(i,t) - sim.at(i,t+1);
    end
    sim.K(t+1) = mean(sim.at(:,t+1));
    [~, sim.K_index(t+1)] = min(abs(sim.K(t+1) - Kgrid)); 
    sim.r(t+1) = (alpha)*At(sim.A(t+1))*(sim.K(t+1)/L).^(alpha-1) - delta;
    sim.w(t+1) = (1-alpha)*At(sim.A(t+1))*(sim.K(t+1)/L).^(alpha);
    
end
fprintf('\n simulation is over \n')

% plot aggregate capital time series
plot(1:length(sim.K), sim.K,'-b','linewidth',0.9);

% regression (only for 2 state in At)
% pick out good state and bad state index
good = (sim.A == 2);
good(end) = 0; % drop 1001 period obs.
good_plus = logical([0,good(1:end-1)]);

bad  = (sim.A == 1);
bad(end) = 0; % drop 1001 period obs.
bad_plus = logical([0,bad(1:end-1)]);

sim.Kgood = sim.K(good);
sim.Kgood_plus = sim.K(good_plus);
sim.Kbad = sim.K(bad);
sim.Kbad_plus = sim.K(bad_plus);

B_plus(:,2) = regress(sim.Kgood_plus',[ones(length(sim.Kgood),1),sim.Kgood']);
B_plus(:,1) = regress(sim.Kbad_plus',[ones(length(sim.Kbad),1),sim.Kbad']);
B_plus = B_plus';
distance = max( abs(B_plus - B) ,[],'all');

% we find that given the paramters, B (regression coefficient) is close
% enough (distance = around 0.01)

% only use stationary distribution result (t>300)
sta.ct = sim.ct(:,300:end);
sta.yt = sim.yt(:,300:end);
sta.at = sim.at(:,300:end);

% find Gini for asset (c - 0.240, y - 0.187, a - 0.784)
model.Gini(sta.ct(:)',ones(1,numel(sta.ct))/numel(sta.ct));
model.Gini(sta.yt(:)',ones(1,numel(sta.yt))/numel(sta.yt));
model.Gini(sta.at(:)',ones(1,numel(sta.at))/numel(sta.at));


