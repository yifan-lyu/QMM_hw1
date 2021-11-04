classdef model
    
    % load functions
    methods (Static)
        
        function EB = expect_bequest(p_b, miu_b, sigma_b)
        % This function use prob of receiving bequest, mean of bequest, and
        % std of bequest as input and returns the expected bequest, EB.
            
        [x,w] = model.GaussHermite(8, miu_b, sigma_b); % assume 8 quadrature point
             % or use directly the expectation formula: exp(miu + sigma^2/2)
             % reweight prob of receiving bequest - add extra grid:
        EB = sum([0;exp(x)] .* [p_b;(1-p_b)*w]); % expected bequest      
          
        end
        
       	function [x, w] = GaussHermite(n, mu, sigma)
        %----------------------------
        % GaussHermite Quadradrue, mu is mean and sigma is s.d. of normal
        % distributied error term. It returns quadrature point and weight
        % Note sum(x.*w) must be equal to 1
        % Example 1: calculate E(x^2), where x is normal
        % f = @(x) x.^2; mu = 0; sigma = 1;
        % [x,w] =  fun.GaussHermite(8, 0, 1)
        % sum(f(x).*w) = 1
        % Example 2
        % calculate E(e^error), where error is normal with mu=1 and
        % sigma=2, We know the resutls is exp(mu+sigma^2/2) by hand
        % [x,w] =  fun.GaussHermite(8, 1, 2)
        % sum(exp(x).*w) = 20.08, verified
        %----------------------------
        
        i   = 1:n-1;
        a   = sqrt(i/2);
        CM  = diag(a,1) + diag(a,-1);
        [V, L]   = eig(CM);
        [x, ind] = sort(diag(L));
        V       = V(:,ind)';
        w       = sqrt(pi) * V(:,1).^2;
        
        %adjust so that r.v. can be a generalised normal
        x  = x*sqrt(2)*sigma + mu;
        w  = w./sqrt(pi);
        assert(abs(sum(w.*x)) < abs(mu) +1e-8) %WHY? see note
        end
        
        function x = nonlinspace(lo,hi,n,phi)
        %column vector is the output
        % recursively constructs an unequally spaced grid.
        % phi > 1 -> more mass at the lower end of the grid.
        % lo can be a vector (x then becomes a matrix).

        x      = NaN(n,length(lo));
        x(1,:) = lo;
        for i = 2:n
            x(i,:) = x(i-1,:) + (hi-x(i-1,:))./((n-i+1)^phi);
        end
        end
        
        function figplot(x,y)
            plot(x,y,'-','linewidth',1.5);
            grid on; set(gca,'Fontsize',13); %axis equal;
            
        end
        
        function [mu_hat, mu_var] = GMM(moment,para0)
% By Yifan Lyu, August, 2021
% GMM standard procedure (Adda and Cooper (2003), p84)
% This program execute k steps GMM/ Simulated methods of moment
% Variance matrix calculated using numerical derivatives 
% of the simulated moments around the evaluation point (see robot paper p64)
%
% input--------------------
%   moment: functions that returns individual moment (demeaned, N by M)
%   para0: initial guess of parameters
%
% output-------------------
%   mu_hat: estimated coefficient
%   mu_var: variance covariance matrix of mu
%   
% set up ------------------
method = 1; % if 1, then use fminsearch, otherwise, quasi newton
iteration = 2; % number of iterations. 1: equal weight, 2: two steps GMM etc.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% first step solve

    [N,M] = size(moment(para0)); %number of obs and moment conditions.
    % N: obs
    % M: moment conditions
    assert(N>M,"number of observations must be greater than number of moments");
    
    W = eye(M); %first step weighting.
    % condense the individual moment to get M by 1 moment average
    avemom = @(beta) mean(moment(beta))'; % M by 1 matrix
    % the objective func to be minimised
    obj = @(beta) avemom(beta)'*W*avemom(beta); % 1 by 1 scalar
    
    %optimizer option
    options1 = optimset('MaxFunEvals',350*M,'MaxIter',350*M);
    options2 = optimoptions('fminunc','Display','none','Algorithm','quasi-newton',...
       'StepTolerance',1e-6,'FunctionTolerance',1e-6,'MaxIterations',500);
    
    if method == 1
        mu_hat = fminsearch(obj,para0,options1);
    else
        mu_hat = fminunc(obj,para0,options2);
    end
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% second step: (optional) update weighting matrix
for iter = 2:iteration
  
    moment_est = moment(mu_hat);
    S_hat = (1/N)*(moment_est'*moment_est); 
% S_hat is empirical variance of moment condition, size is M by M
% Optimal weighting matrix becomes inv(S_hat)
% define new objective function with new weight matrix
    obj = @(beta) avemom(beta)'*S_hat^-1*avemom(beta);

    if method == 1
        mu_hat = fminsearch(obj,para0,options1);
    else
        mu_hat = fminunc(obj,para0,options2);
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% use delta method -> get variance covariance matrix

% we need to get empirical Jacobian matrix first (no closed form sol to Jacobian)

try % if empirical variance already computed in second step
    lambda = S_hat;
catch %otherwise, compute S_hat
    lambda = (1/N)*moment(mu_hat)'*moment(mu_hat); % (M*N) * (N*M) -> M*M empirical variance of moment
end

dev = nan(numel(mu_hat),M); % K*M
gap = 1e-5; % precision of the derivatives, f'(x) = (f(x+gap) - f(x-gap))/(2*gap)
% warning: this way of getting jacobian is imprecise

for j = 1:numel(mu_hat)
% generate empirical jacobian, M*K matrix, K is the number of parameters
gap_adj = zeros(numel(mu_hat),1);
gap_adj(j) = gap;
dev(j,:) = mean(  moment(mu_hat+gap_adj) - moment(mu_hat-gap_adj)./(2*gap)  );
end

G_hat = dev';  % empirical Jacobian matrix, M*K  (keep consistent with note)
mu_var = (1/N)*(G_hat'*lambda^(-1)*G_hat)^(-1); % variance matrix 
% Note if it is Asyvar, then do not divide by N.

diag_mu_err = sqrt(diag(mu_var)); %standard error vector
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% print out standard error, return var-cov matrix
  for i = 1:numel(mu_hat)
      fprintf('estimated parameters (%1.0f) / standard error = %5.4f (%5.4f) \n', i, mu_hat(i), diag_mu_err(i));
  end
  
        end

        function Gini(y,S)
%calculate gini coefficient given y (income/wealth) and S (stationary dist)
%step1: generate cumulative share of people from low to high income
% By Yifan Lyu, April, 2021

y = sort(y); % sort wealth from low to high
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
%{
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
%}

%step6: calcualte gini coefficient, using intergation
gini = (0.5 - integral(Lorenz,0,1))/0.5; %triangular area is 0.5
fprintf('%8s = %5.3f \n','gini coefficient',gini);
fprintf('%8s = %5.3f, %5.3f, %5.3f \n','share of wealth at 20th, 5th and 1st '...
    ,Lorenz(1-0.2), Lorenz(1-0.05), Lorenz(1-0.01));
end

    end
end