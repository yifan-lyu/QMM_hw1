function [grid] = getGrid(minongrid, maxongrid, GridPoints, method)

% Acknowledgement: This code borrows some code from Chris Carroll of Johns 
% Hopkins University. His website is here
% http://www.econ2.jhu.edu/people/ccarroll/ and his notes on solving 
% dynamic models are very good - they can be found in
% http://www.econ2.jhu.edu/people/ccarroll/SolvingMicroDSOPs.pdf
%
% ------------------------------------------------------------------------ 
% DESCRIPTION
% This function constructs a grid from 'minongrid' to 'maxongrid' with 
% 'GridPoints' grid points and using 'method' from the list equalsteps, 
% logsteps, 3logsteps, 5logsteps or 10logsteps
%
% ------------------------------------------------------------------------ 
% EXPLANATION
% We need to construct a grid from a to b. A basic approach could involve 
% spacing out the points equally. The following line of code would achieve 
% this:
%     grid= linspace(a, b, GridPoints);
%
% We can space the grip points out so that the growth rate of the distance
% between spaces is equal by spacing out equally the grid points in logs.  
% We can do this as follows:
%     grid= exp(linspace(log(a), log(b), GridPoints));
%
% However, this does not work when a<=0. Our approach is to get a equally 
% spaced grid from log(1) to log(b - a + 1), exponentiate each point and 
% subtract 1 to form a grid from 0 to b-a. We then add a to each point to 
% obtain a log-spaced grid from a to b.

%% ------------------------------------------------------------------------ 

span = maxongrid - minongrid;     % b - a                  

if strcmp(method, 'equalsteps')
    grid= linspace(0, span, GridPoints);
    
elseif strcmp(method, 'logsteps')
  loggrid = linspace(log(1), log(1+span), GridPoints);
  grid = exp(loggrid)-1;
  
elseif strcmp(method, '3logsteps')
  loggrid = linspace(log(1+log(1+log(1))), log(1+log(1+log(1+span))), GridPoints);
  grid = exp(exp(exp(loggrid)-1)-1)-1;   
  
elseif strcmp(method, '5logsteps')
  loggrid = linspace(log(1+log(1+log(1+log(1+log(1))))), ...
                     log(1+log(1+log(1+log(1+log(1+span))))), GridPoints);
  grid = exp(exp(exp(exp(exp(loggrid)-1)-1)-1)-1)-1;   
  
elseif strcmp(method, '10logsteps')
  loggrid = linspace(log(1+log(1+log(1+log(1+log(1+log(1+log(1+log(1+log(1+log(1)))))))))), ...
                     log(1+log(1+log(1+log(1+log(1+log(1+log(1+log(1+log(1+log(1+span)))))))))), GridPoints);
  grid = exp(exp(exp(exp(exp(exp(exp(exp(exp(exp(loggrid)-1)-1)-1)-1)-1)-1)-1)-1)-1)-1;   
  
else
    error('Error in getgrid. You have entered an invalid method for choosing the distance between grid points. Method must be one of equalsteps, logsteps, 3logsteps, 5logsteps or 10logsteps.');
end

grid = grid + minongrid*ones(1, GridPoints);

end

