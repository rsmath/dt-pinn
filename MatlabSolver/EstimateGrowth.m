function [ q,tau,q2,qm] = EstimateGrowth( G,X,tau,h)
%ESTIMATEGROWTH Estimates spurious growth q in gradient matrix G on node set
%X as a function of wave number p.
if h==[]
    nw = 10000;
else
    nw = 2/h;
end
%nw = 100000;
f = zeros(length(X),1);
gc = zeros(length(X),1);
if size(X,2) == 1        
        f(:,1) = exp(1i*nw*X(:,1));
        gc(:,1) = 1i*nw*exp(1i*nw*X(:,1));    
elseif size(X,2)==2    
        f(:,1) = exp(1i*nw*X(:,1)).*exp(1i*nw*X(:,2));
        gc(:,1) = 1i*nw*exp(1i*nw*X(:,1)).*exp(1i*nw*X(:,2));    
elseif size(X,2)==3    
        f(:,1) = exp(1i*nw*X(:,1)).*exp(1i*nw*X(:,2)).*exp(1i*nw*X(:,3));
        gc(:,1) = 1i*nw*exp(1i*nw*X(:,1)).*exp(1i*nw*X(:,2)).*exp(1i*nw*X(:,3));    
end
g = G*f;

q2 = (log(norm(g(:,1)-gc(:,1),2)) - log(tau) - log(norm(f(:,1),2)))./log(nw);
qm = (log(norm(g(:,1)-gc(:,1),inf)) - log(tau) - log(norm(f(:,1),inf)))./log(nw);
q = max([q2,qm]);
end

