%% Generates matrices and vectors for solution improvement problem
%% Also generates alpha, beta, f, g

warning('off','MATLAB:nearlySingularMatrix');


%% Set spatial dimension. For now, code only handles 2
s_dim = 2; 

%% RBF-FD parameters for lower and higher order methods
xi1 = 1; %lower-order method;
lap_params1 = rbffdop(s_dim,xi1,2,0);
grad_params1 = rbffdop(s_dim,xi1,1,0);

xi2 = 3; %higher-order
lap_params2 = rbffdop(s_dim,xi2,2,0);
grad_params2 = rbffdop(s_dim,xi2,1,0);
hyp_params = rbffdop(s_dim,xi2,2,1);


%% Set the type of boundary condition.
%%For now, don't do Neumann.
bctype = 3; %1-Neumann,2-Dirichlet, 3- Robin

%% Make up functions for a solution u, and rhs f and bc g
syms x y;
syms nrx nry;
u = 1 + sin(pi*x).*cos(pi*y); % change this to change solutions
if bctype==2
    g = u;
elseif bctype==1
    g = (nrx.*diff(u,x) + nry.*diff(u,y)); 
elseif bctype==3
    g = (nrx.*diff(u,x) + nry.*diff(u,y)) + u;
end    
f = simplify(diff(u,x,2) + diff(u,y,2));
u_func = matlabFunction(u);
f_func = matlabFunction(f);
g_func = matlabFunction(g);
if bctype==2
   g_func = @(nrx,nry,x,y) g_func(x,y);
end
clear x y t nrx nry u f g u;



%% Select the node set. 
load('DiskPoissonNodes.mat');
k = 1;
Xi = fullintnodes{k};    
Xb = bdrynodes{k};       
n = normals{k};        


%% Set Ni,Nb, and use domain convenience class to create a domain Omega
%%that contains some data structures to find nearest neighbors, ghost
%%nodes, etc.
Ni = length(Xi);
Nb = length(Xb);
h = 1/sqrt(Ni + Nb);
Om = domain(Xi,Xb,n,h);

%% Set alpha and beta
if bctype==1        
    Neucoeff = ones(Om.Nb,1); %alpha
    Dircoeff = zeros(Om.Nb,1); %beta
elseif bctype==2
    Neucoeff = zeros(Om.Nb,1); %alpha
    Dircoeff = ones(Om.Nb,1);  %beta      
else
    Neucoeff = ones(Om.Nb,1); %alpha
    Dircoeff = ones(Om.Nb,1); %beta                 
end     

%% Build L1 and B1, solve for u1
[L1,~,~] = FormLaplacian(Om.X,lap_params1.rbf,lap_params1.drbfor,lap_params1.d2rbf,Om.tree,lap_params1.stencilSize,lap_params1.ell);    
[B1,~] = FormBC(Neucoeff,Dircoeff,Om.Xb,Om.nr,grad_params1.rbf,grad_params1.drbfor,Om.tree,grad_params1.stencilSize,grad_params1.ell);
f = f_func(Om.X(:,1),Om.X(:,2));
g = g_func(Om.nro(:,1),Om.nro(:,2),Om.Xb(:,1),Om.Xb(:,2));
u1 = [L1;B1]\[f;g];

%% build L2 and B2, solve for u2
[L2f,~,~] = FormLaplacian(Om.Xf,lap_params2.rbf,lap_params2.drbfor,lap_params2.d2rbf,Om.tree,lap_params2.stencilSize,lap_params2.ell); 
L2 = L2f(1:Ni+Nb,:);
[B2,~] = FormBC(Neucoeff,Dircoeff,Om.Xb,Om.nr,grad_params2.rbf,grad_params2.drbfor,Om.tree,grad_params2.stencilSize,grad_params2.ell);
u2 = [L2;B2]\[f;g];
 
%% Get the exact solution
u = u_func(Om.X(:,1),Om.X(:,2));    

%% Measure relative l2 errors in numerical solutions u1 and u2
relative_error_in_u1 = norm(u1(1:Ni+Nb) - u)./norm(u)
relative_error_in_u2 = norm(u2(1:Ni+Nb) - u)./norm(u)

%% Get the gradient operator and stabilize it with hyperviscosity
[Gx2,Gy2,~,~] = FormGradients(Om.Xf,grad_params2.rbf,grad_params2.drbfor,Om.tree,grad_params2.stencilSize,grad_params2.ell);
%[Hyp, ~] = FormHyp(Om.Xf,hyp_params.rbfexp,Om.tree,hyp_params.stencilSize,hyp_params.ell,hyp_params.hyppow);
Hyp = L2f^hyp_params.hyppow;
opts.tol=1e-3;
tau1 = real(eigs(Gx2,1,'LR',opts));
tau2 = real(eigs(Gy2,1,'LR',opts));
hx = 1/nthroot(Ni+2*Nb,s_dim);
[q1,~,~,~] = EstimateGrowth(Gx2,Om.Xf,tau1,hx);
[q2,~,~,~] = EstimateGrowth(Gy2,Om.Xf,tau2,hx);        
hyp_gamma_1 = (-1)^(1-hyp_params.hyppow)*(2^(q1 - 2*hyp_params.hyppow))*hx.^(2*hyp_params.hyppow-q1)*tau1;
hyp_gamma_2 = (-1)^(1-hyp_params.hyppow)*(2^(q2 - 2*hyp_params.hyppow))*hx.^(2*hyp_params.hyppow-q2)*tau2;                
Gx2 = Gx2 + hyp_gamma_1*Hyp;
Gy2 = Gy2 + hyp_gamma_2*Hyp;
Gx2 = Gx2(1:Ni+Nb,:); Gy2 = Gy2(1:Ni+Nb,:);

%% Save out the stuff we need
save('L1.mat','L1');
save('B1.mat','B1');
save('L2.mat','L2');
save('B2.mat','B2');
save('Gx2.mat','Gx2');
save('Gy2.mat','Gy2');
save('u1.mat','u1');
save('u2.mat','u2');
save('u.mat','u');
 
