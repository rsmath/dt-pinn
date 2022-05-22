%% Generates matrices and vectors for solution improvement problem
%% Also generates alpha, beta, f, g

warning('off','MATLAB:nearlySingularMatrix');


%% Set spatial dimension. For now, code only handles 2 and 3
s_dim = 2; 

%% RBF-FD parameters for lower and higher order methods
orders = [2, 3, 4, 5];
for o = 1: length(orders)
xi1 = orders(o) %lower-order method;
lap_params1 = rbffdop(s_dim,xi1,2,0);
grad_params1 = rbffdop(s_dim,xi1,1,0);

%% Set the type of PDE
nonlinear = 1; %0 - linear Poisson, 1 - Poisson nonlinear source term


%% Set the type of boundary condition.
%%For now, don't do Neumann.
bctype = 3; %1-Neumann,2-Dirichlet, 3- Robin

%% Make up functions for a solution u, and rhs f and bc g
if s_dim==2
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
    if nonlinear==0
        save_string = "scai"
        f = simplify(diff(u,x,2) + diff(u,y,2));
    elseif nonlinear==1
        save_string = 'nonlinear'
        f = simplify((diff(u,x,2) + diff(u,y,2)) - exp(u));
    end
    u_func = matlabFunction(u);
    f_func = matlabFunction(f);
    g_func = matlabFunction(g);
    if bctype==2
       g_func = @(nrx,nry,x,y) g_func(x,y);
    end
    clear x y t nrx nry u f g u;
elseif s_dim==3
    syms x y z;
    syms nrx nry nrz;    
    u = 1 + sin(pi*x).*cos(pi*y).*cos(pi*z); % change this to change solutions
    if bctype==2
        g = u;
    elseif bctype==1
        g = (nrx.*diff(u,x) + nry.*diff(u,y) + nrz.*diff(u,z)); 
    elseif bctype==3
        g = (nrx.*diff(u,x) + nry.*diff(u,y) + nrz.*diff(u,z))  + u;
    end    
    f = simplify(diff(u,x,2) + diff(u,y,2) + diff(u,z,2));
    u_func = matlabFunction(u);
    f_func = matlabFunction(f);
    g_func = matlabFunction(g);
    if bctype==2
       g_func = @(nrx,nry,nrz,x,y,z) g_func(x,y,z);
    end
    clear x y z t nrx nry nrz u f g u;    
end

%% Select the node set. 
nodes = ["DiskPoissonNodes", "DiskPoissonNodesLarge"];

for nodeset = 1:2
cur_nodeset = nodes(nodeset);
if s_dim==2
    if cur_nodeset == "DiskPoissonNodes"
        N_idx = 5;
    else
        N_idx = 7;
    end

    load(cur_nodeset);
elseif s_dim==3
    load('SpherePoissonNodes.mat');
end
    for n_idx = 1: N_idx
    k = n_idx;
    Xi = fullintnodes{k};    
    Xb = bdrynodes{k};       
    n = normals{k};        


    %% Set Ni,Nb, and use domain convenience class to create a domain Omega
    %%that contains some data structures to find nearest neighbors, ghost
    %%nodes, etc.
    Ni = length(Xi)
    Nb = length(Xb)
    training_size = Ni + Nb;
    h = 1/nthroot(Ni + Nb,s_dim);
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
    if s_dim==2
        f = f_func(Om.X(:,1),Om.X(:,2));
        g = g_func(Om.nro(:,1),Om.nro(:,2),Om.Xb(:,1),Om.Xb(:,2));
        u = u_func(Om.X(:,1),Om.X(:,2));
    else
        f = f_func(Om.X(:,1),Om.X(:,2),Om.X(:,3));    
        g = g_func(Om.nro(:,1),Om.nro(:,2),Om.nro(:,3),Om.Xb(:,1),Om.Xb(:,2),Om.Xb(:,3));
        u = u_func(Om.X(:,1),Om.X(:,2),Om.X(:,3));
    end

    u1 = [L1;B1]\[f;g];

    %% Save out what we need

    X_g = Om.Xg;
    test_string = '';
    if training_size == 21748
        test_string = "_test";
    end

    folder_name = strcat('../', save_string, '/files_', num2str(xi1), '_', num2str(training_size), test_string);
    mkdir(folder_name);

    save(strcat(folder_name, '/', 'L1.mat'),'L1');
    save(strcat(folder_name, '/', 'B1.mat'),'B1');
    save(strcat(folder_name, '/', 'Xi.mat'),'Xi');
    save(strcat(folder_name, '/', 'Xb.mat'),'Xb');
    save(strcat(folder_name, '/', 'Xg.mat'), 'X_g');
    save(strcat(folder_name, '/', 'f.mat'), 'f');
    save(strcat(folder_name, '/', 'g.mat'), 'g');
    save(strcat(folder_name, '/', 'n.mat'),'n');
    save(strcat(folder_name, '/', 'alpha.mat'), 'Neucoeff');
    save(strcat(folder_name, '/', 'beta.mat'), 'Dircoeff');
    save(strcat(folder_name, '/', 'u.mat'),'u');

    end
end
end
