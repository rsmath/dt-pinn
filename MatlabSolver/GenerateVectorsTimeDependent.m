%% Generates matrices and vectors for solution improvement problem
%% Also generates alpha, beta, f, g

warning('off','MATLAB:nearlySingularMatrix');


%% Set spatial dimension. For now, code only handles 2 and 3
s_dim = 2; 

%% Set time and timestep
initial_time = 0.0;
final_time = 1.0;
Nt = 24; %Nt+1 including time 0
dt = (final_time - initial_time)/Nt;

%% Set the type of boundary condition.
%%For now, don't do Neumann.
bctype = 3; %1-Neumann,2-Dirichlet, 3- Robin


%% Make up functions for a solution u, f, and g
if s_dim==2
    syms x y t;
    syms nrx nry;    
    u = 1 + sin(pi*x).*cos(pi*y).*sin(pi*t); % change this to change solutions
    if bctype==2
        g = u;
    elseif bctype==1
        g = (nrx.*diff(u,x) + nry.*diff(u,y)); 
    elseif bctype==3
        g = (nrx.*diff(u,x) + nry.*diff(u,y)) + u;
    end    
    f = simplify(diff(u,t) - (diff(u,x,2) + diff(u,y,2)));
    u_func = matlabFunction(u);
    f_func = matlabFunction(f);
    g_func = matlabFunction(g);
    if bctype==2
       g_func = @(nrx,nry,t,x,y) g_func(t,x,y);
    end
    clear x y t nrx nry u f g u;
elseif s_dim==3
    syms x y z t;
    syms nrx nry nrz;    
    u = 1 + sin(pi*x).*cos(pi*y).*cos(pi*z).*sin(pi*t); % change this to change solutions
    if bctype==2
        g = u;
    elseif bctype==1
        g = (nrx.*diff(u,x) + nry.*diff(u,y) + nrz.*diff(u,z)); 
    elseif bctype==3
        g = (nrx.*diff(u,x) + nry.*diff(u,y) + nrz.*diff(u,z))  + u;
    end    
    f = simplify(diff(u,t) - (diff(u,x,2) + diff(u,y,2) + diff(u,z,2)));
    u_func = matlabFunction(u);
    f_func = matlabFunction(f);
    g_func = matlabFunction(g);
    if bctype==2
       g_func = @(nrx,nry,nrz,x,y,z) g_func(x,y,z);
    end
    clear x y z t nrx nry nrz u f g u;    
end

%% Select the node set. 
if s_dim==2
    load('DiskPoissonNodes.mat');
elseif s_dim==3
    load('SpherePoissonNodes.mat');
end

if s_dim==2
    k = 2;
elseif s_dim==3
    k=1;
end
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

%% Loop over time, for each time level evaluate solution at all spatial points in order Xi,Xb,Xg
u = []; f = []; g = [];
if s_dim==2
    for nt=0:Nt %Nt+1 steps including step 0. step 0 is always given
        u = [u; u_func(nt*dt,Om.X(:,1),Om.X(:,2))];
        f = [f; f_func(nt*dt,Om.X(:,1),Om.X(:,2))];
        g = [g; g_func(Om.nr(:,1),Om.nr(:,2),nt*dt,Om.Xb(:,1),Om.Xb(:,2))];
    end
elseif s_dim==3
    for nt=0:Nt %Nt+1 steps including step 0. step 0 is always given
        u = [u; u_func(nt*dt,Om.X(:,1),Om.X(:,2),Om.X(:,3))];
        f = [f; f_func(nt*dt,Om.X(:,1),Om.X(:,2),Om.X(:,3))];
        g = [g; g_func(Om.nr(:,1),Om.nr(:,2),Om.nr(:,3),nt*dt,Om.Xb(:,1),Om.Xb(:,2),Om.Xb(:,3))];
    end        
end

orders = [2, 3, 4, 5];
for order_idx = 1: length(orders)
order = orders(order_idx);
folder_name = strcat('../scai/files_', num2str(order), '_', num2str(training_size));
mkdir(folder_name);

save(strcat(folder_name, '/', 'u_heat.mat'), 'u');
save(strcat(folder_name, '/', 'f_heat.mat'), 'f');
save(strcat(folder_name, '/', 'g_heat.mat'), 'g');

end
