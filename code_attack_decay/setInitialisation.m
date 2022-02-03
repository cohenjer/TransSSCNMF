function initialisation = setInitialisation(templates,X,H,parameters)

setGlobalDM();
global Tt

[F,T] = size(X);
R = parameters.R;

if isempty(templates)       
    initialisation.W = rand(F,R);
    initialisation.TS = ones(F,R);
    initialisation.a = ones(R,1);
    initialisation.pattern = ones(1,2*Tt+1);
else
    initialisation = templates;
end

if isempty(H)
    rng('default')
    rng(42);
    initialisation.H = rand(R,T);
else
    initialisation.H = H;
end

initialisation.R = parameters.R;
initialisation.update = parameters.update;
initialisation.sparsity = parameters.sparsity;
initialisation.beta = 1;
initialisation.iter = 50;
