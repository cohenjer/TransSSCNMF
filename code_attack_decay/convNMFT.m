function result = convNMFT(X, initialisation)
% X: spectrogram

W = initialisation.W; % harmonic templates
TS = initialisation.TS; % percussive templates
a = initialisation.a; % decay rates
H = initialisation.H; % activations

pattern = initialisation.pattern; % transient pattern
R = initialisation.R; %number of pitches
update = initialisation.update; % update flags for [W,TS,a,H,pattern]
sparsity = initialisation.sparsity; % control the sparseness of H
beta = initialisation.beta; % KL-divergence
iter = initialisation.iter; % iteration number

setGlobalDM();
global Wt Tt Tmax

% declear variables
T = size(X,2);
% ?????
Tmax = T;
ea = zeros(R,Tmax);
eat = zeros(R,Tmax);
Hea = zeros(R,T);
Heat = zeros(R,T);
WVXup = zeros(R,T);
WVdown = zeros(R,T);
TVXup = zeros(R,T);
TVdown = zeros(R,T);
Pup = zeros(1,2*Tt+1);
Pdown = zeros(1,2*Tt+1);

% reconstruction of V

t = 0:Tmax-1;
for r = 1:R
    ea(r,:) = exp(-a(r)*t);
end
Ea = [zeros(R,Wt),ea(:,1:end-Wt),zeros(R,T-Tmax)] + eps;
for t = 1:T
    Hea(:,t) = sum(H(:,1:t) .* Ea(:,t:-1:1),2);
end
Hs = conv2(H,pattern,'same');
V = W*Hea + TS*Hs + eps;

% update flag
updateW = update(1);
updateTS = update(2);
updatea = update(3);
updateH = update(4);
updateP = update(5);

spar = ones(iter,1);
if length(sparsity)==1
    spar(:) = sparsity;
elseif length(sparsity)==2
    spar = sparsity(1) + (sparsity(2)-sparsity(1))*(1:iter)/iter;
end


for it = 1:iter
    
    if updateW
        % update Wfr
        W = W.*(V.^(beta-2).*X * Hea') ./ (V.^(beta-1) * Hea')+eps;
        % update V
        V = W*Hea + TS*Hs + eps;
    end
    
    if updateTS
        % update TSfr
        TS = TS.*(V.^(beta-2).*X * Hs') ./ (V.^(beta-1) * Hs')+eps;
        % update V
        V = W*Hea + TS*Hs + eps;
    end
    
    if updatea
        % update ar
        t= 0:Tmax-1;
        for r = 1:R
            eat(r,:) = exp(-a(r)*t).*t;
        end
        Eat = [zeros(R,Wt),eat(:,1:end-Wt),zeros(R,T-Tmax)] + eps;
        
        for t = 1:T
            Heat(:,t) = sum(H(:,1:t) .* Eat(:,t:-1:1),2);
        end
        
        for r = 1:R
            a(r) = a(r) * (W(:,r)'*V.^(beta-1)*Heat(r,:)') / (W(:,r)'*(V.^(beta-2).*X)*Heat(r,:)');
        end
        
        % update V
        t= 0:Tmax-1;
        for r = 1:R
            ea(r,:) = exp(-a(r)*t);
        end
        Ea = [zeros(R,Wt),ea(:,1:end-Wt),zeros(R,T-Tmax)] + eps;
        
        for t = 1:T
            Hea(:,t) = sum(H(:,1:t) .* Ea(:,t:-1:1),2);
        end
        
        V = W*Hea + TS*Hs + eps;
    end
    
    if updateH
        % update Hrt
        WVX = [W'* (V.^(beta-2).*X) zeros(R,T)];
        WV = [W'* V.^(beta-1) zeros(R,T)];

        for t = 1:T
            WVXup(:,t) = sum(WVX(:,t+Wt:t+Tmax-1) .* ea(:,1:Tmax-Wt),2);
            WVdown(:,t) = sum(WV(:,t+Wt:t+Tmax-1) .* ea(:,1:Tmax-Wt),2);
        end
        
        TVX = [zeros(r,Tt) TS'*(V.^(beta-2).*X) zeros(r,Tt)];
        TV = [zeros(r,Tt) TS'*V.^(beta-1) zeros(r,Tt)];
        
        for t = 1:T
            TVXup(:,t) = TVX(:,t:t+2*Tt)*pattern';
            TVdown(:,t) = TV(:,t:t+2*Tt)*pattern';
        end
        
        H = H.*(WVXup + TVXup) ./ (WVdown + TVdown);
        H = H.^spar(it);
        
        % normalise
        if R ==1
            H = H./max(max(H));
        end
        
        % update V
        for t = 1:T
            Hea(:,t) = sum(H(:,1:t) .* Ea(:,t:-1:1),2);
        end
        Hs = conv2(H,pattern,'same');
        V = W*Hea + TS*Hs + eps;
    end
    
    if updateP
        % update Px
        TVX = [zeros(r,Tt) TS'*(V.^(beta-2).*X) zeros(r,Tt)];
        TV = [zeros(r,Tt) TS'*V.^(beta-1) zeros(r,Tt)];
        
        for t = 1:2*Tt+1
            Pup(t) = sum(sum(H.*TVX(:,t:t+T-1)));
            Pdown(t) = sum(sum(H.*TV(:,t:t+T-1)));
        end
        pattern = pattern.*Pup ./Pdown;
        pattern = pattern/max(pattern);
        
        % update V
        Hs = conv2(H,pattern,'same');
        V = W*Hea +  TS*Hs + eps;
    end
end

result.W = W;
result.TS = TS;
result.a = a;
result.H = H;
result.pattern = pattern;
result.Ha = Hs;