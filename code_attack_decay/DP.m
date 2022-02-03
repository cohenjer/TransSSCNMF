function [D, path]= DP(C, w)

[S,T] = size(C);
D = zeros(S,T);
E = zeros(S,T);
E(:,1) = 1:S;
path = zeros(1,T);

D(:,1) = C(:,1);
for t = 2:T
    for s = 1:S
        [D(s,t), E(s,t)] = min(D(:,t-1)+C(s,t)*w(:,s));
    end
end

[~, path(T)] = min(D(:,T));
for t = T-1:-1:1
    path(t) = E(path(t+1),t);
end