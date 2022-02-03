function [Note, pianoRoll] = noteTracking(X, result, threshold)
W = result.W;
TS = result.TS;
a = result.a;
H = result.H;
pattern = result.pattern;

% attack activations
Ha = conv2(H,pattern,'same');

% detect onsets
HP = detectingOnsets(Ha,threshold); 
HP = detectingOnsets(HP,threshold);


setGlobalDM();
global frame fs
interval = frame/fs;

global Wt Tmax

% reconstruction of V
[R,T] = size(H);
ea = zeros(R,Tmax);
Hea = zeros(R,T);
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

Note = zeros(length(find(HP)>0),3);
pianoRoll = zeros(size(H));
num = 0;
for r = 1:R
    [~, onsets] = findpeaks(HP(r,:));
    if ~isempty(onsets)
        Vp = W(:,r)*Hea(r,:) +  TS(:,r)*Hs(r,:) + eps;
        Vep = V-Vp + eps;
        
        Cp = zeros(2,T);
        Cp(1,:) = sum(D(X, Vep));
        Cp(2,:) = sum(D(X, V));
        rCp = Cp./repmat(sum(Cp),2,1);
        
        w = [0.5,0.55;0.55,0.5];
        
        offsets = T;
        [onsets, offsets] = detectingOffsets(onsets', offsets');
        for i = 1:length(onsets)
            index = find(abs(medfilt1(diff(rCp(:,onsets(i):offsets(i))),10))<0.005);
            index(index<2) = [];
            if ~isempty(index)
                offsets(i) = onsets(i)+index(1)-1;
            end
            
            [Dis,path] = DP(rCp(:,onsets(i):offsets(i)),w);
            duration = find(diff([path,1])==-1);
            if ~isempty(duration)
                offsets(i) = onsets(i)+duration(1)-1;
            end
        end
        
        for i = 1:length(onsets)
            num = num+1;
            pianoRoll(r,onsets(i):offsets(i)) = 1;
            Note(num,1) = onsets(i)*interval;
            Note(num,2) = offsets(i)*interval;
            Note(num,3) = r+20;
%             Note(num,4) = HP(r,onsets(i));
        end
    end
end

Note = sortrows(Note,1);