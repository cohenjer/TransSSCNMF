function [onsets, offsets] = detectingOffsets(onsets, offsets)

offsets(offsets<onsets(1)) = [];

Lon = length(onsets);
Loff = length(offsets);

P = zeros(Lon+Loff,2);
P(:,1) = [onsets;offsets];
P(:,2) = [ones(Lon,1);-ones(Loff,1)];

P = sortrows(P);

Po = P;
for i = 1:Lon+Loff-1        
    if P(i,2)+P(i+1,2) == 2
        Po = [Po; [P(i+1,1)-1,-1]];
    elseif P(i,2)+P(i+1,2) == -2
        Po(i+1,:) = 0;
    end
end

Po(Po(:,1)==0,:) = [];

P = sortrows(Po);

offsets = P(2:2:end,1);

