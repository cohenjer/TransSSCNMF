function [HP, P]= detectingOnsets(H,Threshold)

[R,T] = size(H);
HP = zeros(R,T);
P = zeros(R,T);

Thre = 10^(Threshold/20)*max(max(H));

for r = 1:R
    tH = H(r,:);
    tH(tH' - smooth(tH,20)<Thre) = 0;
%     tH(tH<Thre) = 0;
    [c,ind] = findpeaks(tH);
    if ~isempty(ind)
        dind = diff(ind);
        for i = 1:length(dind)
            if dind(i)<5
                ind(i) = round((c(i)*ind(i)+c(i+1)*ind(i+1))/(c(i)+c(i+1)));
                ind(i+1) = 0;
                c(i) = c(i)+c(i+1);
                c(i+1) = 0;
            end
        end
        ind(ind==0) = [];
        c(c==0) = [];
        HP(r,ind) = c;
        P(r,ind) = 1;
    end
end

