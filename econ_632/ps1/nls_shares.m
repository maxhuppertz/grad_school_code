function SSD = nls_shares(S,delta)
[M,J] = size(S);

D = ones(M,1) * delta;

SSD = sum(sum((S - exp(D)./(sum(exp(D),2)*ones(1,J))).^2, 2), 1);
end