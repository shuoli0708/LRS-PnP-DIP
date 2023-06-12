function Aout = columnNormalise(A)
[m,n] = size(A);
An = sqrt(sum(A.^2,1));
Aout = A./(ones(m,1)*An);