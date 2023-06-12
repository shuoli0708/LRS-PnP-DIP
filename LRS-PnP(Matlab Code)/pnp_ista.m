function [x,J] = pnp_ista(y,H,lambda,alpha,Nit,noise_sigma)
% [x, J] = ista(y, H, lambda, alpha, Nit)
% L1-regularized signal restoration using the iterated
% soft-thresholding algorithm (ISTA)
% Minimizes J(x) = norm2(y-H*x)^2 + lambda*norm1(x) % INPUT
% y - observed signal
% H - matrix or operator
% lambda - regularization parameter
% alpha - need alpha >= max(eig(H?*H))
% Nit - number of iterations % OUTPUT
% x - result of deconvolution
% J - objective function

J = zeros(1, Nit);       % Objective function
x = 0*H'*y;              % Initialize x
T = lambda/(2*alpha);   

addpath('BM3D');
if exist('BM3D.m','file') == 0 %Check existence of  function BM3D
         errordlg({'Function BM3D.m not found! ','Download from http://www.cs.tut.fi/~foi/GCF-BM3D and install it in the folder .../BM3D'});
     error('Function BM3D.m not found!  Download from http://www.cs.tut.fi/~foi/GCF-BM3D and install it in the folder .../BM3D');
end


for k = 1:Nit
        Hx = H*x;
        J(k) = sum(abs(Hx(:)-y(:)).^2) + lambda*sum(abs(x(:)));
        gradient = x + (H'*(y - Hx))/alpha;
        
        x =NLmeansfilter(gradient,3,3,T*0.1);

end


