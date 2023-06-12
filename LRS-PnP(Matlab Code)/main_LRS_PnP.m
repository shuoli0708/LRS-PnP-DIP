clear;
clc;
close all;
load datasets/Chikusei.mat;

%load trained_36_36_dictionary.mat;
load trained_dictionary.mat;
D=Dictionary;
Full_Dictionary = D;


img_clean =  reshape(img,[128,144,144]);
img_clean = squeeze(permute(img_clean,[2,3,1]));

% cropping to smaller size
crop_size = 36;
cropped = img_clean(50:crop_size+49,50:crop_size+49,:);
%cropped = img_clean(79:crop_size+78,79:crop_size+78,:);
clean_image = cropped;
view_clean=reshape(clean_image,[crop_size*crop_size,128]);

% add noise
sigma =0.12;
randn('seed',0);
noise = sigma.*randn(size(cropped));
noisy_image = clean_image+noise;



% load mask
K=128;
msk = ones(crop_size,crop_size);
%msk(1:crop_size/2,11:12) = 1000;
msk(8:13,27:28) = 1000;
msk(4:5,7:12) = 1000;
msk(18:24,5:6) = 1000;
msk(16:17,13:19) = 1000;
msk(24:25,13:19) = 1000;
generaed_strip_mask = msk;

generaed_strip_mask(msk > 500) = 0;
for i =1:K
    final_mask(:,:,i) = generaed_strip_mask;
end

masked_image = final_mask.* noisy_image;


Y_observed = reshape(masked_image,[crop_size*crop_size,128]);

M_transpose_Y = Y_observed;
M_transpose_M = reshape(final_mask,[crop_size*crop_size,128]);



% regularization parameters, they should be tuned to obtain the best performance
gamma = 0.5;     % data-fidelity term
lambda_1 =  zeros(size(Y_observed));   % lagrangian for sparsity
lambda_2 =  zeros(size(Y_observed));   % lagrangian for low rank
mu_1 = 0.15;       % regularization parameter for sparsity
mu_2 = 0.15;            % regularization parameter for low rank


lambda = 0.1;
 Nit = 80;
 noise_sigma =0.12;
iteration_num = 13;   
X = Y_observed;



% take record of the location of missing pixel. we are not going to use it
% in sparse cosing!
 bb =36;
 slidingDis =36;
 
 
 list_MPSNR = [];
[blocks_copy,idx] = my_im2col(Y_observed,[bb,bb],slidingDis);   % input size (20736, 128), ??? ?????????8x8?patch ??

 best_PSNR = 0;
for itr=1:iteration_num
    disp('Outer-Loop Iteration:');
    disp(itr);
    
    
   
    % optimization with respect of  D using sparse coding
	disp('begin sparse coding  with learned dictionary D');
    tic;
  

    [blocks,idx] = my_im2col((X+lambda_1/mu_1),[bb,bb],slidingDis);   % input size (20736, 128), ??? ?????????8x8?patch ??
    
       
    
    Max_num_zero =0;
    count_missing = 0;
    Phi_z = zeros(size(blocks));
    parfor jj = 1:size(blocks,2)   % 250829
        pruned_Dictionary = Full_Dictionary;      % load original dictionary
        Max_num_zero =0;
        valid_pixel = reshape(blocks(:,jj),[bb^2,1]);  %current block
        Num_zero = length(find(~reshape(blocks_copy(:,jj),[bb*bb,1])) );  %does current block contain missing value??
   
        if( Num_zero > 0) 
            %disp(['Num of zero in each blockis: ',num2str(Num_zero)]);
            missing_index = find(~reshape(blocks_copy(:,jj),[bb*bb,1])) ;
            
            valid_pixel(missing_index,:) = [];
            pruned_Dictionary(missing_index,:) = [];
            
             max_eigen = max(abs(eig(pruned_Dictionary'*pruned_Dictionary)));
            [ Coefs,J] = pnp_ista(valid_pixel,pruned_Dictionary,lambda,max_eigen,Nit,noise_sigma)
            
            count_missing = count_missing+1;
            
            Phi_z(:,jj)= Full_Dictionary*Coefs ;
                 
             Num_zero_2 = length(find(~reshape(Phi_z(:,jj),[bb*bb,1])) );    
             %disp(['Num of zero after sparse coding: ',num2str(Num_zero_2)]);    
        else   
            
             max_eigen = max(abs(eig(Full_Dictionary'*Full_Dictionary)));
             [ Coefs,J] = pnp_ista(blocks(:,jj),Full_Dictionary,lambda,max_eigen,Nit,noise_sigma)
             
             Phi_z(:,jj)= Full_Dictionary*Coefs ;
        end
     
    end
    %disp(['Num of blocks with missing pixels:  ',num2str(count_missing)]);
    toc;


    
    
    % optimization with respect of U
    disp('begin optimize U');
    tic;
    U = Do(1/mu_2, X  + (1/mu_2)*lambda_2);
    toc;
    view_X = reshape(X,[36,36,128]);
    
    
    
    disp('begin optimize X using closed-form-solution');
    tic;
	% optimization with respect of X

    count = 1;
    Weight = zeros(size(Y_observed));
    IMout = zeros(size(Y_observed));
    
    [blocks_lamdba_1,idx] = my_im2col(lambda_1,[bb,bb],slidingDis);
    lambda1_summation = zeros(size(Y_observed));
    
    [rows,cols] = ind2sub(size(Y_observed)-bb+1,idx);    
    for i  = 1:length(cols)     % i:250829 ?? ??? 8x8 patch
        col = cols(i); row = rows(i);        
        block = reshape(Phi_z(:,count),[bb,bb]); %  blocks(:,count) ????? 8 x 8 ?patch????????? Sum(D *alpha
        blocks_lambda = reshape(blocks_lamdba_1(:,count),[bb,bb]);
        IMout(row:row+bb-1,col:col+bb-1)=IMout(row:row+bb-1,col:col+bb-1)+block;
        Weight(row:row+bb-1,col:col+bb-1)=Weight(row:row+bb-1,col:col+bb-1)+ones(bb);
        
        lambda1_summation(row:row+bb-1,col:col+bb-1)=lambda1_summation(row:row+bb-1,col:col+bb-1)+blocks_lambda;
        count = count+1;
    end
    X = (gamma* M_transpose_Y + mu_1*IMout + mu_2*U - lambda1_summation - lambda_2) ./ (gamma*M_transpose_M + mu_1* Weight+  mu_2); 
    toc;
    

    
    
    
     % Update Lagrangian parameter lambda_1, lambda_2, and mu
      lambda_1 = lambda_1+mu_1*(X-IMout);
      lambda_2 = lambda_2+mu_2*(X-U);
    
      
     
     
	% Performance Test after each itration
    final_inpainted = X;
    band_vis=80;
    fildata = reshape(final_inpainted,crop_size,crop_size,128);
  
    for num=1:128    % compute the assessing indices
    PSNRIn(1,num) = 10*log10(255/sqrt(mean(mean((masked_image(:,:,num)-clean_image(:,:,num)).^2))));
    PSNROut(1,num) = 10*log10(255/sqrt(mean(mean((fildata(:,:,num)-clean_image(:,:,num)).^2))));
    end
    MPSNRIn=mean(PSNRIn);
    MPSNROut=mean(PSNROut);
    
    list_MPSNR =[list_MPSNR, MPSNROut];
     if (MPSNROut > best_PSNR)
         best_PSNR  =MPSNROut;
         best_rank = num2str(rank(reshape(fildata,[crop_size*crop_size,128])));
     end

    disp(['MPSNR: : ',num2str( MPSNROut)  ]);
    
end



figure()
plot(list_MPSNR);
xlabel('Itration');
ylabel('MPSNR');
title('MPSNR');

% 
figure()
subplot(1,3,1);
imshow(cropped(:,:,band_vis),'InitialMagnification', 800);
title('clean image');

subplot(1,3,2);
imshow(masked_image(:,:,band_vis),'InitialMagnification', 800);
title('masked and noisy image');

subplot(1,3,3);
imshow(fildata(:,:,band_vis),'InitialMagnification', 800);
title('inpainted result');


missing_spectrmum = reshape(clean_image(16,19,:), [1,128]);
recovered_spectrum = reshape(fildata(16,19,:), [1,128]);

figure()
subplot(1,2,1);
plot(missing_spectrmum);
title('clean spectrum');
subplot(1,2,2);
plot(recovered_spectrum);
title('recovered spectrum');




function r = So(tau, X)
    % shrinkage operator
    r = sign(X) .* max(abs(X) - tau, 0);
end

function r = Do(tau, X)
    % shrinkage operator for singular values
    [U, S, V] = svd(X, 'econ');
    r = U*So(tau, S)*V';
end

