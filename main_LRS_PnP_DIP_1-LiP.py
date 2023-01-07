from __future__ import print_function
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TKAgg')
import os
import numpy as np
import scipy.io
import torch
import torch.optim
from PIL import Image
import h5py
import math
from scipy.io import loadmat
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark =True
import warnings
warnings.filterwarnings('ignore')
from scipy import linalg
from numpy.linalg import eig
from skimage.restoration import denoise_nl_means
from models.my_Lipschitz_Unet import my_Lipschitz_Unet
from utils.inpainting_utils import *
from include import *
from scipy.io import savemat
from numpy import save
from numpy import load
import pytorch_ssim
from numpy.linalg import matrix_rank
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset


def state_convergence(current,previous):
        Distance = torch.log(torch.norm(current-previous, p=2))    
        return Distance


def tic():
    #Homemade version of matlab tic and toc functions
    import time
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()

def toc():
    import time
    if 'startTime_for_tictoc' in globals():
        print("Elapsed time is " + str(time.time() - startTime_for_tictoc) + " seconds.")
    else:
        print("Toc: start time not set")

def psnr(img1, img2):
    mse = torch.mean((img1  - img2 ) ** 2)
    if mse < 1.0e-10:
        print('they are the same')
        return 100
    PIXEL_MAX = 255
    return 10 * math.log10(PIXEL_MAX / math.sqrt(mse))


def bach_mpsnr(x_true, x_pred):
    n_bands = x_true.shape[1] 
    batch_size = x_true.shape[0] 
    mean_for_each = 0
    for i in range(batch_size):
        a = x_true[i,:, :, :]
        b = x_pred[i,:, :, :]
        p = [psnr( a[k,:, :], b[k,:, :]) for k in range(n_bands)]
        mean_for_each += np.mean(p)        
    return mean_for_each/batch_size

class EarlyStop():
    def __init__(self, size, patience):
        self.patience = patience
        self.wait_count = 0
        self.best_score = float('inf')
        self.best_epoch = 0
        self.img_collection = []
        self.stop = False
        self.size = size

    def check_stop(self, current, cur_epoch):
      #stop when variance doesn't decrease for consecutive P(patience) times
        if current < self.best_score:
            self.best_score = current
            self.best_epoch = cur_epoch
            self.wait_count = 0
            should_stop = False
        else:
            self.wait_count += 1
            should_stop = self.wait_count >= self.patience
        return should_stop

    def update_img_collection(self, cur_img):
        self.img_collection.append(cur_img)
        if len(self.img_collection) > self.size:
            self.img_collection.pop(0)

    def get_img_collection(self):
        return self.img_collection


def myMetric(x1, x2):
    return ((x1 - x2) ** 2).sum() / x1.size
    # return (np.abs(x1 - x2)).sum() / x1.size


def Accu_Energy_ratio(X, p):  
    U, S, V = np.linalg.svd(X, full_matrices=False)
    
    r = np.sum(np.diag(S)[:p-1])/np.sum(np.diag(S))
    
    return r

def torch_to_np(img_var):
    '''Converts an image in torch.Tensor format to np.array.
    From 1 x C x W x H [0..1] to  C x W x H [0..1]
    '''
    return img_var.detach().cpu().numpy()[0]

def get_image_block(input_img, block_size,slidingDis):
    bb=block_size
    
    idx_Mat = torch.zeros(input_img.size(0)-bb+1,input_img.size(1)-bb+1)
    #print(idx_Mat.size())
    row,col = idx_Mat.size()

    idx_Mat[0:row+1:slidingDis,0:col+1:slidingDis] = 1

    ################### append the last row or column ###################
    if((input_img.size()[1])%bb !=0):
        #print('append column')
        idx_Mat[0:row+1:slidingDis,col-1] = 1
    if((input_img.size()[0])%bb !=0):
        #print('append row')
        idx_Mat[row-1,0:col+1:slidingDis] = 1
      
    if((input_img.size()[0])%bb !=0 and input_img.size()[1]%bb !=0):
        idx_Mat[row-1, col-1] = 1
   
    #block_index = np.where(idx_Mat.flatten() == 1)
    idx =  np.argwhere(idx_Mat.numpy().flatten(order='F')==1)

    x_index,y_index =  np.unravel_index(idx, idx_Mat.size(), order='F')
    
    x_index = x_index.flatten()
    y_index = y_index.flatten()

    blocks = torch.zeros((bb**2,len(x_index)))
    
    for i in range(len(x_index)):
        currBlock = input_img[x_index[i]:x_index[i]+bb,y_index[i]:y_index[i]+bb]
        blocks[:,i] = torch.Tensor(currBlock.numpy().flatten(order='F'))

    return blocks,x_index,y_index ,idx_Mat




def Shrinkage_Operator(X, tau):
    
    r = np.sign(X) * np.maximum(np.abs(X) - tau, 0)
    
    return r

def SVT(X, tau):
    U, S, V = np.linalg.svd(X.numpy(), full_matrices=False)
    r =  np.matmul( np.matmul(U, Shrinkage_Operator(np.diag(S), tau)  ), V )

    return torch.Tensor(r)
    
def print_singular_value(X):  
    U, S, V = np.linalg.svd(X, full_matrices=False)
    plt.plot(S)
    plt.xlabel('band numbers')
    plt.xlim([0,10])
    plt.ylabel('Singular Value')
    plt.show()
        
    return S


def soft_thresh(x, l):
    return np.sign(x) * np.maximum(np.abs(x) - l, 0.)

def ista(y,H, lambda_ista, alpha, Nit):
    x = torch.zeros((H.shape[1],1))
    alpha= 2*(np.trace(torch.mm(H.T, H).numpy())+np.trace(torch.mm(H.T, H).numpy()))
    T = lambda_ista/(2*alpha)
    for i in range(Nit):    
        gradient = x + torch.mm(H.T, y - torch.mm(H,x) )/ alpha

        patch_kw = dict(patch_size=3,      # 3x3 patches
                patch_distance=3,  
         )
		#denoiser_out = bm3d.bm3d(gradient, sigma_psd=0.1, stage_arg=bm3d.BM3DStages.HARD_THRESHOLDING).reshape((H.shape[1],1))
        denoiser_out =  denoise_nl_means(gradient, h=T, fast_mode=True,**patch_kw)
        x = torch.Tensor(denoiser_out).view((-1,1))
    return x


def delete_element(tensor, indices):
    mask = torch.ones(tensor.size(), dtype=torch.bool)
    mask[indices,:] = False
    return tensor[mask].view((-1,tensor.size()[1]))



def get_DIP_out( DIP_target, DIP_input, num_iter,learning_rate,show_every):

    #########################  Network Setup ##########################
    pad = 'reflection' #'zero'
    ln_lambda = 1 #the lambda in Lipschitz normalization, which is used to control spectral bias. in our thi is set to be 1

    net = my_Lipschitz_Unet(num_input_channels=128, num_output_channels=128, ln_lambda=ln_lambda,pad=pad).cuda()
    opt_z = torch.optim.Adam(net.parameters(), learning_rate)   # optimizer
    mse = torch.nn.MSELoss()   # Loss


    #########################  ES Setting     #########################
    variance_history = []
    buffer_size = 30
    patience = 60
    earlystop = EarlyStop(size=buffer_size,patience=patience)
    x_axis = []
    

    ##########################   DIP Main Loop    #########################

    for i in range(num_iter):
        opt_z.zero_grad()

        out = net(DIP_input)
    
        loss =  mse( DIP_target* mask_bkg,   out  * mask_bkg )


        loss.backward()
        opt_z.step()

        print ('Iteration %05d    Loss %f      ' % (i, loss.item(),), '\r', end='')

        if  i % (show_every) == 0:

            #variance hisotry
            r_img_np = torch_to_np(out)
            r_img_np = r_img_np.reshape(-1)
            earlystop.update_img_collection(r_img_np)
            img_collection = earlystop.get_img_collection()
            
            if len(img_collection) == buffer_size:
                ave_img = np.mean(img_collection,axis = 0)
                variance = []
                for tmp in img_collection:
                    variance.append(myMetric(ave_img, tmp))
                cur_var = np.mean(variance)
                cur_epoch = i
                variance_history.append(cur_var)
                x_axis.append(cur_epoch)
                if earlystop.stop == False:
                    earlystop.stop = earlystop.check_stop(cur_var, cur_epoch)
                    
            if earlystop.stop == True:

                return out
           
            


########################## load image ########################## 
data_dict = h5py.File('/home/s1809498/first_paper_code/data/low_rank_sparsity_noisy.mat')
noisy_img = data_dict['masked_image']  # received 36 36 128 1
noisy_img  = np.array(noisy_img)
noisy_img = torch.Tensor(noisy_img.transpose((-1,2,1,0))  ).cuda()   # reshaped 1 128 36 36
test_incomplete  =  noisy_img
DIP_target = noisy_img.clone()

data_dict = h5py.File('/home/s1809498/first_paper_code/data/low_rank_sparsity_clean.mat')
gt_image = data_dict['clean_image']  # received 36 36 128 1
gt_image  = np.array(gt_image)
gt_image = torch.Tensor(gt_image.transpose((-1,2,1,0))  ).cuda()   # reshaped 1 128 36 36
clean_image = gt_image.clone()

data_dict = loadmat('/home/s1809498/first_paper_code/data/low_rank_sparsity_mask.mat')
single_mask = data_dict['msk']  # 36 36 1 1
single_mask  = np.array(single_mask, order='F')
tempt = torch.Tensor(single_mask).cuda()
single_mask = torch.Tensor(single_mask.transpose((0,1,3,2)) ).cuda()


data_dict = loadmat('/home/s1809498/first_paper_code/data/trained_dictionary.mat')
D = data_dict['Dictionary']  # 36 36 1 1
D  = np.array(D, order='F')
D =torch.Tensor(D)
Full_Dictionary = D


################ expand mask to 3D #############################
crop_size = 36
y = torch.ones(1,1,crop_size ,crop_size).cuda()
mask_bkg =tempt.clone()
mask_for_hole = (y - mask_bkg)

mask = torch.zeros((crop_size*crop_size,128))
for i in range(128):
   mask[:,i]= single_mask.flatten()


img_np = gt_image.detach().cpu().view((128,crop_size,crop_size))
Y_observed = torch.Tensor( noisy_img.view((128,crop_size,crop_size)).cpu().numpy().transpose(2, 1,0)).reshape((crop_size*crop_size,128))
M_transpose_Y = Y_observed
M_transpose_M = mask.clone().detach().cpu()



##########################  regularization parameters, they should be tuned to obtain the best performance   ####################
gamma = 0.5    #   data-fidelity term
lambda_1 =  torch.zeros(Y_observed.size())   # lagrangian for sparsity
lambda_2 = torch.zeros(Y_observed.size())     # lagrangian for low rank
mu_1 =0.1                        # regularization parameter for sparsity
mu_2 = 0.1                       # regularization parameter for low rank

lambda_ista = 0.1     
Nit = 100      # number of iterations required for sparse coding step
noise_sigma = 0.12
iteration_num = 250     # number of iterations required for LRS-PnP-DIP(1-Lip)
X = Y_observed

X_state_distance = []
lambda_1_state_distance = []
lambda_2_state_distance = []

bb =36
slidingDis =36

####################  Input HSI Image MPSNR ####################
list_MPSNR = [33.074]


blocks_copy,rows,cols,idx_Mat  = get_image_block(Y_observed, bb ,slidingDis)

best_PSNR = 0
method = '2D'


dip_iter =[5000]*500  # number of iterations required for training DIP

for itr in range(iteration_num):
    print('Outer-Loop Iteration: ',itr)
   
    X_previ = X.clone()
    lambda_1_previ = lambda_1.clone()
    lambda_2_previ = lambda_2.clone()


    ##########   optimization with respect of D using sparse coding      ####################################
    blocks,rows,cols,idx_Mat  = get_image_block((X+lambda_1/mu_1),bb,slidingDis)   
    
    tic()
    print('begin sparse coding  with learned dictionary D')
    

    Max_num_zero = 0 
    count_missing = 0
    Phi_z = torch.zeros(blocks.size())
    total_num_zero = 0 
    count_complete = 0
    for jj in range( Phi_z.size()[1]):
        
        pruned_Dictionary = Full_Dictionary     # load original dictionary
    
        valid_pixel = blocks[:,jj].view((bb**2,1))    # current block
        
        tempt = blocks_copy[:,jj].view((bb**2,1)) 
        
        missing_index = np.where(tempt.flatten() == 0)[0]
        
        Num_zero  =  len(missing_index)

        if( Num_zero > 0):
    
            #####   delete certain elements correspnding to missing pixels  ##########  
            valid_pixel = delete_element(valid_pixel,torch.Tensor(missing_index).tolist())
            pruned_Dictionary = delete_element(pruned_Dictionary, torch.Tensor(missing_index).tolist())
            Coefs = ista(valid_pixel,pruned_Dictionary, lambda_ista,0, Nit)
            Phi_z[:,jj] = torch.mm(Full_Dictionary,Coefs).flatten()
            count_missing = count_missing+1
            
        else:

            Coefs = ista(valid_pixel,Full_Dictionary, lambda_ista,0, Nit)
            Phi_z[:,jj] = torch.mm(Full_Dictionary,Coefs).flatten()
            count_complete =count_complete+1

    toc()
    
    
    
    
    ###########   optimization with respect of U    ######################################################

    print('begin optimize U');
    tic()

    DIP_input = ( X  + (1/mu_2)*lambda_2).numpy().transpose(1,0).reshape((128,crop_size,crop_size)).transpose((0,2,1)).reshape((1,128,crop_size,crop_size))
    DIP_input = torch.Tensor(DIP_input).cuda()

    U = get_DIP_out( DIP_target, DIP_input,
        num_iter = dip_iter[itr],
        learning_rate = 0.1,
        show_every = 1).detach().cpu()       #DIP output size: 1,128,36,36
    U = U.numpy().transpose(0,1,3,2).reshape(128,-1).transpose(1,0)   # U size: 1296,128
    toc()
    
    


    ###########   optimization with respect of X    ######################################################
    print('begin optimize X with closed-form');
    tic()
    count = 0
    Weight = torch.zeros(Y_observed.size())
    IMout = torch.zeros(Y_observed.size())
    lambda_1 = torch.Tensor(lambda_1.view((128,crop_size,crop_size))).reshape((crop_size*crop_size,128))
    blocks_lamdba_1,rows,cols,idxMat = get_image_block(lambda_1,bb,slidingDis)
    lambda1_summation =  torch.zeros(Y_observed.size())

    for i in range(len(cols)):
        row = rows[i]
        col = cols[i]
        
        block = torch.Tensor(Phi_z[:,count].view((bb,bb)).numpy().transpose(1,0))
        
        blocks_lambda = torch.Tensor(blocks_lamdba_1[:,count].view((bb,bb)).numpy().transpose(1,0))
        
        IMout[row:row+bb,col:col+bb] = IMout[row:row+bb,col:col+bb]  +  block
        Weight[row:row+bb,col:col+bb] = Weight[row:row+bb,col:col+bb]  +  torch.ones(bb)
        
        lambda1_summation[row:row+bb,col:col+bb] = lambda1_summation[row:row+bb,col:col+bb]  +  blocks_lambda
        count = count+1

    X = (gamma* M_transpose_Y + mu_1*IMout + mu_2*U - lambda1_summation - lambda_2) / (gamma*M_transpose_M + mu_1* Weight+  mu_2)
    toc()



    ########### Update Lagrangian parameter lambda_1, lambda_2  ######################################################
    lambda_1 = lambda_1 + mu_1 * (X-IMout) 
    lambda_2 = lambda_2 + mu_2 * (X-U)


    X_state_distance.append(state_convergence(X,X_previ))
    lambda_1_state_distance.append(state_convergence(lambda_1,lambda_1_previ))
    lambda_2_state_distance.append(state_convergence(lambda_2,lambda_2_previ))

    ###########  Performance Test after each itration  #######################################################################
    final_inpainted = torch.Tensor(X.reshape((crop_size,crop_size,128)).numpy().transpose(1,0,2))
    PSNRIn = torch.zeros((1,128))
    PSNROut = torch.zeros((1,128))
    tempt_masked_image = torch.Tensor(noisy_img.cpu().view((128,crop_size,crop_size)))
    tempt_clean_image = torch.Tensor(clean_image.cpu().view((128,crop_size,crop_size)))

    for num in range(128):
        PSNRIn[0,num] = 10*torch.log10(255/torch.sqrt(torch.mean(torch.mean((tempt_masked_image[num,:,:]-tempt_clean_image[num,:,:])**2))))
        PSNROut[0,num] =  10*torch.log10(255/torch.sqrt(torch.mean((final_inpainted [:,:,num]-tempt_clean_image[num,:,:])**2)))

    MPSNRIn = torch.mean(PSNRIn)
    MPSNROut = torch.mean(PSNROut)
    
    list_MPSNR.append(MPSNROut)
    
    if (MPSNROut > best_PSNR):
         best_PSNR  = MPSNROut



    band_vis = 80
    generated_test_image = torch.Tensor(final_inpainted.numpy().transpose(2,0,1).reshape((1,128,crop_size,crop_size))).cuda()
    Original_MPSNR = bach_mpsnr(gt_image, test_incomplete)
    Test_MPSNR = bach_mpsnr(gt_image, generated_test_image)  
    Output_MSSIM = pytorch_ssim.ssim(gt_image, generated_test_image)
    Input_MSSIM = pytorch_ssim.ssim(gt_image, test_incomplete)
    show_clean = gt_image.detach().cpu().view((128,crop_size,crop_size)).numpy().transpose(1, 2,0)
    show_corrupted = test_incomplete.detach().cpu().view((128,crop_size,crop_size)).numpy().transpose(1, 2,0)
    show_generated = generated_test_image.detach().cpu().view((128,crop_size,crop_size)).numpy().transpose(1, 2,0)
    gt_hole = (gt_image* mask_for_hole).detach().cpu().view((128,crop_size,crop_size)).numpy().transpose(1, 2,0)
    generated_hole = (generated_test_image * mask_for_hole).detach().cpu().view((128,crop_size,crop_size)).numpy().transpose(1, 2,0)



    f, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1,5, sharey=True, figsize=(15,15))
    ax1.imshow(show_clean[:,:,   band_vis], cmap='gray')
    ax1.title.set_text('Clean Image')
    ax2.imshow(show_corrupted[:,:,band_vis], cmap='gray')
    ax2.title.set_text('Corrupted Image')
    ax2.set_xlabel("Input MPSNR is: {:4f}".format(Original_MPSNR))
    ax3.imshow(show_generated[:,:,band_vis], cmap='gray')
    ax3.title.set_text('LRS-PnP-DIP results')
    ax4.imshow(gt_hole[:,:,band_vis], cmap='gray')
    ax4.title.set_text('Ground-Truth hole region')
    ax5.imshow(generated_hole[:,:,band_vis], cmap='gray')
    ax5.title.set_text(' generated hole region')

    axins = zoomed_inset_axes(ax3, 1.8, loc=4) # zoom = 6
    axins.imshow(show_generated[:,:,band_vis],cmap='gray')
    axins.set_xlim(14, 19)
    axins.set_ylim(23, 13)
    plt.yticks(visible=False)
    plt.xticks(visible=False)
    patch, pp1,pp2 = mark_inset(ax3, axins, loc1=1, loc2=1, fc="none", ec="red")
    pp1.loc1 = 1
    pp1.loc2 = 4
    pp2.loc1 = 3
    pp2.loc2 = 2
    plt.draw()
    

    ax3.set_xlabel("Inpainting MPSNR: {:4f}".format( Test_MPSNR))
    plt.show()

    print("Inpainting MPSNR: {:4f} MSSIM is: {:4f} ".format( Test_MPSNR,   Output_MSSIM ))

'''
np.save("X_state_distance", np.array(X_state_distance))
np.save("lambda_1_state_distance", np.array(lambda_1_state_distance))
np.save("lambda_2_state_distance", np.array(lambda_2_state_distance))
np.save("list_MPSNR", np.array(list_MPSNR))
'''

figure, axis = plt.subplots(2, 2)
# For x 
axis[0, 0].plot(X_state_distance)
axis[0, 0].set_title("Convergence of state x")
axis[0, 0].set_xlabel('iteration Number k')
axis[0, 0].set_ylabel('$\Vert x^{k+1}-x^{k} \Vert^2$')
axis[0, 0].set_xticks(range(0,iteration_num,5)) 
# For lambda 1
axis[0, 1].plot(lambda_1_state_distance)
axis[0, 1].set_title("Convergence of $\lambda_1$")
axis[0, 1].set_xlabel('iteration Number k')
axis[0, 1].set_ylabel('$\Vert \lambda_1^{k+1}-\lambda_1^{k} \Vert^2$')
axis[0, 1].set_xticks(range(0,iteration_num,5)) 

# For lambda 2
axis[1, 0].plot(lambda_2_state_distance)
axis[1, 0].set_title("Convergence of $\lambda_1$")
axis[1, 0].set_xlabel('iteration Number k')
axis[1, 0].set_ylabel('$\Vert \lambda_2^{k+1}-\lambda_2^{k} \Vert^2$')
axis[1, 0].set_xticks(range(0,iteration_num,5)) 
  
# For MPSNR 
axis[1, 1].plot(list_MPSNR)
axis[1, 1].set_title("Inpainting Results")
axis[1, 1].set_xlabel('iteration Number k')
axis[1, 1].set_ylabel('MPSNR')
axis[1, 1].set_xticks(range(0,iteration_num+1,5)) 

# Combine all the operations and display
plt.show()
