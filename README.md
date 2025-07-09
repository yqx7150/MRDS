##  MRDS

Mean-Reverting Diffusion Model-Enhanced Scattering Imaging  
Jiayuan Lin, Qi Yu, Meng Teng, Xinmin Ding, Detang Xiao, Wenbo Wan, and Qiegen Liu, Senior Member, IEEE

###  Abstract:

Scattering media disrupt the rectilinear propagation of light, significantly degrading the resolution and clarity of optical imaging systems. Current scattering imaging techniques usually focus on simple targets and present limitations in imaging quality and reconstruction efficiency. To address these limitations, a mean-reverting diffusion model-enhanced scattering imaging (MRDS) is proposed. During training, prior information is extracted by diffusing the training data into an intermediate state with stable Gaussian noise. Reconstruction begins with low-quality images from physically-guided inversion, followed by iterative solving of reverse-time stochastic differential equations via the Euler-Maruyama method, integrating learned prior information to efficiently reconstruct high-quality images. Simulative and experimental validations demonstrate that MRDS outperforms traditional methods in reconstructing images with fewer artifacts and enhanced detail clarity. Quantitative metrics further demonstrates excellent reconstruction performance, with average metrics reaching 41.19 dB for PSNR, 0.99 for SSIM and 0.0085 for LPIPS. The reconstruction time per image is 2.19 seconds, representing a 44.2-fold acceleration compared to conventional methods. The proposed method achieves high-quality reconstructions of complex targets in a significantly shorter time, which dramatically boosts the efficiency of scattering imaging. 



![MRDS.png](https://github.com/yqx7150/MRDS/blob/main/imgs/MRDS.png)
The main procedure of MRDS. Prior information learning: The network learns the prior information of the data distribution. Physically-guided inverting: The initial reconstruction is executed employing the Wiener filter deconvolution technique. Mean reversion iteration: The target is progressively recovered through the iterative resolution of the reverse-time stochastic differential equation (SDE). GT, ground truth; HQ, high-quality image; LQ, low-quality image; PSF, point spread function; TV, total variation.




###  Requirements and Dependencies:

einops==0.6.0  
lmdb==1.3.0  
lpips==0.1.4  
numpy==1.23.5  
opencv-python==4.6.0.66  
Pillow==9.3.0  
PyYAML==6.0  
scipy==1.9.3  
tensorboardX==2.5.1  
timm==0.6.12  
torch==1.13.0  
torchsummaryX==1.3.0  
torchvision==0.14.0  
tqdm  
gradio  

###  Checkpoints:
We provide the pre-trained model and place it in the Baidu Drive [MRDS](https://pan.baidu.com/s/1CEIuix8AMewR75WU4yE77g?). 

###  Dataset:
The dataset used to train the model in this experiment is Fashion-MNIST dataset.

###  Training:
Place the dataset in the train_GT folder under the train directory, and store the low-quality images  in the train_LQ folder under the same train directory, and then run

python train.py -opt=options/train/refusion.yml


###  Testing:
Place the dataset in the test_GT folder under the test directory, and store the speckle images  in the test_LQ folder under the same test directory, and then run

python test.py -opt=options/test/refusion.yml.

For the system experiment, we provide the data obtained from the experiment and place it in the Baidu Drive [MRDS](https://pan.baidu.com/s/1CEIuix8AMewR75WU4yE77g?).  
Then place the speckle images in the shice folder under the test directory and then run  

python shice.py -opt=options/shice/refusion.yml.


###  Acknowledgement:
Thanks to these repositories for providing us with method code and experimental data: https://github.com/Algolzw/image-restoration-sde. 




