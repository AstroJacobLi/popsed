# popsed
Stellar population inference for galaxy population with neural density estimators

Overleaf draft link: https://www.overleaf.com/3537387468zsymwpdgzmsr

![image](https://user-images.githubusercontent.com/29670581/173158592-1e8ff826-c95b-40af-bb85-02f60c679de6.png)


### Environment on `tiger`

`conda create --prefix /scratch/gpfs/$USER/torch-env pytorch torchvision torchaudio cudatoolkit=11.3 jupyter astropy palettable scikit-learn matplotlib tensorboard faiss-cpu numba tqdm dill --channel pytorch`

Almost all notebooks run on GPU. So first excute `popsed/script/apply_gpu.sh` to apply for a GPU on Tiger. The default time is set to 6 hours. Then execute `popsed/script/setup_env.sh` to load the environment before running code.

Connect to `jupyter` on the allocated GPU from local machine: `tigergpu_tunnel 7777 tiger-xxxxx`


<!-- 
`conda create --name tf2-cpu tensorflow jupyter astropy palettable scikit-learn matplotlib`
`conda create --prefix /scratch/gpfs/$USER/tf2-cpu tensorflow jupyter astropy palettable scikit-learn matplotlib` -->
