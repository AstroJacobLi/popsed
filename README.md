# popsed
Stellar population inference for galaxy population with neural density estimators

Overleaf draft link: https://www.overleaf.com/3537387468zsymwpdgzmsr



### Environment on `tiger`

`conda create --prefix /scratch/gpfs/$USER/torch-env pytorch torchvision torchaudio cudatoolkit=10.2 jupyter astropy palettable scikit-learn matplotlib tensorboard faiss-cpu --channel pytorch`

Almost all notebooks run on GPU. So first excute `popsed/script/apply_gpu.sh` to apply for a GPU on Tiger. The default time is set to 6 hours. Then execute `popsed/script/setup_env.sh` to load the environment before running code.

Connect to `jupyter` on the allocated GPU from local machine: `tigergpu_tunnel 7777 tiger-xxxxx`


<!-- 
`conda create --name tf2-cpu tensorflow jupyter astropy palettable scikit-learn matplotlib`
`conda create --prefix /scratch/gpfs/$USER/tf2-cpu tensorflow jupyter astropy palettable scikit-learn matplotlib` -->
