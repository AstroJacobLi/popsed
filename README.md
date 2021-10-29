# popsed
Stellar population inference for galaxy population with neural density estimators

Overleaf draft link: https://www.overleaf.com/3537387468zsymwpdgzmsr



### Environment on `tiger`

`conda create --prefix /scratch/gpfs/$USER/torch-env pytorch torchvision torchaudio cudatoolkit=10.2 jupyter astropy palettable sep scikit-learn matplotlib tensorboard --channel pytorch`

And execute `popsed/script/setup_env.sh` before running code.