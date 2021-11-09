# popsed
Stellar population inference for galaxy population with neural density estimators

Overleaf draft link: https://www.overleaf.com/3537387468zsymwpdgzmsr



### Environment on `tiger`

`conda create --prefix /scratch/gpfs/$USER/torch-env pytorch torchvision torchaudio cudatoolkit=10.2 jupyter astropy palettable scikit-learn matplotlib tensorboard --channel pytorch`

`conda create --name tf2-cpu tensorflow jupyter astropy palettable scikit-learn matplotlib`
`conda create --prefix /scratch/gpfs/$USER/tf2-cpu tensorflow jupyter astropy palettable scikit-learn matplotlib`

And execute `popsed/script/setup_env.sh` before running code.

- Apply for GPU: `salloc --nodes=1 --ntasks=1 --mem=4G --time=02:00:00 --gres=gpu:1`
- Run `jupyter` on the allocated GPU: `tigergpu_tunnel 7777 tiger-xxxxx`