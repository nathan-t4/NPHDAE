python -m venv env 
source env/bin/activate
python -m pip install matplotlib
python -m pip install tqdm
python -m pip install pyyaml 
python -m pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html # jax
python -m pip install jraph # jraph
python -m pip install networkx[default] # networkx
python -m pip install flax # flax
python -m pip install -q git+https://github.com/google/CommonLoopUtils # clu
python -m pip install tensorflow[and-cuda] # tensorflow