python -m venv env 
source env/bin/activate
python -m pip install .
python -m pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html