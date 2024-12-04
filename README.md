# Neural Port-Hamiltonian Differential Algebraic Equations

Setup virtual environment
```
    python3.11 -m venv env
    source env/bin/activate
    python3.11 -m pip install -r requirements_xxxx.txt # replace with ubuntu version
```
Generate the training data for the distributed generation unit:
```
    cd environments
    python3.11 dgu_random.py
```
Train the Neural Port-Hamiltonian Differential Algebraic Equation
```
    python3.11 run_training.py
    # Then enter `dgu`
```

