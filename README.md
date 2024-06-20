# Port-Hamiltonian Graph Networks

## Use
Setup virtual environment
```
    python3.11 -m venv env
    source env/bin/activate
    pip3 install -r requirements_xx04.txt # replace xx with ubuntu version
```

Generate dataset
```
    python environments/lc_circuit.py --circuit=lc1 --type=train --n=200 --steps=700
    python environments/lc_circuit.py --circuit=lc1 --type=val --n=5 --steps=1500
```

Set config `configs/lc_circuit_1.py`

Train network
```
    python scripts/train_gnn.py --system=LC1
```

## Test composition
Set config `configs/comp_circuits.py`

Test composition
```
    python scripts/comp_circuits.py
```

## Test zero-shot transfer
Set config `configs/reuse_model.py`

Test zero-shot transfer
```
    python scripts/reuse_model.py
```