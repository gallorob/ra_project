# STRIPS experiment

## Installation
Install dependencies with either
```bash
pip install -r requirements.txt
```

or with the provided `Pipfile`.

## Running experiments
Run experiments with
```bash
python learner.py --seed=42 --n=1000
```

At the moment, the STRIPS environment is hardcoded in the `learner.py` file.

**Note**: Since this experiment requires Lydia as backend, make sure it is installed.