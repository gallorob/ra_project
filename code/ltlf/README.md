# LTLf and Reward Shaping experiment

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

To specify a particular temporal goal, add the `--goal={GOAL_NAME}` flag.

To apply reward shaping, add the `--shape_rewards=True` flag.