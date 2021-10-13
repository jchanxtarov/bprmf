# BPRMF
repository to implement Bayesian personalized ranking matrix factorization

```
BPR: Bayesian Personalized Ranking from Implicit Feedback
Steffen Rendle, Christoph Freudenthaler, Zeno Gantner and Lars Schmidt-Thieme
```

## Instllation
```bash
git clone git@github.com:jchanxtarov/bprmf.git
cd bprmf
make setup
```

## Usage Example
### Under the default setting

If you want to save log and best model, please run following command.
```
make run
```

A command to run the code without saving the log or model data (dry-run) is prepared as follow.
```
make dry-run
```

### With your own seting
```
poetry run python src/main.py {options}
```
About options, please [see here](https://github.com/jchanxtarov/bprmf#aguments).

## Aguments
To see all argments and description, please run the following command.
```bash
make args
```

## Refenrence
- https://arxiv.org/pdf/1205.2618.pdf
