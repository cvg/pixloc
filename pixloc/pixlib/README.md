# PixLib - training library

`pixlib` is built on top of a framework whose core principles are:

- modularity: it is easy to add a new dataset or model with custom loss and metrics;
- reusability: components like geometric primitives, training loop, or experiment tools are reused across projects;
- reproducibility: a training run is parametrized by a configuration, which is saved and reused for evaluation;
- simplicity: it has few external dependencies, and can be easily grasped by a new user.

## Framework structure
`pixlib` includes of the following components:
- [`datasets/`](./datasets) contains the dataloaders, all inherited from [`BaseDataset`](./datasets/base_dataset.py). Each loader is configurable and produces a set of batched data dictionaries.
- [`models/`](./models) contains the deep networks and learned blocks, all inherited from [`BaseModel`](./models/base_model.py). Each model is configurable, takes as input data, and outputs predictions. It also exposes its own loss and evaluation metrics.
- [`geometry/`](pixlib/geometry) groups Numpy/PyTorch primitives for 3D vision: poses and camera models, linear algebra, optimization, etc.
- [`utils/`](./utils) contains various utilities, for example to [manage experiments](./utils/experiments.py).

Datasets, models, and training runs are parametrized by [omegaconf](https://github.com/omry/omegaconf) configurations. See examples of training configurations in [`configs/`](./configs/) as `.yaml` files.

## Workflow
<details>
<summary><b>Training:</b></summary><br/>

The following command starts a new training run:
```bash
python3 -m pixloc.pixlib.train experiment_name \
		--conf pixloc/pixlib/configs/config_name.yaml
```

It creates a new directory `experiment_name/` in `TRAINING_PATH` and dumps the configuration, model checkpoints, logs of stdout, and [Tensorboard](https://pytorch.org/docs/stable/tensorboard.html) summaries.

Extra flags can be given:

- `--overfit` loops the training and validation sets on a single batch ([useful to test losses and metrics](http://karpathy.github.io/2019/04/25/recipe/)).
- `--restore` restarts the training from the last checkpoint (last epoch) of the same experiment.
- `--distributed` uses all GPUs available with multiple processes and batch norm synchronization.
- individual configuration entries to overwrite the YAML entries. Examples: `train.lr=0.001` or `data.batch_size=8`.

**Monitoring the training:** Launch a Tensorboard session with `tensorboard --logdir=path/to/TRAINING_PATH` to visualize losses and metrics, and compare them across experiments. Press `Ctrl+C` to gracefully interrupt the training.
</details>

<details>
<summary><b>Inference with a trained model:</b></summary><br/>

After training, you can easily load a model to evaluate it:
```python
from pixloc.pixlib.utils.experiments import load_experiment

test_conf = {}  # will overwrite the training and default configurations
model = load_experiment('name_of_my_experiment', test_conf)
model = model.eval().cuda()  # optionally move the model to GPU
predictions = model(data)  # data is a dictionary of tensors
```

</details>

<details>
<summary><b>Adding new datasets or models:</b></summary><br/>

We simply need to create a new file in [`datasets/`](./datasets/) or [`models/`](./models/). This makes it easy to collaborate on the same codebase. Each class should inherit from the base class, declare a `default_conf`, and define some specific methods. Have a look at the base files [`BaseDataset`](./datasets/base_dataset.py) and [`BaseModel`](./models/base_model.py) for more details. Please follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) and use relative imports.

</details>
