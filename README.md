# dl-quick-train
Experimental faster training for dictionary learning on single GPU machines.

## Installation

Install the package using `pip`:

```bash
pip install dl-quick-train
```

## Usage

You can run the default pipeline from the command line:

```bash
dl-quick-train
```

Alternatively you can import the library and call `run_pipeline` yourself:

```python
from dl_quick_train import run_pipeline

run_pipeline([...], activation_cache_dir="/tmp/activations")
```

Setting `activation_cache_dir` enables caching of model activations on disk.
If the directory already contains cached activations for the requested
configuration, they will be loaded instead of recomputed.

`run_pipeline` accepts a `start_method` argument controlling the
multiprocessing start method (default: `"forkserver"`). Crash reporting is
improved by enabling Python's `faulthandler` in worker processes.
