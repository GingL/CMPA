# Training and Evaluation

We provide bash scripts in [scripts/](../scripts).
Make sure to configure the dataset paths in environment variable `DATA` and run the commands from the main directory `cmpa/`.
Below we provide training and evaluation instructions for CMPA. 

### Few-Shot Learning

All you need is `cmpa/scripts/cmpa/main.sh`, which contains six input arguments.

`DATASET` takes as input a dataset name, like `imagenet` or `caltech101`. The valid names are the files' names in `cmpa/configs/datasets/`.

Below we provide examples on how to run cmpa on Caltech101.

**CLIP + CMPA (M=16, end)**:
- 1 shot: `bash scripts/cmpa/main.sh caltech101 1`
- 2 shots: `bash scripts/cmpa/main.sh caltech101 2`
- 4 shots: `bash scripts/cmpa/main.sh caltech101 4`
- 8 shots: `bash scripts/cmpa/main.sh caltech101 8`
- 16 shots: `bash scripts/cmpa/main.sh caltech101 16`


After the experiments are finished, you can use `parse_test_res.py` to calculate the average results instead of manually looking into the log files. Say the structure of `output/` is

```
output
|–– caltech101/
|   |–– 16shots/
|   |   |–– seed1/
|   |   |–– seed2/
|   |   |–– seed3/
|   |–– 8shots/
|   |   |–– seed1/
|   |   |–– seed2/
|   |   |–– seed3/
```

To calculate the average results for the folder `16shots/`, you can run

```bash
python parse_test_res.py output/caltech101/16shots
```

Then, you will see something like this in your terminal

```bash
Parsing files in output/caltech101/cmpa/rn50_16shots/nctx16_cscFalse_ctpend
file: output/caltech101/16shots/seed1/log.txt. accuracy: 91.81%. error: 8.19%.
file: output/caltech101/16shots/seed2/log.txt. accuracy: 92.01%. error: 7.99%.
file: output/caltech101/16shots/seed3/log.txt. accuracy: 92.17%. error: 7.83%.
===
Summary of directory: output/caltech101/16shots
* accuracy: 92.00% +- 0.15%
* error: 8.00% +- 0.15%
===
```

### Robustness to Distribution Shift
To reproduce the robustness experiments, you can simply load the models learned on ImageNet and evaluate them on the following datasets: `imagenetv2`, `imagenet-sketch`, `imagenet-a` and `imagenet-r`.

The command is provided in `scripts/cmpa/xd_test.sh`. The key arguments are `--model-dir`, `--load-epoch` and `--eval-only`. `--model-dir` indicates the directory where the models are saved (i.e. the entire folder containing `log.txt`, the tensorboard file and `prompt_learner/`). `--load-epoch` tells the code to load the model saved at a specific epoch, like `--load-epoch 50` for ImageNet (see the [source code](https://github.com/KaiyangZhou/Dassl.pytorch/blob/master/dassl/engine/trainer.py#L169) for more details).

For example, to evaluate `CLIP + cmpa (M=16, end)` on ImageNetV2, you can do

```bash
# Don't need to use rn5_ep50 here as no training is performed
bash scripts/cmpa/eval.sh imagenetv2 1
```

The default setting is `SHOTS=16`. Feel free to modify the script.

Again, you can use `parse_test_res.py` to automate the calculation of average performance. This time you should append `--test-log`, e.g., `python parse_test_res.py directory --test-log`.

