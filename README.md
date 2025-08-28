# Turning Tabular Foundation Models into Graph Foundation Models

This is the official repository for the paper "Turning Tabular Foundation Models into Graph Foundation Models". In this repository, we provide code for reproducing our experiments with G2T-FM, GNNs and LightGBM (including ablation). Code for reproduction of our experiments with prior GFMs is coming soon.

> [!NOTE]
> Our work is largely based on [TabPFN](https://github.com/PriorLabs/TabPFN) and ["On Finetuning Tabular Foundation Models" paper](https://github.com/yandex-research/tabpfn-finetuning/tree/main), please consider checking them out too!

## Reproducing Experiments

**Prerequisites**

1. [Install uv](https://github.com/astral-sh/uv?tab=readme-ov-file#installation)
2. Install dependencies
```
uv sync
```
3. Download TabPFNv2 checkpoints
```
wget https://huggingface.co/Prior-Labs/TabPFN-v2-reg/resolve/main/tabpfn-v2-regressor.ckpt?download=true -O checkpoints/tabpfn-v2-regressor.ckpt
wget https://huggingface.co/Prior-Labs/TabPFN-v2-clf/resolve/main/tabpfn-v2-classifier.ckpt?download=true -O checkpoints/tabpfn-v2-classifier.ckpt
```
4. For experiments on [GraphLand](https://github.com/yandex-research/graphland), download datasets and place them in "data" directory

**Running the code**

You can execute a minimal run with a following command:

```
uv run bin/go.py exp/g2t_fm/finetune/tolokers-2/tuning.toml --force
```

## Project Structure

- `bin/` - Training and evaluation scripts
- `exp/` - Experiment configurations and results
- `data/` - Dataset directory (created after download)
- `lib/` - Common utilities and tools

## Configuration

Experiments are configured using TOML files located in the `exp/` directory. Each configuration specifies:
- Dataset path and preprocessing
- Model hyperparameters
- Training settings
- Evaluation metrics

## Results

After training, results are saved in the same directory as the configuration file:
- `report.json` - Evaluation metrics
- Model checkpoints
- Training logs
