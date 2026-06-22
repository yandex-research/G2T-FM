# Turning Tabular Foundation Models into Graph Foundation Models

This is the official repository for the paper "Turning Tabular Foundation Models into Graph Foundation Models" ([arXiv](https://arxiv.org/abs/2508.20906)). In this repository, we provide code for reproducing our key experiments. 

> [!NOTE]
> See also: [GraphPFN](https://github.com/yandex-research/graphpfn), our next step toward applying TFMs and PFNs to graph node-level tasks. GraphPFN augments the LimiX tabular foundation model with graph adapters and is pretrained on millions of synthetic datasets, achieving state-of-the-art results.

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

You can execute a minimal run (G2T-LimiX finetuning with 10 ensemble members) with the following command:

```
uv run bin/go.py exp/g2t/main/limix/finetune/10/tolokers-2/tuning.toml --force
```

## Project Structure

- `bin/` - Training and evaluation scripts
- `exp/` - Experiment configurations and results
- `lib/` - Common utilities and tools
- `vendor/` – Vendored third-party code with minor import modifications for compatibility

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

## Licenses

This project uses third-party components [LimiX](https://github.com/limix-ldm/LimiX), [TabICL](https://github.com/soda-inria/tabicl) and [TabPFN](https://github.com/PriorLabs/TabPFN). See the `NOTICE` file and `LICENSES/` directory for details.

***

Built with PriorLabs-TabPFN
