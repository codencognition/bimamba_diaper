## Server Details
- GPU: NVIDIA RTX 5070 Ti (~16GB VRAM)
- Platform: Vast.ai

## Environment Details
- Conda env: Python 3.10
- torch = 2.9.1+cu130
- torchaudio = 2.9.1+cu130
- numpy = 1.26.4
- scipy = 1.15.3
- librosa = 0.9.1
- soundfile = 0.13.1
- h5py = 3.8.0
- matplotlib = 3.5.3
- transformers = 5.2.0
- yamlargparse = 1.31.1
- tensorboard = 2.20.0
- tqdm = 4.67.3
- safe-gpu = 2.0.0
- mamba-ssm = 2.3.0
- causal-conv1d = 1.6.0

### Training Config (`train.yaml`)
- `gpu: 1` — single GPU (RTX 5070 Ti)
- `train_batchsize: 32`
- `dev_batchsize: 128`
- `num_workers: 16`
- `max_epochs: 200`
- `noam_model_size: 2048`
- `noam_warmup_steps: 200000`
- `num_frames: 600`
- `num_speakers: 3`
- `n_attractors: 10`

## Precomputed Features
# Train split
python diaper/precompute_features.py -c examples/train.yaml \
    --output-features-dir <path to output train directory> \
    --split train \
    --train-data-dir <path to kaldi train directory>

# Validation split
python diaper/precompute_features.py -c examples/train.yaml \
    --output-features-dir <path to output train directory> \
    --split validation \
    --valid-data-dir <path to kaldi validation directory>

## Using Precomputed Features
Update `train.yaml`:
train_features_dir: <path to precomputed train features directory>
valid_features_dir: <path to  precomputed validation features directory>
train_data_dir: null
valid_data_dir: null


## Training
python diaper/train.py -c examples/train.yaml


