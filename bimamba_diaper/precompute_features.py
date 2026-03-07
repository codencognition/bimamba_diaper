#!/usr/bin/env python3

# Copyright 2026 FAST - NUCES, Islamabad (author: Usman)
# Licensed under the MIT license.

"""
Precompute features from KaldiDiarizationDataset and save as pickle files
for use with PrecomputedDiarizationDataset during training.

Usage:
    python diaper/precompute_features.py -c examples/train.yaml \
        --output-features-dir /root/workspace/precomputed_features/train \
        --split train

    python diaper/precompute_features.py -c examples/train.yaml \
        --output-features-dir /root/workspace/precomputed_features/validation \
        --split validation
"""

import os
import pickle
import numpy as np
import torch
import yamlargparse
from common_utils.diarization_dataset import KaldiDiarizationDataset
from tqdm import tqdm


def parse_arguments():
    parser = yamlargparse.ArgumentParser(
        description="Precompute features for DiaPer training"
    )
    parser.add_argument(
        "-c", "--config", help="config file path", action=yamlargparse.ActionConfigFile
    )
    # All the args from train.yaml that KaldiDiarizationDataset needs
    parser.add_argument("--context-size", type=int)
    parser.add_argument("--feature-dim", type=int)
    parser.add_argument("--frame-shift", type=int)
    parser.add_argument("--frame-size", type=int)
    parser.add_argument("--input-transform", default="", type=str)
    parser.add_argument("--min-length", default=0, type=int)
    parser.add_argument("--n-attractors", type=int)
    parser.add_argument("--num-frames", type=int)
    parser.add_argument("--num-speakers", type=int)
    parser.add_argument("--sampling-rate", type=int)
    parser.add_argument("--specaugment", type=bool, default=False)
    parser.add_argument("--subsampling", type=int)
    parser.add_argument("--time-shuffle", action="store_true")
    parser.add_argument("--use-last-samples", default=True, type=bool)
    parser.add_argument("--train-data-dir", default=None)
    parser.add_argument("--valid-data-dir", default=None)
    # Extra args for this script
    parser.add_argument(
        "--output-features-dir",
        type=str,
        required=True,
        help="Directory to save precomputed features",
    )
    parser.add_argument(
        "--split",
        type=str,
        required=True,
        choices=["train", "validation"],
        help="Which split to precompute",
    )
    parser.add_argument(
        "--samples-per-file",
        type=int,
        default=500,
        help="Number of samples per pickle file",
    )
    # Dummy args that may appear in the yaml but are not needed here
    parser.add_argument("--mamba-d-state", type=int, default=16)
    parser.add_argument("--mamba-d-conv", type=int, default=4)
    parser.add_argument("--mamba-expand", type=int, default=2)
    parser.add_argument("--activation-loss-BCE-weight", default=1.0, type=float)
    parser.add_argument("--activation-loss-DER-weight", default=0.0, type=float)
    parser.add_argument("--attractor-existence-loss-weight", default=1.0, type=float)
    parser.add_argument("--attractor-frame-comparison", default="dotprod", type=str)
    parser.add_argument("--att-qty-loss-weight", default=0.0, type=float)
    parser.add_argument("--condition-frame-encoder", type=bool, default=True)
    parser.add_argument("--context-activations", type=bool, default=False)
    parser.add_argument("--d-latents", type=int, default=None)
    parser.add_argument("--detach-attractor-loss", type=bool, default=False)
    parser.add_argument("--dev-batchsize", default=1, type=int)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--dropout_attractors", type=float, default=0.0)
    parser.add_argument("--dropout_frames", type=float, default=0.0)
    parser.add_argument("--frame-encoder-heads", type=int, default=4)
    parser.add_argument("--frame-encoder-layers", type=int, default=4)
    parser.add_argument("--frame-encoder-units", type=int, default=2048)
    parser.add_argument("--gpu", "-g", default=-1, type=int)
    parser.add_argument("--gradclip", default=-1, type=int)
    parser.add_argument("--init-epochs", type=str, default="")
    parser.add_argument("--init-model-path", type=str, default="")
    parser.add_argument("--intermediate-loss-frameencoder", default=False, type=bool)
    parser.add_argument("--intermediate-loss-perceiver", default=False, type=bool)
    parser.add_argument("--latents2attractors", type=str, default="dummy")
    parser.add_argument("--length-normalize", default=False, type=bool)
    parser.add_argument("--log-report-batches-num", default=1, type=float)
    parser.add_argument("--lr", type=float, default=0.0)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--model-type", default="AttractorsPath")
    parser.add_argument("--n-blocks-attractors", type=int, default=3)
    parser.add_argument("--n-blocks-embeddings", type=int, default=None)
    parser.add_argument("--n-embeddings", type=int, default=1)
    parser.add_argument("--n-internal-blocks-attractors", type=int, default=1)
    parser.add_argument("--n-latents", type=int, default=128)
    parser.add_argument("--n-sa-heads-attractors", type=int, default=4)
    parser.add_argument("--n-sa-heads-embeddings", type=int, default=None)
    parser.add_argument("--n-sa-heads-frames", type=int, default=None)
    parser.add_argument("--n-selfattends-attractors", type=int, default=2)
    parser.add_argument("--n-selfattends-embeddings", type=int, default=None)
    parser.add_argument("--n-speakers-softmax", type=int, default=0)
    parser.add_argument("--n-xa-heads-attractors", type=int, default=4)
    parser.add_argument("--n-xa-heads-embeddings", type=int, default=None)
    parser.add_argument("--n-xa-heads-in", type=int, default=1)
    parser.add_argument("--n-xa-heads-out", type=int, default=None)
    parser.add_argument("--noam-model-size", type=int, default=512)
    parser.add_argument("--noam-warmup-steps", type=float, default=200000)
    parser.add_argument("--norm-loss-per-spk", type=bool, default=False)
    parser.add_argument("--num-workers", default=1, type=int)
    parser.add_argument("--optimizer", type=str, default="noam")
    parser.add_argument("--osd-loss-weight", default=0.0, type=float)
    parser.add_argument("--output-path", type=str, default="")
    parser.add_argument("--posenc-maxlen", type=int, default=36000)
    parser.add_argument("--pre-xa-heads", type=int, default=4)
    parser.add_argument("--save-intermediate", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--shuffle-spk-order", type=bool, default=False)
    parser.add_argument("--speakerid-loss", type=str, default="")
    parser.add_argument("--speakerid-loss-weight", default=0.0, type=float)
    parser.add_argument("--speakerid-num-speakers", type=int, default=-1)
    parser.add_argument("--train-batchsize", default=1, type=int)
    parser.add_argument("--train-features-dir", default=None)
    parser.add_argument("--use-detection-error-rate", default=False, type=bool)
    parser.add_argument("--use-frame-selfattention", default=False, type=bool)
    parser.add_argument("--use-posenc", default=False, type=bool)
    parser.add_argument("--use-pre-crossattention", default=False, type=bool)
    parser.add_argument("--vad-loss-weight", default=0.0, type=float)
    parser.add_argument("--valid-features-dir", default=None)
    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()

    if args.split == "train":
        data_dir = args.train_data_dir
    else:
        data_dir = args.valid_data_dir

    assert data_dir is not None, f"Data dir for split '{args.split}' is not set"

    os.makedirs(args.output_features_dir, exist_ok=True)

    print(f"Loading dataset from {data_dir}...")
    dataset = KaldiDiarizationDataset(
        data_dir,
        chunk_size=args.num_frames,
        context_size=args.context_size,
        feature_dim=args.feature_dim,
        frame_shift=args.frame_shift,
        frame_size=args.frame_size,
        input_transform=args.input_transform,
        n_speakers=min(args.num_speakers, args.n_attractors),
        sampling_rate=args.sampling_rate,
        shuffle=args.time_shuffle,
        subsampling=args.subsampling,
        use_last_samples=args.use_last_samples,
        min_length=args.min_length,
        specaugment=args.specaugment,
    )

    print(f"Total samples: {len(dataset)}")
    print(
        f"Saving {args.samples_per_file} samples per file to {args.output_features_dir}"
    )

    xs, ts, names, begs, ends, spk_ids_list = [], [], [], [], [], []
    file_idx = 0

    for i in tqdm(range(len(dataset)), desc="Precomputing features"):
        x, t, name, beg, end, spk_ids = dataset[i]
        xs.append(x)
        ts.append(t)
        names.append(name)
        begs.append(beg)
        ends.append(end)
        spk_ids_list.append(spk_ids)

        if len(xs) >= args.samples_per_file:
            batch = {
                "xs": xs,
                "ts": ts,
                "names": names,
                "beg": begs,
                "end": ends,
                "spk_ids": spk_ids_list,
            }
            filepath = os.path.join(
                args.output_features_dir, f"batch_{file_idx:05d}.pkl"
            )
            with open(filepath, "wb") as f:
                pickle.dump(batch, f)
            print(f"Saved {filepath} ({len(xs)} samples)")
            xs, ts, names, begs, ends, spk_ids_list = [], [], [], [], [], []
            file_idx += 1

    # Save remaining samples
    if len(xs) > 0:
        batch = {
            "xs": xs,
            "ts": ts,
            "names": names,
            "beg": begs,
            "end": ends,
            "spk_ids": spk_ids_list,
        }
        filepath = os.path.join(args.output_features_dir, f"batch_{file_idx:05d}.pkl")
        with open(filepath, "wb") as f:
            pickle.dump(batch, f)
        print(f"Saved {filepath} ({len(xs)} samples)")

    print(f"Done! Saved {file_idx + 1} files to {args.output_features_dir}")


if __name__ == "__main__":
    main()
