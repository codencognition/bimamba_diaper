#!/usr/bin/env python3

# Copyright 2026 FAST - NUCES, Islamabad (author: Usman)
# Licensed under the MIT license.

"""
Evaluate a trained DiaPer model on a test set using checkpoints.
Computes and displays DER, loss, and other metrics in the console.

Usage:
    cd /root/workspace/diaper_local_base
    python diaper/evaluate.py -c examples/infer_test.yaml
"""

from backend.losses import get_loss, pad_labels_zeros, pad_sequence
from backend.models import (
    average_checkpoints,
    get_model,
)
from common_utils.diarization_dataset import KaldiDiarizationDataset
from common_utils.metrics import (
    calculate_metrics,
    new_metrics,
    reset_metrics,
    update_metrics,
)
from torch.utils.data import DataLoader
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple
import numpy as np
import os
import random
import torch
import yamlargparse


def _init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _convert(batch):
    return {
        "xs": [x for x, _, _, _, _, _ in batch],
        "ts": [t for _, t, _, _, _, _ in batch],
        "names": [r for _, _, r, _, _, _ in batch],
        "beg": [b for _, _, _, b, _, _ in batch],
        "end": [e for _, _, _, _, e, _ in batch],
        "spk_ids": [s for _, _, _, _, _, s in batch],
    }


def compute_loss_and_metrics(
    model, labels, inputs, n_speakers, spkid_labels, acum_metrics, args
):
    (
        all_frame_embs,
        per_frameenclayer_ys_logits,
        per_frameenclayer_attractors_logits,
        per_frameenclayer_attractors,
        per_prcvblock_ys_logits,
        per_prcvblock_attractors_logits,
        per_prcvblock_attractors,
        per_prcvblock_l2a_entropy_term,
        per_prcvblock_latents,
    ) = model.forward(inputs, args)

    y_probs = torch.sigmoid(per_frameenclayer_ys_logits[:, :, :, -1])
    (
        activation_loss_BCE,
        activation_loss_DER,
        attractor_existence_loss,
        att_qty_loss,
        vad_loss,
        osd_loss,
        spkid_loss,
    ) = get_loss(
        per_frameenclayer_ys_logits[:, :, :, -1],
        labels,
        n_speakers,
        per_frameenclayer_attractors_logits[:, :, -1],
        model,
        per_frameenclayer_attractors[:, :, :, -1],
        args.speakerid_num_speakers,
        spkid_labels,
        args,
    )

    l2a_entropy_term = per_prcvblock_l2a_entropy_term[-1]

    if args.intermediate_loss_frameencoder:
        intermediate_activation_losses_BCE = torch.zeros(
            per_frameenclayer_ys_logits.shape[-1] - 1
        )
        intermediate_activation_losses_DER = torch.zeros(
            per_frameenclayer_ys_logits.shape[-1] - 1
        )
        intermediate_attractor_existence_losses = torch.zeros(
            per_frameenclayer_ys_logits.shape[-1] - 1
        )
        intermediate_att_qty_losses = torch.zeros(
            per_frameenclayer_ys_logits.shape[-1] - 1
        )
        intermediate_vad_losses = torch.zeros(per_frameenclayer_ys_logits.shape[-1] - 1)
        intermediate_osd_losses = torch.zeros(per_frameenclayer_ys_logits.shape[-1] - 1)
        intermediate_spkid_losses = torch.zeros(
            per_frameenclayer_ys_logits.shape[-1] - 1
        )
        for j in range(per_frameenclayer_ys_logits.shape[-1] - 1):
            (
                activation_loss_BCE_j,
                activation_loss_DER_j,
                attractor_existence_loss_j,
                att_qty_loss_j,
                vad_loss_j,
                osd_loss_j,
                spkid_loss_j,
            ) = get_loss(
                per_frameenclayer_ys_logits[:, :, :, j],
                labels,
                n_speakers,
                per_frameenclayer_attractors_logits[:, :, j],
                model,
                per_frameenclayer_attractors[:, :, :, j],
                args.speakerid_num_speakers,
                spkid_labels,
                args,
            )
            intermediate_activation_losses_BCE[j] = activation_loss_BCE_j
            intermediate_activation_losses_DER[j] = activation_loss_DER_j
            intermediate_attractor_existence_losses[j] = attractor_existence_loss_j
            intermediate_att_qty_losses[j] = att_qty_loss_j
            intermediate_vad_losses[j] = vad_loss_j
            intermediate_osd_losses[j] = osd_loss_j
            intermediate_spkid_losses[j] = spkid_loss_j
        activation_loss_BCE += torch.mean(intermediate_activation_losses_BCE)
        activation_loss_DER += torch.mean(intermediate_activation_losses_DER)
        attractor_existence_loss += torch.mean(intermediate_attractor_existence_losses)
        att_qty_loss += torch.mean(intermediate_att_qty_losses)
        vad_loss += torch.mean(intermediate_vad_losses)
        osd_loss += torch.mean(intermediate_osd_losses)
        spkid_loss += torch.mean(intermediate_spkid_losses)

    if args.intermediate_loss_perceiver:
        intermediate_activation_losses_BCE = torch.zeros(
            per_prcvblock_ys_logits.shape[-1] - 1
        )
        intermediate_activation_losses_DER = torch.zeros(
            per_prcvblock_ys_logits.shape[-1] - 1
        )
        intermediate_attractor_existence_losses = torch.zeros(
            per_prcvblock_ys_logits.shape[-1] - 1
        )
        intermediate_att_qty_losses = torch.zeros(per_prcvblock_ys_logits.shape[-1] - 1)
        intermediate_vad_losses = torch.zeros(per_prcvblock_ys_logits.shape[-1] - 1)
        intermediate_osd_losses = torch.zeros(per_prcvblock_ys_logits.shape[-1] - 1)
        intermediate_spkid_losses = torch.zeros(per_prcvblock_ys_logits.shape[-1] - 1)
        for i in range(per_prcvblock_ys_logits.shape[-1] - 1):
            (
                activation_loss_BCE_i,
                activation_loss_DER_i,
                attractor_existence_loss_i,
                att_qty_loss_i,
                vad_loss_i,
                osd_loss_i,
                spkid_loss_i,
            ) = get_loss(
                per_prcvblock_ys_logits[:, :, :, i],
                labels,
                n_speakers,
                per_prcvblock_attractors_logits[:, :, i],
                model,
                per_prcvblock_attractors[:, :, :, i],
                args.speakerid_num_speakers,
                spkid_labels,
                args,
            )
            intermediate_activation_losses_BCE[i] = activation_loss_BCE_i
            intermediate_activation_losses_DER[i] = activation_loss_DER_i
            intermediate_attractor_existence_losses[i] = attractor_existence_loss_i
            intermediate_att_qty_losses[i] = att_qty_loss_i
            intermediate_vad_losses[i] = vad_loss_i
            intermediate_osd_losses[i] = osd_loss_i
            intermediate_spkid_losses[i] = spkid_loss_i
        activation_loss_BCE += torch.mean(intermediate_activation_losses_BCE)
        l2a_entropy_term += torch.mean(per_prcvblock_l2a_entropy_term[:-1])
        activation_loss_DER += torch.mean(intermediate_activation_losses_DER)
        attractor_existence_loss += torch.mean(intermediate_attractor_existence_losses)
        att_qty_loss += torch.mean(intermediate_att_qty_losses)
        vad_loss += torch.mean(intermediate_vad_losses)
        osd_loss += torch.mean(intermediate_osd_losses)
        spkid_loss += torch.mean(intermediate_spkid_losses)

    loss = (
        activation_loss_BCE * args.activation_loss_BCE_weight
        + l2a_entropy_term
        + activation_loss_DER * args.activation_loss_DER_weight
        + attractor_existence_loss * args.attractor_existence_loss_weight
        + att_qty_loss * args.att_qty_loss_weight
        + vad_loss * args.vad_loss_weight
        + osd_loss * args.osd_loss_weight
        + spkid_loss * args.speakerid_loss_weight
    )

    metrics = calculate_metrics(labels.detach(), y_probs.detach(), threshold=0.5)

    acum_metrics = update_metrics(acum_metrics, metrics)
    acum_metrics["loss"] += loss.item()
    acum_metrics["activation_loss_BCE"] += activation_loss_BCE.item()
    acum_metrics["l2a_entropy_term"] += l2a_entropy_term.item()
    acum_metrics["activation_loss_DER"] += activation_loss_DER.item()
    acum_metrics["attractor_existence_loss"] += attractor_existence_loss.item()
    acum_metrics["att_qty_loss"] += att_qty_loss.item()
    acum_metrics["vad_loss"] += vad_loss.item()
    acum_metrics["osd_loss"] += osd_loss.item()
    acum_metrics["spkid_loss"] += spkid_loss.item()
    return loss, acum_metrics


def parse_arguments():
    parser = yamlargparse.ArgumentParser(
        description="Evaluate DiaPer model on test set"
    )
    parser.add_argument(
        "-c", "--config", help="config file path", action=yamlargparse.ActionConfigFile
    )
    parser.add_argument("--activation-loss-BCE-weight", default=1.0, type=float)
    parser.add_argument("--activation-loss-DER-weight", default=0.0, type=float)
    parser.add_argument("--attractor-existence-loss-weight", default=1.0, type=float)
    parser.add_argument("--attractor-frame-comparison", default="dotprod", type=str)
    parser.add_argument("--att-qty-loss-weight", default=0.0, type=float)
    parser.add_argument("--att-qty-reg-loss-weight", default=0.0, type=float)
    parser.add_argument("--condition-frame-encoder", type=bool, default=True)
    parser.add_argument("--context-activations", type=bool, default=False)
    parser.add_argument("--context-size", type=int)
    parser.add_argument("--d-latents", type=int)
    parser.add_argument("--detach-attractor-loss", default=False, type=bool)
    parser.add_argument("--dropout_attractors", type=float, default=0.0)
    parser.add_argument("--dropout_frames", type=float, default=0.0)
    parser.add_argument(
        "--epochs",
        type=str,
        help="epochs to average separated by commas or - for intervals.",
    )
    parser.add_argument("--estimate-spk-qty", default=-1, type=int)
    parser.add_argument("--estimate-spk-qty-thr", default=-1, type=float)
    parser.add_argument("--feature-dim", type=int)
    parser.add_argument("--frame-encoder-heads", type=int, default=4)
    parser.add_argument("--frame-encoder-layers", type=int, default=4)
    parser.add_argument("--frame-encoder-units", type=int, default=2048)
    parser.add_argument("--mamba-d-state", type=int, default=16)
    parser.add_argument("--mamba-d-conv", type=int, default=4)
    parser.add_argument("--mamba-expand", type=int, default=2)
    parser.add_argument("--frame-shift", type=int)
    parser.add_argument("--frame-size", type=int)
    parser.add_argument("--gpu", "-g", default=-1, type=int)
    parser.add_argument(
        "--infer-data-dir",
        type=str,
        default=None,
        help="test data directory (Kaldi-style)",
    )
    parser.add_argument("--input-transform", default="", type=str)
    parser.add_argument("--intermediate-loss-frameencoder", default=False, type=bool)
    parser.add_argument("--intermediate-loss-perceiver", default=False, type=bool)
    parser.add_argument("--latents2attractors", type=str, default="linear")
    parser.add_argument("--length-normalize", default=False, type=bool)
    parser.add_argument("--log-report-batches-num", default=1, type=float)
    parser.add_argument("--median-window-length", default=11, type=int)
    parser.add_argument("--model-type", default="AttractorsPath", type=str)
    parser.add_argument(
        "--models-path", type=str, default=None, help="directory with model checkpoints"
    )
    parser.add_argument("--n-attractors", type=int)
    parser.add_argument("--n-blocks-attractors", type=int, default=3)
    parser.add_argument("--n-internal-blocks-attractors", type=int, default=1)
    parser.add_argument("--n-latents", type=int, default=128)
    parser.add_argument("--n-selfattends-attractors", type=int, default=2)
    parser.add_argument("--n-sa-heads-attractors", type=int, default=4)
    parser.add_argument("--n-xa-heads-attractors", type=int, default=4)
    parser.add_argument("--norm-loss-per-spk", type=bool, default=False)
    parser.add_argument("--normalize-probs", default=False, type=bool)
    parser.add_argument("--num-frames", default=-1, type=int)
    parser.add_argument("--num-speakers", type=int, default=2)
    parser.add_argument("--osd-loss-weight", default=0.0, type=float)
    parser.add_argument("--posenc-maxlen", type=int, default=36000)
    parser.add_argument("--pre-xa-heads", type=int, default=4)
    parser.add_argument("--rttms-dir", type=str, default="")
    parser.add_argument("--sampling-rate", type=int)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--speakerid-loss", type=str, default="")
    parser.add_argument("--speakerid-loss-weight", default=0.0, type=float)
    parser.add_argument("--speakerid-num-speakers", type=int, default=-1)
    parser.add_argument("--specaugment", type=bool, default=False)
    parser.add_argument("--subsampling", type=int)
    parser.add_argument("--threshold", default=0.5, type=float)
    parser.add_argument("--time-shuffle", action="store_true")
    parser.add_argument("--use-frame-selfattention", default=False, type=bool)
    parser.add_argument("--use-posenc", default=False, type=bool)
    parser.add_argument("--use-pre-crossattention", default=False, type=bool)
    parser.add_argument("--vad-loss-weight", default=0.0, type=float)
    parser.add_argument("--shuffle-spk-order", type=bool, default=False)
    parser.add_argument("--use-detection-error-rate", default=False, type=bool)
    # eval-specific args
    parser.add_argument(
        "--eval-batchsize", default=128, type=int, help="batch size for evaluation"
    )
    parser.add_argument("--num-workers", default=4, type=int)
    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()

    # For reproducibility
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ["PYTHONHASHSEED"] = str(args.seed)

    if args.gpu >= 1:
        args.device = torch.device("cuda")
    else:
        args.device = torch.device("cpu")

    print(f"Loading test data from: {args.infer_data_dir}")
    print(f"Loading checkpoints from: {args.models_path}")
    print(f"Averaging epochs: {args.epochs}")
    print(f"Device: {args.device}")
    print()

    # Load test dataset
    test_set = KaldiDiarizationDataset(
        args.infer_data_dir,
        chunk_size=args.num_frames,
        context_size=args.context_size,
        feature_dim=args.feature_dim,
        frame_shift=args.frame_shift,
        frame_size=args.frame_size,
        input_transform=args.input_transform,
        n_speakers=args.num_speakers,
        sampling_rate=args.sampling_rate,
        shuffle=args.time_shuffle,
        subsampling=args.subsampling,
        use_last_samples=True,
        min_length=0,
        specaugment=False,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=args.eval_batchsize,
        collate_fn=_convert,
        num_workers=args.num_workers,
        shuffle=False,
        worker_init_fn=_init_fn,
    )

    print(f"Test samples: {len(test_set)}")
    print(f"Batch size: {args.eval_batchsize}")
    print()

    # Load model and average checkpoints
    model = get_model(args)
    model = average_checkpoints(args.device, model, args.models_path, args.epochs)
    model.eval()

    # Evaluate
    acum_metrics = new_metrics()
    batches_qty = 0

    print("Running evaluation...")
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            batches_qty += 1
            features = batch["xs"]
            labels = batch["ts"]
            spkids = batch["spk_ids"]
            n_speakers = np.asarray(
                [
                    max(torch.where(t.sum(0) != 0)[0]) + 1 if t.sum() > 0 else 0
                    for t in labels
                ]
            )
            max_n_speakers = args.n_attractors
            # For variable-length evaluation (num_frames=-1), pad to max length in batch
            seq_len = args.num_frames
            if seq_len <= 0:
                seq_len = max(f.shape[0] for f in features)
            features, labels = pad_sequence(features, labels, seq_len)
            labels = pad_labels_zeros(labels, max_n_speakers)
            features = torch.stack(features).to(args.device)
            labels = torch.stack(labels).to(args.device)
            _, acum_metrics = compute_loss_and_metrics(
                model, labels, features, n_speakers, spkids, acum_metrics, args
            )
            if (i + 1) % 10 == 0:
                print(f"  Processed batch {i+1}/{len(test_loader)}")

    # Print results
    n = batches_qty
    print()
    print("=" * 70)
    print("TEST SET RESULTS")
    print("=" * 70)
    print(f"  Batches evaluated:  {n}")
    print(f"  Loss:               {acum_metrics['loss']/n:.4f}")
    print(f"  DER:                {acum_metrics['DER']/n:.2f}%")
    print(f"    - Miss:           {acum_metrics['DER_miss']/n:.2f}%")
    print(f"    - False Alarm:    {acum_metrics['DER_FA']/n:.2f}%")
    print(f"    - Confusion:      {acum_metrics['DER_conf']/n:.2f}%")
    print(f"  VAD Miss:           {acum_metrics['VAD_miss']/n:.2f}%")
    print(f"  VAD FA:             {acum_metrics['VAD_FA']/n:.2f}%")
    print(f"  OSD Miss:           {acum_metrics['OSD_miss']/n:.2f}%")
    print(f"  OSD FA:             {acum_metrics['OSD_FA']/n:.2f}%")
    print(f"  Avg ref spk qty:    {acum_metrics['avg_ref_spk_qty']/n:.2f}")
    print(f"  Avg pred spk qty:   {acum_metrics['avg_pred_spk_qty']/n:.2f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
