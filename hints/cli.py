from __future__ import annotations

import argparse

from .dataio.unified import convert_to_unified
from .preprocess.pipeline_3d import run_3d_preprocessing
from .runtime import configure_environment
from .settings import Settings
from .training.engine import evaluate_fold, run_cross_validation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="HINTS — 3D multimodal MRI survival analysis CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ---- convert ----
    conv_parser = subparsers.add_parser("convert", help="Convert raw dataset to unified canonical format")
    conv_parser.add_argument("--raw-dir", required=True, help="Raw dataset directory (NPC NIfTI, NPC NPY, or BraTS)")
    conv_parser.add_argument("--output-dir", required=True, help="Output unified directory")
    conv_parser.add_argument("--source-type", choices=["npc_nifti", "npc_npy", "brats"],
                             help="Source type (auto-detected if omitted)")
    conv_parser.add_argument("--case-id", help="Convert single case")
    conv_parser.add_argument("--max-cases", type=int, help="Limit number of cases")

    # ---- train ----
    train_parser = subparsers.add_parser("train", help="Run 3D cross-validation training")
    _add_common_overrides(train_parser)

    # ---- eval ----
    eval_parser = subparsers.add_parser("eval", help="Evaluate one fold")
    _add_common_overrides(eval_parser)
    eval_parser.add_argument("--fold", type=int, default=1, help="Fold index (1-based)")

    # ---- preprocess-3d ----
    pre_parser = subparsers.add_parser("preprocess-3d", help="Offline 3D graph preprocessing from unified format")
    pre_parser.add_argument("--unified-dir", required=True, help="Unified format directory")
    pre_parser.add_argument("--output-dir", required=True, help="Output directory for 3D graphs")
    pre_parser.add_argument("--n-segments", type=int, default=200, help="Target supervoxel count")
    pre_parser.add_argument("--num-nodes", type=int, default=None,
                            help="Graph node capacity (default: n-segments)")
    pre_parser.add_argument("--roi-margin", type=int, default=5)
    pre_parser.add_argument("--compactness", type=float, default=0.1)
    pre_parser.add_argument("--sigma", type=float, default=1.0)
    pre_parser.add_argument("--supervoxel-backend", choices=["skimage"], default="skimage")
    pre_parser.add_argument("--case-id", help="Process single case")
    pre_parser.add_argument("--max-cases", type=int, help="Limit number of cases")
    pre_parser.add_argument("--overwrite", action="store_true")

    return parser


def _add_common_overrides(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", dest="batch_size", type=int)
    parser.add_argument("--num-workers", dest="num_workers", type=int)
    parser.add_argument("--k-fold", dest="k_fold", type=int)
    parser.add_argument("--eval-interval", dest="eval_interval", type=int)
    parser.add_argument("--pretrained-path", dest="pretrained_path")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--unified-dir", dest="unified_dir", help="Unified format directory for online graph building")
    parser.add_argument("--raw-dir", dest="raw_data_dir", help="(deprecated) Raw NIfTI directory")
    parser.add_argument("--preprocess-output-dir", dest="preprocess_output_dir",
                        help="Directory of precomputed 3D graphs")
    parser.add_argument("--num-nodes", dest="num_nodes", type=int, help="Graph node capacity")
    parser.add_argument("--n-segments", dest="n_segments", type=int, help="Target supervoxel count")
    parser.add_argument("--hidden-dim", dest="hidden_dim", type=int, help="GCN hidden dimension")
    parser.add_argument("--roi-margin", dest="roi_margin", type=int)
    parser.add_argument("--slic-compactness", dest="slic_compactness", type=float)
    parser.add_argument("--slic-sigma", dest="slic_sigma", type=float)
    parser.add_argument("--supervoxel-backend", dest="supervoxel_backend", choices=["skimage"])
    parser.add_argument("--graph-build-mode", dest="graph_build_mode", choices=["offline", "online", "auto"])
    parser.add_argument("--contra-weight", dest="contra_weight", type=float)
    parser.add_argument("--lr", dest="learning_rate", type=float)


def main():
    configure_environment()
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "convert":
        convert_to_unified(
            raw_dir=args.raw_dir,
            output_dir=args.output_dir,
            source_type=getattr(args, "source_type", None),
            case_id=getattr(args, "case_id", None),
            max_cases=getattr(args, "max_cases", None),
        )

    elif args.command in ("train", "eval"):
        settings = Settings().apply_overrides(
            epochs=getattr(args, "epochs", None),
            batch_size=getattr(args, "batch_size", None),
            num_workers=getattr(args, "num_workers", None),
            k_fold=getattr(args, "k_fold", None),
            eval_interval=getattr(args, "eval_interval", None),
            pretrained_path=getattr(args, "pretrained_path", None),
            seed=getattr(args, "seed", None),
            unified_dir=getattr(args, "unified_dir", None),
            raw_data_dir=getattr(args, "raw_data_dir", None),
            preprocess_output_dir=getattr(args, "preprocess_output_dir", None),
            num_nodes=getattr(args, "num_nodes", None),
            n_segments=getattr(args, "n_segments", None),
            roi_margin=getattr(args, "roi_margin", None),
            slic_compactness=getattr(args, "slic_compactness", None),
            slic_sigma=getattr(args, "slic_sigma", None),
            supervoxel_backend=getattr(args, "supervoxel_backend", None),
            graph_build_mode=getattr(args, "graph_build_mode", None),
            contra_weight=getattr(args, "contra_weight", None),
            learning_rate=getattr(args, "learning_rate", None),
            hidden_dim=getattr(args, "hidden_dim", None),
        )

        if args.command == "train":
            run_cross_validation(settings)
        else:
            evaluate_fold(settings, target_fold=args.fold)

    elif args.command == "preprocess-3d":
        run_3d_preprocessing(
            unified_dir=args.unified_dir,
            output_dir=args.output_dir,
            n_segments=args.n_segments,
            num_nodes=args.num_nodes,
            roi_margin=args.roi_margin,
            compactness=args.compactness,
            sigma=args.sigma,
            supervoxel_backend=args.supervoxel_backend,
            case_id=getattr(args, "case_id", None),
            max_cases=getattr(args, "max_cases", None),
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
