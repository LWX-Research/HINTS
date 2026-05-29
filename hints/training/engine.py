from __future__ import annotations

from pathlib import Path

import torch
import torch.optim as optim
from sklearn.model_selection import KFold
from tqdm import tqdm

from ..dataio import build_dataset_3d, create_data_loaders_3d
from ..losses import contrastive_jsd_loss, cox_loss
from ..metrics import auc_score, c_index
from ..models.hints_model import MultiModalGCN
from ..runtime import configure_environment, seed_everything
from ..settings import Settings
from .logging import log_epoch_info, report_results, save_fold_metrics_csv, setup_logger


def load_pretrained_weights(model: torch.nn.Module, pretrained_path: str | None, device: str) -> None:
    if not pretrained_path:
        return
    checkpoint = torch.load(pretrained_path, map_location=device)
    model_state = model.state_dict()
    compatible = {
        key: value for key, value in checkpoint.items()
        if key in model_state and value.size() == model_state[key].size()
    }
    model_state.update(compatible)
    model.load_state_dict(model_state)
    print(f"Loaded pretrained: {len(compatible)}/{len(model_state)} layers from {pretrained_path}")


def evaluate_model(model: torch.nn.Module, data_loader, settings: Settings) -> tuple[float, float]:
    model.eval()
    predictions, events, times = [], [], []
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="eval", leave=False):
            batch = batch.to(settings.device, non_blocking=True)
            pred, _, _ = model(batch)
            predictions.append(pred.view(-1))
            events.append(batch.dead)
            times.append(batch.deadtime)
    predictions = torch.cat(predictions).cpu().numpy()
    events = torch.cat(events).cpu().numpy()
    times = torch.cat(times).cpu().numpy()
    return c_index(times, -predictions, events), auc_score(predictions, events, times)


def train_one_fold(train_loader, val_loader, settings: Settings, fold: int, log_dir: Path, logger):
    model = MultiModalGCN(settings).to(settings.device)
    load_pretrained_weights(model, settings.pretrained_path, settings.device)

    optimizer = optim.SGD(model.parameters(), lr=settings.learning_rate, momentum=0.937)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=settings.epochs, eta_min=settings.learning_rate * settings.eta_min,
    )

    best_ci = best_auc = 0.0
    last_train_c = last_train_auc = float("nan")
    last_valid_c = last_valid_auc = float("nan")
    epoch_records: list[dict] = []

    for epoch in tqdm(range(settings.epochs), desc=f"Fold {fold}", total=settings.epochs):
        model.train()
        epoch_loss = cox_sum = contra_sum = 0.0
        n_batches = 0

        for batch in tqdm(train_loader, desc="train", leave=False):
            optimizer.zero_grad(set_to_none=True)
            batch = batch.to(settings.device, non_blocking=True)
            pred, share, distinct = model(batch)
            loss_cox = cox_loss(pred, batch, model, settings.lambda_1)
            loss_contra = contrastive_jsd_loss(share, distinct, settings.tau)
            loss = loss_cox + settings.contra_weight * loss_contra
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            cox_sum += loss_cox.item()
            contra_sum += loss_contra.item()
            n_batches += 1

        epoch_loss /= n_batches
        cox_sum /= n_batches
        contra_sum /= n_batches

        should_eval = ((epoch + 1) % settings.eval_interval == 0) or (epoch == settings.epochs - 1)
        train_c, train_auc = last_train_c, last_train_auc
        valid_c, valid_auc = last_valid_c, last_valid_auc

        if should_eval:
            if settings.eval_train_metrics:
                train_c, train_auc = evaluate_model(model, train_loader, settings)
                last_train_c, last_train_auc = train_c, train_auc
            valid_c, valid_auc = evaluate_model(model, val_loader, settings)
            last_valid_c, last_valid_auc = valid_c, valid_auc
            if valid_c > best_ci and valid_auc > best_auc:
                best_ci = valid_c
                best_auc = valid_auc
                torch.save(model.state_dict(), log_dir / f"best_model_fold{fold}.pth")

        record = log_epoch_info(logger, settings, fold, epoch, epoch_loss, cox_sum, contra_sum,
                                train_c, train_auc, valid_c, valid_auc)
        epoch_records.append(record)
        scheduler.step()

    torch.save(model.state_dict(), log_dir / f"final_model_fold{fold}.pth")
    save_fold_metrics_csv(fold, epoch_records, log_dir)
    return last_train_c, last_train_auc, best_ci, best_auc


def run_cross_validation(settings: Settings):
    configure_environment()
    seed_everything(settings.seed)
    logger, log_dir = setup_logger(settings)
    logger.info("===== %d-Fold Cross Validation (3D) =====", settings.k_fold)

    dataset = build_dataset_3d(settings)
    logger.info("Total cases: %d", len(dataset))

    if hasattr(dataset, "warm_cache"):
        dataset.warm_cache()

    splitter = KFold(n_splits=settings.k_fold, shuffle=True, random_state=settings.seed)
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(splitter.split(range(len(dataset))), start=1):
        logger.info("\n=== Fold %d/%d ===", fold, settings.k_fold)
        logger.info("Train: %d, Val: %d", len(train_idx), len(val_idx))
        train_loader, val_loader = create_data_loaders_3d(dataset, train_idx, val_idx, settings)
        metrics = train_one_fold(train_loader, val_loader, settings, fold, log_dir, logger)
        fold_metrics.append((fold, *metrics))

    report_results(logger, fold_metrics, log_dir)
    return log_dir


def evaluate_fold(settings: Settings, target_fold: int):
    configure_environment()
    seed_everything(settings.seed)
    dataset = build_dataset_3d(settings)
    if hasattr(dataset, "warm_cache"):
        dataset.warm_cache()
    splitter = KFold(n_splits=settings.k_fold, shuffle=True, random_state=settings.seed)

    for fold, (train_idx, val_idx) in enumerate(splitter.split(range(len(dataset))), start=1):
        if fold != target_fold:
            continue
        _, val_loader = create_data_loaders_3d(dataset, train_idx, val_idx, settings)
        model = MultiModalGCN(settings).to(settings.device)
        load_pretrained_weights(model, settings.pretrained_path, settings.device)
        valid_c, valid_auc = evaluate_model(model, val_loader, settings)
        print(f"[Fold {fold}] Val CI: {valid_c:.4f} | Val AUC: {valid_auc:.4f}")
        return valid_c, valid_auc
    raise ValueError(f"Fold {target_fold} out of range for k_fold={settings.k_fold}")
