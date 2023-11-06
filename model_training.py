import pandas as pd
import torch
from pytorch_tabnet.tab_model import TabNetRegressor
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor


def train_tabnet(X_train, y_train, X_test, y_test):
    regressor = TabNetRegressor(
        n_d=12,
        n_a=12,
        n_steps=3,
        gamma=1.3,
        n_independent=2,
        n_shared=2,
        lambda_sparse=1e-3,
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=0.02),
        mask_type="sparsemax",
        scheduler_params=dict(mode="min", patience=8, min_lr=1e-8, factor=0.9),
        scheduler_fn=torch.optim.lr_scheduler.ReduceLROnPlateau,
        seed=42,
        verbose=1,
    )
    regressor.fit(
        X_train.values,
        y_train.values.reshape(-1, 1),
        eval_set=[(X_test.values, y_test.values.reshape(-1, 1))],
        eval_name=["valid"],
        eval_metric=["mae"],
        max_epochs=1,
        patience=10,
        batch_size=256 * 4,
        virtual_batch_size=128,
        num_workers=0,
        drop_last=False,
        loss_fn=torch.nn.functional.l1_loss,
    )
    return regressor


def train_trees(X_train, y_train):
    rf = RandomForestRegressor(
        n_estimators=30, max_depth=30, random_state=42, verbose=1
    )
    regressor = AdaBoostRegressor(
        estimator=rf,
        n_estimators=30,
        learning_rate=0.05,
        random_state=42,
    )
    regressor.fit(X_train, y_train)

    return regressor
