# core/losses.py
"""
File: core/losses.py

ソースコードの役割:
本モジュールは、クラス不均衡に対処するためのFocal Lossなど、
PyTorchベースのカスタム損失関数の定義を提供します。
"""

import torch
import torch.nn as nn


class FocalLoss(nn.Module):
    """予測が容易なサンプルの重みを下げ、困難なサンプルに学習を集中させる損失関数。

    不均衡データ（Neutralクラスが圧倒的多数など）の学習において、
    クロスエントロピー損失を動的にスケーリングします。

    Attributes:
        gamma (float): フォーカシングパラメータ。大きいほど困難なサンプルを重視する。
        reduction (str): 損失の集約方法 ('mean', 'sum', 'none')。
        alpha (torch.Tensor, optional): クラスごとの重みテンソル。
    """

    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        """
        Args:
            alpha (torch.Tensor, optional): クラスの重み。デフォルトはNone。
            gamma (float, optional): フォーカシングパラメータ。デフォルトは2.0。
            reduction (str, optional): 集約方法。デフォルトは'mean'。
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.register_buffer("alpha", alpha)

    def forward(self, inputs, targets):
        """損失を計算します。

        Args:
            inputs (torch.Tensor): モデルの出力ロジット (Batch, NumClasses)。
            targets (torch.Tensor): 正解ラベル (Batch)。

        Returns:
            torch.Tensor: 計算された損失値。
        """
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)

        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
        else:
            loss = (1 - pt) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss
