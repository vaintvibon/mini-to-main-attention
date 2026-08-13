# losses/diversity_loss.py

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class HeadDiversityLoss(nn.Module):
    """
    Active Main head들이 서로 같은 representation만 만들지 않도록 하는 diversity loss.

    입력:
        head_out: [B, H, N, Dh]
            Main attention의 head별 output.
            H = main_heads
            Dh = head_dim

        active_mask: [B, H]
            현재 budget에서 활성화된 Main head mask.

        direct_mask: Optional [B, H]
            direct-bound head mask.

        mixed_mask: Optional [B, H]
            mixed head mask.

    mode:
        "active":
            active head들끼리 전체 pairwise similarity를 낮춘다.

        "direct_mixed":
            direct-bound head와 mixed head 사이의 similarity를 낮춘다.
            네 연구 개념에는 이쪽이 더 잘 맞는다.
            단, budget=1이면 mixed head가 없으므로 loss=0이 된다.

    출력:
        scalar loss
    """

    def __init__(
        self,
        mode: str = "direct_mixed",
        eps: float = 1e-6,
    ):
        super().__init__()

        if mode not in ["active", "direct_mixed"]:
            raise ValueError(
                f"mode must be 'active' or 'direct_mixed', got {mode}."
            )

        self.mode = mode
        self.eps = eps

    def forward(
        self,
        head_out: torch.Tensor,
        active_mask: torch.Tensor,
        direct_mask: Optional[torch.Tensor] = None,
        mixed_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if head_out.dim() != 4:
            raise ValueError(
                f"Expected head_out shape [B, H, N, Dh], got {head_out.shape}."
            )

        if active_mask.dim() != 2:
            raise ValueError(
                f"Expected active_mask shape [B, H], got {active_mask.shape}."
            )

        B, H, N, Dh = head_out.shape

        if active_mask.shape != (B, H):
            raise ValueError(
                f"active_mask shape mismatch. "
                f"Expected {(B, H)}, got {active_mask.shape}."
            )

        # [B, H, N, Dh] -> [B, H, N*Dh]
        z = head_out.reshape(B, H, N * Dh)

        # head representation normalize
        z = F.normalize(z, dim=-1, eps=self.eps)

        # cosine similarity matrix: [B, H, H]
        sim = torch.matmul(z, z.transpose(-1, -2))

        if self.mode == "active":
            pair_mask = self._build_active_pair_mask(active_mask)

        else:
            if direct_mask is None or mixed_mask is None:
                raise ValueError(
                    "direct_mask and mixed_mask are required when mode='direct_mixed'."
                )

            pair_mask = self._build_direct_mixed_pair_mask(
                direct_mask=direct_mask,
                mixed_mask=mixed_mask,
            )

        pair_mask = pair_mask.to(device=head_out.device, dtype=head_out.dtype)

        # 자기 자신과의 similarity는 제외
        eye = torch.eye(H, device=head_out.device, dtype=torch.bool)
        eye = eye.unsqueeze(0)  # [1, H, H]
        pair_mask = pair_mask * (~eye).to(dtype=head_out.dtype)

        denom = pair_mask.sum()

        if denom.item() == 0:
            # 값은 0이지만 head_out과 graph 연결을 유지한다.
            # budget=1처럼 direct-mixed pair가 없는 경우에도 loss.backward()가 안전하다.
            return head_out.sum() * 0.0

        # similarity가 클수록 penalty
        loss = ((sim ** 2) * pair_mask).sum() / (denom + self.eps)

        return loss

    def _build_active_pair_mask(self, active_mask: torch.Tensor) -> torch.Tensor:
        """
        active head들끼리 pair를 만든다.

        active_mask: [B, H]
        output: [B, H, H]
        """
        active = active_mask.bool()
        pair_mask = active.unsqueeze(2) & active.unsqueeze(1)
        return pair_mask

    def _build_direct_mixed_pair_mask(
        self,
        direct_mask: torch.Tensor,
        mixed_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        direct head와 mixed head 사이의 pair만 만든다.

        direct_mask: [B, H]
        mixed_mask:  [B, H]
        output: [B, H, H]
        """
        direct = direct_mask.bool()
        mixed = mixed_mask.bool()

        direct_to_mixed = direct.unsqueeze(2) & mixed.unsqueeze(1)
        mixed_to_direct = mixed.unsqueeze(2) & direct.unsqueeze(1)

        pair_mask = direct_to_mixed | mixed_to_direct

        return pair_mask