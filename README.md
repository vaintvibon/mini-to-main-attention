# Mini-to-Main Attention

This repository implements a Mini-to-Main Two-Level Head Scheduler for Vision Transformers.

The core idea is to use Mini-attention not as a simple auxiliary branch, but as a lightweight scheduler. Mini-attention first computes coarse relation confidence using pooled key-value attention. Then, a Mini-to-Main allocator assigns a limited Main-attention head budget into direct-bound, mixed, and inactive heads.

## Current Status

Implemented:

- MiniAttention
- MiniImportance
- MiniToMainAllocator
- TwoLevelHeadScheduler
- TwoLevelMiniMainAttention
- MiniGuidedBlock
- MiniGuidedViT
- HeadDiversityLoss
- CIFAR-10 sanity training script

Current implementation is v1.

Important limitation:

- v1 computes all Main heads and masks inactive head outputs.
- Therefore, v1 validates the architecture but does not provide real computational speedup.
- v2 should implement selective computation by calculating only active Main heads.

## Quick Test

```bash
python test_forward_mini_attention.py
python test_forward_allocator.py
python test_forward_scheduler.py
python test_forward_two_level_attention.py
python test_forward_block.py
python test_forward_vit.py
python test_train_step.py
python test_diversity_loss.py

##CIFAR-10 Sanity training
python train_cifar10_sanity.py

