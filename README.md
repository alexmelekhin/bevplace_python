# BEVPlace++ method packaged

## Installation

Requires Python 3.10+. Install with uv:

```bash
uv pip install -e .
```

## Usage

Two entrypoints:

- Build index (PCA off by default):
```bash
python -m bevplace.cli index build --map-dir /path/to/pcds_or_bins --out /path/to/index \
  --D 40.0 --g 0.4 --store-locals
```

- Localize a single query:
```bash
python -m bevplace.cli localize --db /path/to/index --bin /path/to/query.bin \
  --estimate-pose --pose-thresh-m 0.5 --pose-kps fast
```

Notes:
- BEV images are computed on-the-fly with upstream-compatible settings (voxel ds=g, per-cell cap=10, per-image max normalization).
- Weights are auto-downloaded from the official repo; if not available, you can pass `--weights PATH` during index build to use a local checkpoint.

## Acknowledgments

This package draws on ideas and assets from the original BEVPlace/BEVPlace++ project by Luo et al.

- Original repository: [BEVPlace2 (official)](https://github.com/zjuluolun/BEVPlace2)

Please cite the original papers if you use this work in academic contexts:

```
@ARTICLE{luo2024bevplaceplus,
      journal={IEEE Transactions on Robotics (T-RO)}, 
      title={BEVPlace++: Fast, Robust, and Lightweight LiDAR Global Localization for Unmanned Ground Vehicles}, 
      author={Lun Luo and Si-Yuan Cao and Xiaorui Li and Jintao Xu and Rui Ai and Zhu Yu and Xieyuanli Chen},
      volume={41},
      number={},
      pages={4479-4498},
      year={2025},
}
```

```
@INPROCEEDINGS{luo2023bevplace,
      author={Luo, Lun and Zheng, Shuhang and Li, Yixuan and Fan, Yongzhi and Yu, Beinan and Cao, Si-Yuan and Li, Junwei and Shen, Hui-Liang},
      booktitle={2023 IEEE/CVF International Conference on Computer Vision (ICCV)}, 
      title={BEVPlace: Learning LiDAR-based Place Recognition using Bird’s Eye View Images}, 
      year={2023},
      pages={8666-8675},
      doi={10.1109/ICCV51070.2023.00799}
}
```
