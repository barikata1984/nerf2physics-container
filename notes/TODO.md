# TODO

- [ ] フル MC (N=3, 約 50 分) を回すか判断し、回すなら `scripts/mc_variability.sh --n 3` ([議事録](LOGS/2026-09-23_nerf2physics-reproduction-and-density-volume.md))
- [ ] 未 push コミット `5d770f4` と未コミットのスクリプト (`compare_volumes.py`, `llm_robustness.py`, `mc_variability.sh`, `--nerf_seed`) をコミット・push ([議事録](LOGS/2026-09-23_nerf2physics-reproduction-and-density-volume.md))
- [ ] `lego/`, `loaded_cube_merged_seed42/` に同じ再現手順を適用 ([議事録](LOGS/2026-09-23_nerf2physics-reproduction-and-density-volume.md))
- [ ] 退避ディレクトリの削除可否を決める: `NeRF2Physics.local/`, `data/merged_seed42.bak/`, `ns_broken_24k_uniform/` ([議事録](LOGS/2026-09-23_nerf2physics-reproduction-and-density-volume.md))
- [ ] 公式 GPT-3.5 経路 (temperature 1.0) での材質辞書のブレを計測 (API キー要) ([議事録](LOGS/2026-09-23_nerf2physics-reproduction-and-density-volume.md))
