# ISSUES

- NeRF2Physics の密度推定が 5 材質の平均に潰れる (木/鋼比 0.87 vs GT 0.070)。CLIP softmax (T=0.1) がほぼ均等になるため。手法の構造的な限界で、質量 −40%・重心 9.4 cm の主因 ([議事録](LOGS/2026-09-23_nerf2physics-reproduction-and-density-volume.md))
- upstream `ns_reconstruction.py` の既定サンプリング (`uniform / 0.4 / 6.0`) は小さく細い物体で色の sigmoid 飽和を起こす。既定値は upstream 互換のまま残しており、他データでも `--proposal_initial_sampler piecewise --near_plane --far_plane` の指定が必要 ([議事録](LOGS/2026-09-23_nerf2physics-reproduction-and-density-volume.md))
- `gpt-3.5-turbo` (既定 `--gpt_model_name`) は 2026-10-23 に停止予定 ([議事録](LOGS/2026-09-23_nerf2physics-reproduction-and-density-volume.md))
- `ns-download-data` の公式サンプルは Google Drive のレート制限で取得できなかった (自前データで代替済み) ([議事録](LOGS/2026-09-23_nerf2physics-reproduction-and-density-volume.md))
