# 2026-09-23 NeRF2Physics 再現と voxel 密度ボリューム拡張

## Topic
NeRF2Physics の再現と voxel 密度ボリューム拡張

## Summary
NeRF2Physics の推定ロジックは無改造のまま動かした。AABB voxel 格子で密度を問い合わせる拡張を加え、別マシンで再現できる形に整えた。再現手順の正本は `NeRF2Physics/README_LOCAL.md` である。
自前データでは upstream が固定している NeRF サンプリング設定が色の sigmoid 飽和を起こしていた。FTA で原因を特定し、サンプリング設定を引数化して解消した。
GT との比較では形状は一致したが、密度は 5 材質の平均に潰れ、木/鋼の密度比は GT 0.070 に対し 0.87 だった。この平坦化が重心誤差 9.3 cm の主因であり、質量 −40% は頭の過小と柄の過大が相殺した結果である。
推定のランダムなブレは質量で ±1.6% (NeRF 乱数) と ±1.5% (LLM 辞書) にとどまり、GT との系統誤差より 1〜2 桁小さい。フル MC は N=3 で十分と見積もり、実行は保留した。

## History
作業開始時、マシンには Python も conda も uv も無かった。この devcontainer は直前のコミットで Pixi 連携を外していたためである。環境管理ツールの候補を 3 つ挙げた。

- uv
- Miniconda
- pixi

nerfstudio は公式リポジトリに `pixi.toml` を持ち、pixi に対応している。ユーザーはこれを理由に pixi を指定した。公式 `pixi.toml` は CUDA 11.8 + PyTorch 2.2 + colmap の組み合わせを固定していた。この構成に NeRF2Physics の `requirements.txt` を pypi 依存として足す形で `NeRF2Physics/pixi.toml` を作った。CUDA 11.8 は pixi 環境内に閉じている。pixi 環境内の `nvcc` が `.pixi/envs/default/bin/nvcc` を指し、システムの CUDA 12.8 は変更していないことを確認した。

`pixi install` 後に依存衝突が 3 件出た。

- NumPy 2.x が conda 版 PyTorch 2.2 の C-API と合わない
- setuptools 84 が `pkg_resources` を削除しており tiny-cuda-nn がビルドできない
- システム gcc 13 を CUDA 11.8 の nvcc が拒否する

いずれも pixi 側のピン留めで解消した。後日 `transformers` 5.x が `torch>=2.5` を要求して BLIP-2 を読み込めない問題も出た。`transformers<4.50` を追加した。ピンの一覧は `NeRF2Physics/README_LOCAL.md` の "Known dependency conflicts" にある。

作業場所について、`/workspace` がテンプレートリポジトリそのものである点を懸念して別ディレクトリを提案した。ユーザーはテンプレートを使ってこのプロジェクトを進めるだけだと指示した。`/workspace/NeRF2Physics` を作業場所とした。

公式リポジトリの調査で注意事項が 3 点見つかった。

- `captioning.py` と `material_proposal.py` は `info_new.json` に書くが、`predict_property.py` の既定は `info.json` を読む
- `utils.load_images()` は RGBA を前提にアルファをマスクとして使う
- `predict_physical_property_query()` の query 点は nerfstudio の内部正規化座標ではなく `transforms.json` の座標系である

この 3 点は拡張スクリプトの既定値と README_LOCAL.md に反映した。

大容量ダウンロードは段階的に承認を取る方針にした。ユーザーは最初、環境構築と Nerfstudio 動作確認までに範囲を限定した。後に BLIP-2 のダウンロードを許可した。Hugging Face の BLIP-2 リポジトリは safetensors と `pytorch_model-*.bin` の同じ重みを二重に持っていた。途中で 30 GB を超えたため中断した。`lfs.fetchexclude` で `.bin` を除外し、safetensors 15.8 GB だけを取得した。

OpenAI 経路について、公式コードは `openai==0.28` の旧 API を直接呼ぶ。SDK を旧版にピンする案と現行 SDK にパッチする案を比べた。タスク指示の「最小変更で動く方」に従い旧版ピンを採った。

ユーザーが Anthropic のサブスクリプションを持つと述べた。`--llm_provider anthropic` 経路を追加し、OpenAI 経路は既定のまま残した。あわせて、ChatGPT や Claude のサブスクリプションは API 課金と別であることを確認した。

API キーは用意できなかった。材質提案は公式プロンプトを Opus 5 サブエージェントに渡して代行した。回答は公式パーサ `parse_material_list()` で形式検証してから `info_new.json` に書いた。ユーザーはこの代行を承認した。

自前データは `merged_seed42` である。中身はシミュレーション由来で、`complete/*.png` は RGBA でアルファが `masks/` と画素単位で一致した。`transforms.json` は nerfstudio 形式だった。`ground_truth.csv` には 128³ 格子の真の質量密度と総質量・重心・慣性テンソルが入っていた。元データは触らず、`data/merged_seed42/scenes/sledgehammer/` に symlink で配置した。

nerfstudio 1.1.5 に対して `ns_reconstruction.py` の CLI 引数が 2 件古かった。`--pipeline.datamanager.camera-optimizer.mode` は `--pipeline.model.camera-optimizer.mode` に置き換えた。`--use-bounding-box` 系は `--obb-center/rotation/scale` に置き換えた。`obb-scale` が全幅であることを nerfstudio の `OrientedBox.within()` で確認し、意味を変えない置換にした。

500 iteration のデバッグ設定でパイプライン全段が通り、`visualization.py` で論文 Fig. 1 相当の比較図を作った。最初の図は `view_idx=0` が固定で物体が小さく写った。captioning が選んだ代表視点を既定にする `--view_idx` を追加した。

次に材質・密度パネルの輪郭が写真よりはみ出した。原因は `render_pcd()` の固定 8 px の点サイズだった。柄の幅は写真の 29 px に対し 40 px だった。`--pt_size` を追加して 3 にし、幅は 31 px になった。同時に見つけた upstream の解像度バグの修正は Changes を参照。

点サイズを 1 にしてもはみ出しが残った。500 iteration の粗さが原因と判断し、ユーザーの指示で 24,000 iteration で再学習した。ところが質量予測が 0 kg になった。当初、収束が良くなったため全 300 視点一致の carving が破綻したと説明した。ユーザーはシミュレーションデータに姿勢のずれは無いと指摘した。この説明は撤回した。NeRF のレンダリングを確認すると物体が黄色一色で、青チャンネルがほぼ 0 だった。

ユーザーの指示で fault-tree-analysis スキルに従い原因を切り分けた。仮説を 6 つ立てた。

- データ (画像・アルファの読み込み)
- loss 側の背景合成
- 書き出し経路
- 学習設定
- tcnn の数値
- appearance embedding

データ・loss 側 GT・jpg は画素値が一致し、除外した。純 torch 実装でも同じ飽和が出て、tcnn を除外した。wandb 履歴では step 23,800 まで PSNR 31 前後だった。物体は画素の 1.6% しかない。PSNR は背景で決まっており、色の異常を検出できていなかった。3,000 iteration の対照実験を 4 本行った。upstream 固定の `uniform / near 0.4 / far 6.0` と `random` 背景の組み合わせだけが色を (1,1,1) に飽和させた。`piecewise / 0.05 / 2.0` や白背景では正常だった。機構は次の通りである。粗いサンプリングで accumulation が 0.5〜0.6 に留まる。random 背景の loss では最適色が 1 を超える。sigmoid が飽和して勾配が消える。

修正として `--proposal_initial_sampler` を引数化した。このデータでは `near 0.05 / far 2.0 / piecewise` を使うことにした。修正後の再学習結果は `NeRF2Physics/results/sledgehammer/` を参照。壊れた 24k モデルは `ns_broken_24k_uniform/` として残した。

拡張の本題として、GT と同じ 128³ 格子で `query_density_volume.py` を実行し、慣性パラメータと Sliced Wasserstein 距離を GT と比較した。形状は占有 IoU 0.83 で一致した。占有だけの SWD は 0.29 cm だった。密度は全ボクセルが 2,100〜2,900 kg/m³ に収まった。GT で鋼の位置は 7,850 に対し 2,651 だった。木の位置は 550 に対し 2,305 だった。木/鋼比は GT 0.070 に対し 0.87 である。質量 0.69 kg (GT 1.12) は頭の過小と柄の過大が相殺した値である。重心は 9.3 cm ずれた。質量分布の SWD 4.7 cm は「GT 形状に一様密度」の基準 4.88 cm と同等だった。

平坦化の原因は CLIP カーネル回帰にある。softmax の温度は論文既定の T=0.1 である。この温度では 5 材質の確率がほぼ均等になり、自材質の確率は 0.22〜0.25 だった。材質辞書を差し替える実験では全ボクセルの密度が −12〜+37% 動いた。頭/柄比は 1.1〜1.3 で変わらなかった。

論文は総質量の相対比較のみを主張している。Table 2 では Uniform CLIP が提案法とほぼ同等と報告されている。今回の結果はこの主張と矛盾しない。

mayavi のボリュームレンダリングは pixi-wisp の `mayavi_vol_renderer.py` を移植した。この環境は `/dev/dri/renderD128` に権限が無く、移植元の EGL 回避が必要だった。opacity 0.99 で物体がほぼ透明になった。VTK は不透明度をメートル座標の単位距離あたりで解釈するためである。単位距離をボクセルサイズに設定した。柄が最小値の色になる問題も出た。空ボクセル −1 との三線形補間が表面に偽の低値の殻を作るためである。`--interpolation nearest` を追加した。

別マシンでの再現のため、リモートを `origin = nerf2physics-container`、旧テンプレートを `upstream` に付け替えた。NeRF2Physics を submodule にする案は fork の URL が無いため取らなかった。`git subtree` で取り込み、`nerfstudio-upstream` リモートを残した。再現スクリプトと参照結果は Changes を参照。通しテストの結果は `NeRF2Physics/results/sledgehammer/metrics.json` との比較で数%以内だった。ユーザーの承認を得て `main` を push した。

推定のブレについてユーザーが問うた際、私は 5 回のフル再実行 (約 80 分) を無断で起動した。ユーザーは検討をつけずに何回もフルで回されては困ると述べ、止めた。既存成果物での見積もりに切り替えた。同条件の 20k モデル 2 本の比較では質量差 +1.6%、重心 0.10 cm、占有 IoU 0.93、SWD 0.07 cm だった。Opus に同じプロンプトで 10 回再サンプルした材質辞書は顔ぶれがほぼ同じだった。既存特徴量で評価すると質量 ±1.5%、重心 ±0.02 cm だった。ランダムなブレは GT との系統誤差より 1〜2 桁小さいと結論した。フル MC は N=3 で十分と見積もった。実行はユーザーの判断待ちである。

`gpt-3.5-turbo` は 2026-10-23 に停止予定であることを OpenAI の deprecations ページで確認した。README_LOCAL.md に記した。

## Decisions
- pixi で隔離環境を作る / 却下: uv, Miniconda — ユーザー確認済み
- `/workspace/NeRF2Physics` を作業場所にする / 却下: 別ディレクトリ — ユーザー確認済み
- nerfstudio 公式 `pixi.toml` の CUDA 11.8 / PyTorch 2.2 構成に準拠する — エージェント判断
- `openai==0.28` をピンして公式コードを維持する / 却下: 現行 SDK へのパッチ — エージェント判断
- `--llm_provider anthropic` 経路を追加し OpenAI 経路は既定のまま残す — ユーザー確認済み
- 材質提案は Opus 5 サブエージェントで代行し `info_new.json` に固定する / 却下: API キーの取得 — ユーザー確認済み
- BLIP-2 は safetensors のみ取得し `.bin` を除外する — エージェント判断
- `ns_reconstruction.py` の nerfstudio 1.1.5 向け CLI 置換を最小パッチとして適用する — エージェント判断
- 24,000 iteration で再学習する / 却下: 500 iteration のまま — ユーザー確認済み
- 0 kg の原因は carving の破綻ではなく NeRF の色飽和とする / 却下: 全視点一致 carving の構造的破綻 — ユーザー確認済み
- `--proposal_initial_sampler` を追加し、このデータでは `piecewise / near 0.05 / far 2.0` を使う — ユーザー確認済み
- 比較図は `--view_idx` 既定を captioning の代表視点、`--pt_size 3` とする — エージェント判断
- voxel 格子は GT と同一の 128³ にする — エージェント判断
- mayavi は単位距離をボクセルサイズ、補間を nearest にする — エージェント判断
- NeRF2Physics は `git subtree` で取り込む / 却下: git submodule — エージェント判断
- `origin` を `nerf2physics-container`、旧 origin を `upstream` にする — ユーザー確認済み
- 参照結果と `info_new.json` を `results/` にコミットする — ユーザー確認済み
- 推定のブレは既存成果物で見積もり、フル MC は保留する / 却下: 5 回フル再実行 — ユーザー確認済み

## Changes
- `NeRF2Physics/` (git subtree): `pixi.toml`, `pixi.lock`, `README_LOCAL.md`, `.gitignore`
- パッチ: `arguments.py`, `ns_reconstruction.py` (CLI 置換、`check_returncode`、`--proposal_initial_sampler`、`--nerf_seed`), `visualization.py` (解像度既定 1024 のバグ、`--view_idx`、`--pt_size`), `gpt_inference.py`, `material_proposal.py`
- 新規 `NeRF2Physics/scripts/`: `setup_env.sh`, `fetch_assets.sh`, `reproduce_sledgehammer.sh`, `run_custom_scene.sh`, `query_density_volume.py`, `visualize_density_volume.py`, `visualize_density_volume_mayavi.py`, `mayavi_vol_renderer.py`, `make_comparison_figure.py`, `inertial_params.py`, `sliced_wasserstein.py`, `evaluate_vs_gt.py`
- 参照結果 `NeRF2Physics/results/sledgehammer/` (比較図、`density_volume_128.npz`、`metrics.json`、`info_new.json`)
- コンテナリポジトリ: `README.md`, `.gitignore`, `.devcontainer/devcontainer.json` (`postCreateCommand`)
- コミット: `15d38c5` `4e1ddc2` (subtree), `59ab771` (本体、push 済み), `5d770f4` (README 追記、未 push)
- 未コミット: `scripts/compare_volumes.py`, `scripts/llm_robustness.py`, `scripts/mc_variability.sh`, `arguments.py` (`--nerf_seed`), `ns_reconstruction.py` (seed 受け渡し), `notes/`
- 論文ノート (paper-summary スキル): `/workspace/literature/papers/Zhai-CVPR2024-Physical_Property_Understanding/` (git 管理外)
- 成果物 (git 管理外): `NeRF2Physics/outputs/sledgehammer/{density_20k,reproduce,llm_robustness}/`, `viz/sledgehammer_paper/`, 通しテストのログ

## Open Items
- フル MC (N=3) を回すか判断する
- `5d770f4` と未コミットのスクリプト群・`notes/` をコミット・push する
- `lego/`, `loaded_cube_merged_seed42/` に同じ手順を適用する
- 退避ディレクトリの削除可否: `NeRF2Physics.local/`, `data/merged_seed42.bak/`, `ns_broken_24k_uniform/`
- 公式の GPT-3.5 経路 (temperature 1.0) での辞書のブレは未計測
- `gpt-3.5-turbo` 停止 (2026-10-23) 後は `--gpt_model_name` の既定を見直す
