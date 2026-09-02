# Devcontainer Base Template

NVIDIA CUDA + Ubuntu (既定 24.04) ベースの VS Code devcontainer テンプレート。Python の環境管理は各リポジトリに任せ、特定のパッケージ管理ツールを同梱しない。
GPU 開発環境を新規プロジェクトごとに素早く立ち上げるためのベース設定。

## ディレクトリ構成

```
<repo-root>/
├── .devcontainer/
│   ├── devcontainer.json       # VS Code 拡張・Python・シェル設定
│   ├── Dockerfile              # nvidia/cuda + Ubuntu (既定 24.04), 非 root ユーザー
│   ├── docker-compose.yaml     # GPU・ボリューム・ipc: host
│   ├── entrypoint.sh           # zsh 初期化 + gosu による非 root 切替
│   ├── init-env.sh             # リポジトリ名とホスト情報から .env を生成
│   ├── .dockerignore           # ビルドコンテキスト除外
│   └── .env.example            # マシン固有の設定テンプレート
├── .gitignore                  # .devcontainer/.env 等を除外
└── README.md
```

## 使い方

### 1. リポジトリをクローン

```bash
git clone <this-repo-url> my-new-project
cd my-new-project
```

### 2. プロジェクト名を確認する

`init-env.sh` はリポジトリのディレクトリ名を小文字の Compose 名へ整形し、`.devcontainer/.env` の `COMPOSE_PROJECT_NAME` に設定する。別名を使う場合は、生成後の `.devcontainer/.env` を編集する。

ただし [.devcontainer/devcontainer.json](.devcontainer/devcontainer.json) の `"name"` フィールド（VS Code 左下に "Dev Container: ..." と表示されるラベル）は devcontainer 仕様上 `.env` 補間に対応しないため SSoT から自動派生しない。プロジェクト識別を視覚的に揃えたい場合は、ここも併せて手動更新する：

```jsonc
"name": "Your Project Name",  // 表示ラベル。Docker 識別子 (image/container 名) には影響しない
```

更新後は `Cmd/Ctrl+Shift+P → Developer: Reload Window` で反映（リビルド不要）。

### 3. 環境変数を設定

[.devcontainer/.env](.devcontainer/.env) は `.devcontainer/init-env.sh` が以下を素材に自動生成する：

- リポジトリのディレクトリ名 → `COMPOSE_PROJECT_NAME`
- ホストの UID/GID → `HOST_UID` / `HOST_GID`
- 環境変数または既定値 → `CUDA_VERSION` / `UBUNTU_VERSION` / `WORKSPACE_DIR` / `DEFAULT_USER` / `LOCALE` / `DISPLAY_NUM` / `NVIDIA_VISIBLE_DEVICES`

実行タイミング:
- **VS Code (devcontainer)**: 自動。`devcontainer.json` の `initializeCommand` がスクリプトを呼ぶ
- **CLI (standalone)**: 起動前に 1 度だけ手動実行
  ```bash
  bash .devcontainer/init-env.sh
  ```

既定値を上書きしたい場合の選択肢:
- 生成された `.devcontainer/.env` を直接編集する（最も手軽）
- `.devcontainer/.env` を削除し、override をシェルで export してから再実行：
  ```bash
  rm .devcontainer/.env
  CUDA_VERSION=12.6.0 UBUNTU_VERSION=22.04 LOCALE=ja_JP.UTF-8 bash .devcontainer/init-env.sh
  ```

各変数の意味は [.devcontainer/.env.example](.devcontainer/.env.example) を参照。`.devcontainer/.env` は git 管理外。

### 4. コンテナを起動

**VS Code (devcontainer)**:

コマンドパレット → `Dev Containers: Reopen in Container`

**CLI (standalone)**:

```bash
bash .devcontainer/init-env.sh
docker compose -f .devcontainer/docker-compose.yaml build
docker compose -f .devcontainer/docker-compose.yaml up -d
docker compose -f .devcontainer/docker-compose.yaml exec dev zsh
```

## 含まれる設定

### ベースイメージ

`nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION}` — CUDA・Ubuntu のバージョンは `.env` の `CUDA_VERSION` / `UBUNTU_VERSION` で切替可能。

### GPU サポート

デフォルトで NVIDIA GPU 全台を割当。`ipc: host` により PyTorch DataLoader / NCCL の共有メモリも有効。

### 非 root ユーザー

ホストの UID/GID をビルド時に注入し、コンテナ内でもホストと同じ権限で動作。`gosu` でランタイム切替。

### ボリュームマウント

| ホスト | コンテナ | 用途 |
|-------|---------|------|
| プロジェクトルート | `${WORKSPACE_DIR}` (既定 `/workspace`) | ワークスペース |
| `~/.ssh` | `~/.ssh` (ro) | SSH 鍵 |
| `~/.gitconfig` | `~/.gitconfig` (ro) | Git 設定 |
| `/tmp/.X11-unix` | `/tmp/.X11-unix` | GUI 転送 |

### VS Code 拡張

Claude Code, Python, Pylance, Ruff, Jupyter, Docker, GitLens, Git Graph, Debugpy, YAML, TOML, Markdown, Error Lens, Todo Tree, Spell Checker, Path Intellisense

### entrypoint.sh の動作

1. 初回起動時に zsh の設定ファイルを生成
2. ユーザー用ディレクトリを作成し、非 root ユーザーの所有に補正
3. `gosu` で非 root ユーザーに切り替えてコマンドを実行

Python 環境の作成と依存の同期は行わない。

### Doppler によるシークレット管理 (オプション)

API キー (`WANDB_API_KEY`, `HF_TOKEN`, `OPENAI_API_KEY` 等) を [Doppler](https://www.doppler.com/) 経由でコンテナに注入できる。Doppler CLI はイメージに同梱され、シェル起動時に `/etc/zsh/zshenv` が `doppler secrets download` を実行して全 secret を環境変数として展開する。`DOPPLER_TOKEN` 未設定ならその処理はスキップ (no-op) するので、Doppler を使わない人にはテンプレートが透過的。

#### Doppler 側の作業 (workplace ごとに 1 回)

ブラウザのみで完結:

1. [Doppler dashboard](https://dashboard.doppler.com/) で Project を作成 (例: `research-keys`)
2. デフォルトの 3 environment (`dev` / `stg` / `prd`) のうち `dev` を使う
3. `dev` config を開いて **Add Secret** で必要なキーを登録 (例)

   ```text
   WANDB_API_KEY = <your-wandb-key>
   HF_TOKEN = <your-hf-token>
   ANTHROPIC_API_KEY = <your-anthropic-key>
   ```

4. Project → 該当 config (`dev`) → **Access** タブ → **Service Tokens** → **Generate**
   - Access: **Read** (コンテナからは読み取りのみ)
   - 表示された `dp.st.dev.xxxxxxxxxxxxxxx` をコピー (この画面でしか見られない)

#### マシンごとの作業 (各ホストで 1 回)

ホストにファイル 1 個作るだけ。Doppler CLI のインストール不要:

```bash
mkdir -p ~/.config
echo 'DOPPLER_TOKEN=dp.st.dev.ここにペースト' > ~/.config/doppler.env
chmod 600 ~/.config/doppler.env
```

このファイルは [.devcontainer/docker-compose.yaml](.devcontainer/docker-compose.yaml) の `env_file:` で読まれ、ホストシェルには load されない。dotfiles repo で `~/.config/<file>` を個別 symlink している運用なら、新規作成する `doppler.env` は symlink されない = tracked にならない。念のため dotfiles repo の `.gitignore` に `doppler.env` を追加しておくと事故防止になる。

#### 起動と確認

```bash
# rebuild が必要 (Doppler CLI を image に同梱するため)
docker compose -f .devcontainer/docker-compose.yaml down
docker compose -f .devcontainer/docker-compose.yaml build
docker compose -f .devcontainer/docker-compose.yaml up -d

# 確認 (コンテナ内シェルで)
docker compose -f .devcontainer/docker-compose.yaml exec dev sh -c 'echo $WANDB_API_KEY'
```

#### 運用ポイント

- **auto-discovery**: dashboard で secret を追加するだけで、次回シェル起動時に自動的にコンテナの env に流入する。`docker-compose.yaml` 編集や container rebuild は不要 (新しいシェルを開けば反映)
- **常に最新**: シェル起動ごとに fetch するので、dashboard で値を変更しても次のシェルで反映 (約 500ms の起動時オーバーヘッド)
- **Read-only**: service token は read-only スコープなのでコンテナ側から secret を書き換え不可
- **Token rotation**: 漏洩疑いがあれば dashboard で revoke → 新規生成 → 各マシンの `~/.config/doppler.env` を更新
- **プロジェクトごとに別 config を使いたい場合**: `.devcontainer/docker-compose.override.yaml` (gitignored) で `env_file:` を上書きする
- **wandb など `.netrc` ベースのツール**: env var (`WANDB_API_KEY` 等) が優先されるので、Doppler 経由で渡せば `.netrc` マウントは不要にできる

## カスタマイズ例

### GPU 不要の場合

`docker-compose.yaml` の `deploy` セクションと GPU 関連の `environment` を削除し、ベースイメージを `ubuntu:24.04` に変更する。

### 追加サービスが必要な場合

`docker-compose.yaml` に `services` を追加する (例: DB, Redis, Ollama 等)。
