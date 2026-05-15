# 环境搭建记录

> 从零搭建 AlignSQL 项目的完整操作日志。

---

## 一、项目设计

**时间：** 项目启动阶段  
**目标：** 基于 Qwen3-8B + LLaMA-Factory 跑通 NL2SQL 的 SFT → DPO 全流程  
**定位：** LLM 微调工程实践，聚焦数据处理、训练配置、偏好对齐的完整链路  
**数据：** Spider 1.0（Yale 官方 NL2SQL 基准）  
**硬件：** RTX 4090 (24GB) × 1  
**协议：** MIT

---

## 二、GitHub 仓库

https://github.com/gzhzk/alignsql

---

## 三、本地数据准备

**环境：** WSL2 Ubuntu, uv venv, Python 3.12

```bash
mkdir hf_dataset && cd hf_dataset
uv venv --python 3.12
uv pip install datasets

# 下载 Spider（国内需 HF 镜像）
export HF_ENDPOINT=https://hf-mirror.com
uv run python scripts/download_spider.py
```

下载结果：

| 数据集 | 条数 |
|--------|------|
| train  | 7,000 |
| validation | 1,034 |

数据路径：`hf_dataset/data/spider/`

---

## 四、AutoDL 租用 GPU 实例

**配置选择：**

| 项 | 选择 |
|------|------|
| GPU | RTX 4090 (24GB) |
| 镜像 | PyTorch 2.5.1 + CUDA 12.4 |
| Python | 3.12.3（镜像自带） |
| 计费 | 按量计费 |

**连接方式：**

```bash
# 本地 PowerShell 开 SSH 隧道（WebUI 用）
ssh -L 7860:localhost:7860 -p <端口号> root@<AutoDL实例地址>

# 浏览器打开
http://localhost:7860
```

---

## 五、AutoDL 环境搭建

### 5.1 安装 tmux

```bash
apt-get update && apt-get install tmux -y
tmux new -s alignsql
```

### 5.2 克隆项目与框架

```bash
# 开启学术加速
source /etc/network_turbo

# 克隆 AlignSQL
git clone https://github.com/gzhzk/alignsql.git

# 克隆 LLaMA-Factory（源码安装方式）
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory

# 安装（注意：学术加速与阿里云 pip 镜像冲突，需关闭加速）
pip install -e ".[torch,metrics]"

# 验证
llamafactory-cli version
```

### 5.3 修复 WebUI 启动问题

首次 `llamafactory-cli webui` 报 `libcudart.so.13` 错误，原因是 torchaudio 版本与 CUDA 12.4 不匹配：

```bash
pip install torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
```

重新启动 WebUI 即可。

### 5.4 日常工作流

每次重连：

```bash
# AutoDL SSH
tmux new -s alignsql  # 或 tmux attach -t alignsql

# 本地 PowerShell（另开一个窗口）
ssh -L 7860:localhost:7860 -p <端口号> root@<AutoDL实例地址>

# 浏览器访问
http://localhost:7860
```

---

## 六、下载模型

**方案尝试记录：**

| 方案 | 结果 | 原因 |
|------|------|------|
| `huggingface-cli download` | ❌ 命令已废弃 | 被 `hf` 替代 |
| `hf download` | ❌ 503/401 | 学术加速代理不稳定，Xet 认证失败 |
| `HF_ENDPOINT=hf-mirror.com hf download` | ❌ 同样网络错误 | 代理环境未完全清理 |
| **`modelscope download`（最终方案）** | ✅ 成功 | 阿里源，AutoDL 内网满速 |

最终采用 ModelScope：

```bash
pip install modelscope
modelscope download Qwen/Qwen3-8B --local_dir /root/autodl-tmp/models/qwen3-8b
```

模型约 16GB，AutoDL 内网下载预计 10-15 分钟。

> **实际耗时：** 约 24 分钟（50GB 文件，含多语言冗余文件）

---

## 七、Spider 数据集下载

ModelScope 只提供了模型，数据集仍需从 HuggingFace 获取。

在 AutoDL 上创建脚本下载：

```bash
cat > /root/autodl-tmp/download_spider.py << 'PYEOF'
from datasets import load_dataset
ds = load_dataset("xlangai/spider")
ds.save_to_disk("/root/autodl-tmp/data/spider")
print("OK")
PYEOF

python /root/autodl-tmp/download_spider.py
```

下载结果：

| 文件 | 大小 |
|------|------|
| train | 7000 条 |
| validation | 1034 条 |

下载完成后删除临时脚本：

```bash
rm /root/autodl-tmp/download_spider.py
```

最终 `/root/autodl-tmp/` 结构：

```
/root/autodl-tmp/
├── models/qwen3-8b/     ← 模型 (16GB)
└── data/spider/         ← 数据集
```

---

## 八、W&B 实验追踪配置

```bash
pip install wandb
wandb login
```

执行 `wandb login` 后终端显示授权 URL，在本地浏览器打开该链接，复制 API key 并粘贴回终端即可。

登录成功标志：

```
Currently logged in as: <用户名> to https://api.wandb.ai
```

在 SFT/DPO 的 yaml 配置中添加 wandb 参数：

```yaml
report_to: wandb
run_name: sft_lora32_lr2e4   # 可选，方便区分实验
```

---

## 九、目前状态

AutoDL 环境已就绪：

| 项目 | 路径 | 状态 |
|------|------|:----:|
| 基座模型 | `/root/autodl-tmp/models/qwen3-8b/` | ✅ |
| 训练数据集 | `/root/autodl-tmp/data/spider/` | ✅ |
| LLaMA-Factory | `/root/LLaMA-Factory/` | ✅ |
| W&B 登录 | `wandb login` 已完成 | ✅ |
| SSH 隧道 | `ssh -L 7860:localhost:7860 -p <端口> root@<地址>` | ✅ |
