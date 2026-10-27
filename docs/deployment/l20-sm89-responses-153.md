# DeepSeek V4.1 Responses API 修复与 L20 × 8 验证

资料截止：2026-10-27。此分支用于内网验证，未在 L20 上完成模型服务或性能验收。

## 代码来源和本次修复

- SM89 基线：`yhfgyyf/vllm-deepseek-v4-sm89` 的
  `94dbeca9aecdd5da209d02534bc1b1c1144723b7`。
- 问题：[SM89 issue #153](https://github.com/yhfgyyf/vllm-deepseek-v4-sm89/issues/153)。
  Codex 使用 Responses API 时，V4.1 编码器拒绝 `input_text`。
- 移植来源：[vLLM PR #56299](https://github.com/vllm-project/vllm/pull/56299)，
  已合并提交 `eb42686a30cddf325ffeff7b3bd5e3a7298c2c00`。
- 修复：`vllm/tokenizers/deepseek_v41.py` 将 `text`、`input_text`、
  `output_text` 按同一文本规则编码。保留段落间的双换行、图片占位和不支持媒体的错误。
- 测试：在既有测试文件增加用户输入、助手历史两种回归，并检查不修改传入消息。

这是 Python 编码器兼容修复；本次没有修改 CUDA 内核、模型权重、认证或权限。
没有重建 wheel、Docker 镜像，也没有把 `vllm-project/vllm/main` 整体合入。
现有 Release wheel 不会因为推送源码而自动包含此补丁。

## 在已有 V4.1 环境应用

新建环境可按根 README 的源码安装流程，检出 `fix/dsv41-responses-sm89`，
保留该基线要求的配套 FlashInfer、PyTorch、CUDA 和 Triton 版本。
不要用普通上游 vLLM wheel 覆盖 SM89 适配。

对于已经能运行此 fork 的 V4.1 服务，可只应用这个 Python 修复，避免重编译原生扩展。
先停止服务，激活**服务实际使用的虚拟环境**，然后执行以下命令。
脚本检查目标代码，保留备份；遇到其他版本布局时退出，不猜测替换。

```bash
.venv/bin/python - <<'PY'
from importlib.util import find_spec
from pathlib import Path
import shutil

spec = find_spec("vllm")
if spec is None or spec.origin is None:
    raise SystemExit("当前解释器没有安装 vLLM")
path = Path(spec.origin).parent / "tokenizers" / "deepseek_v41.py"
text = path.read_text(encoding="utf-8")
old = 'if part_type == "text":'
new = 'if part_type in ("text", "input_text", "output_text"):'
if new in text:
    print("已经包含修复:", path)
elif text.count(old) == 1:
    backup = path.with_name(path.name + ".before-153")
    if backup.exists():
        raise SystemExit("备份已经存在，请先核对: " + str(backup))
    patched = text.replace(old, new, 1)
    compile(patched, str(path), "exec")
    shutil.copy2(path, backup)
    path.write_text(patched, encoding="utf-8")
    print("已修复:", path)
    print("回滚备份:", backup)
else:
    raise SystemExit("编码器版本与补丁不匹配，请人工核对: " + str(path))
PY
```

重新启动全部 vLLM API/worker 进程，沿用原来可工作的模型和启动参数。
回滚时先停止服务，将输出的 `.before-153` 文件复制回原文件，再重启。
容器内临时改动会在容器重建后丢失；持久部署需把补丁纳入自己的镜像构建。

## 内网验收

1. 用实际服务环境运行官方测试：

   ```bash
   VLLM_TARGET_DEVICE=cpu .venv/bin/python -m pytest \
     tests/tokenizers_/test_deepseek_v41.py -q
   ```

2. 验证 Responses 请求，按实际服务修改地址、模型名和 API key：

   ```bash
   export BASE_URL=http://127.0.0.1:8000
   export SERVED_MODEL=deepseek-v4.1-flash
   curl --fail-with-body "$BASE_URL/v1/responses" \
     -H "Authorization: Bearer $OPENAI_API_KEY" \
     -H 'Content-Type: application/json' \
     -d "{\"model\":\"$SERVED_MODEL\",\"input\":[{\"role\":\"user\",\"content\":[{\"type\":\"input_text\",\"text\":\"Reply with OK.\"}]}],\"reasoning\":{\"effort\":\"none\"},\"max_output_tokens\":128}"
   ```

3. 在实际 Codex 客户端执行首轮、后续多轮、流式输出、工具调用及工具结果续写。
   后续请求需覆盖助手历史中的 `output_text`；仅首轮成功不足以验收。
   如果使用图片，还需验证图片与文字混排；音频等不支持类型应继续被拒绝。

本机为 Windows CPU 环境，没有安装可运行该 fork 的 CUDA/PyTorch 服务栈。
本次用隔离 harness 编译生产 normalizer、tokenizer wrapper 和 reference encoder，
复用既有测试和 FakeHfTokenizer，绕开包级 CUDA 依赖：

- 修复前：`2 failed, 29 passed, 1 skipped`，失败分别为 `input_text` 和 `output_text`。
- 修复后：`31 passed, 1 skipped`，包括参考文本、图片顺序和不支持媒体用例。
- 跳过项为依赖完整 `ChatCompletionRequest` 的 API 参数集成测试。
- Ruff 检查和格式检查通过。完整官方 pytest、HTTP 服务、真实 Codex 和 L20 验收未执行。

隔离 harness 只证明编码器行为，不证明服务已经可用或推理已经提速。

## 最近的上游提速是否适用于 L20 SM89

[vLLM 2026-10-07 官方文章](https://vllm-project.github.io/2026/10/07/deepseek-v41-flash.html)
汇总了 bounded replay、内核融合和 Engram 优化。文中的性能数据采用
GB200/GB300 等测试条件，不能据此承诺 L20 × 8 获得同样倍率。

| 优化 | 对当前 L20/SM89 分支的判断 | 本次处理 |
| --- | --- | --- |
| Responses 文本类型，PR #56299 | 与 GPU 架构无关 | 已移植 |
| fork 的 CED decoder replay | 已有 SM89 实现，实验性近似 prefill；不等于上游完整 SWA bounded replay | 随 SM89 最新基线保留，默认设置未改 |
| fork 的 DSpark / adaptive verification | 已有 SM89 路径，要求相配的 checkpoint/confidence head | 随基线保留，需 L20 A/B |
| 上游 encoder/decoder SWA replay 与 CUDA graphs，PR #56227 / #58132 / #59532 | 调度器、KV 管理和 graph 接口有较多联动；不能直接把上游参数加到旧基线 | 本次未合入 |
| Mega-mHC / Mega-Gate / mHC overlap，PR #56962 / #56266 / #57603 | 所核对的上游启用路径依赖 SM100/DeepGEMM 条件 | 本次未合入 SM89 路径 |
| Sparse MQA logits，PR #56254 | 所核对的上游新内核路径要求 SM100 系列 | 本次未合入 |
| MegaAttention / NVFP4 KV，PR #56935；WO-A 融合，PR #58634 | 此处新增的后端/融合路径针对 SM100/SM103 | 本次未合入 |
| Engram 异步预取，PR #56512；分片，PR #57651；THP，PR #56926 | 有定向移植价值；需要处理模型生命周期、stream/event、host memory、UVA 和分布式依赖 | 本次未合入，优先候选 |
| Engram 排序/inline lookup，PR #59327 | 上游阈值只在 Blackwell 启用，其他架构保留旧行为 | 不强制向 SM89 开启 |

这里的 SM100 限制针对所列上游实现；应继续保留本 fork 自己的 SM89 attention、
量化和混合缓存实现。不能仅删除架构判断来宣称完成适配。
Engram #56512 本身已涉及 12 个文件、约 1800 行新增代码，且后续提交使用了新的模型目录；
它不是一个可独立 cherry-pick 后就认定与当前 CED/DSpark 兼容的小补丁。

对应源码和讨论：
[SWA encoder](https://github.com/vllm-project/vllm/pull/56227)、
[SWA decoder](https://github.com/vllm-project/vllm/pull/58132)、
[replay graphs](https://github.com/vllm-project/vllm/pull/59532)、
[Mega-mHC](https://github.com/vllm-project/vllm/pull/56962)、
[Mega-Gate](https://github.com/vllm-project/vllm/pull/56266)、
[mHC overlap](https://github.com/vllm-project/vllm/pull/57603)、
[sparse MQA](https://github.com/vllm-project/vllm/pull/56254)、
[MegaAttention](https://github.com/vllm-project/vllm/pull/56935)、
[WO-A](https://github.com/vllm-project/vllm/pull/58634)、
[Engram prefetch](https://github.com/vllm-project/vllm/pull/56512)、
[Engram sharding](https://github.com/vllm-project/vllm/pull/57651)、
[THP](https://github.com/vllm-project/vllm/pull/56926)、
[lookup thresholds](https://github.com/vllm-project/vllm/pull/59327)。

## L20 × 8 的性能验证顺序

先沿用现有可运行配置，只验证此次兼容修复。记录 `nvidia-smi`、
`nvidia-smi topo -m`、实际 GPU compute capability、空闲显存、主机内存、NUMA
和完整软件版本。八张卡的显存不能当作单卡显存；Engram offload 也需要额外主机内存。

之后固定权重、TP8、精度、batch、context 和并发：

1. 仅对比 `--hf-overrides '{"ced_prefill":false}'` 和 `true`；
   两组都保持相同 DSpark 设置。当前 fork 使用 `ced_prefill`，不要套用上游新 flag。
2. 单独对比 DSpark 关闭/开启；使用匹配的 draft 权重和已支持的 adaptive 配置。
3. 使用不同随机 prompt，完成 JIT/graph 预热后比较 TTFT、TPOT、吞吐、显存和主机内存。
   从并发 1、输入 8K/32K 开始，再逐步增加长度、batch 和并发。
4. CED 为近似算法，DSpark 也有采样条件；每组都运行实际任务质量回归，
   不能只比较 tokens/s。优先检查代码修改、工具调用和长对话任务。

一个固定工作负载示例：

```bash
vllm bench serve --backend openai-chat \
  --base-url http://127.0.0.1:8000 --model "$SERVED_MODEL" \
  --dataset-name random --random-input-len 8192 --random-output-len 128 \
  --num-prompts 16 --max-concurrency 1 --save-result \
  --result-dir results/l20-ced-off
```

开启 CED 并完成预热后，用同样参数输出到 `results/l20-ced-on`。
实际可运行的长度和并发以每卡可用显存、模型权重及缓存预算为准。

本次使用 AI 协助移植、测试与分析；推送的是验证分支，没有向上游提交 PR。
