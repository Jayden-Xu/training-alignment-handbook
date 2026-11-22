"""Verify full-model TIC receipts and maintain the Case 2 handoff."""
import csv
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

CASE = Path(__file__).resolve().parents[2]
SUITE = CASE / "tic" / "results" / os.environ["CASE2_C9_SUITE"]
ALIGNED_SUITE = SUITE.with_name(os.environ.get("CASE2_C9_ALIGNED_SUITE", SUITE.name))
STEPS = range(1, 11)
ARMS = ("aligned", "baseline")
FIELDS = ("step", "arm", "mismatch", "tokens", "k3_kl", "max_abs", "nonzero_adv")


def write(path, contents):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(contents)
    tmp.replace(path)


def read(path):
    return json.loads(path.read_text())


def arm_dir(arm):
    parent = ALIGNED_SUITE if arm == "aligned" else SUITE
    return parent.with_name(f"{parent.name}-{arm}")


def verify_arm(arm):
    run = arm_dir(arm)
    assert (run / f"DONE.{run.name}").read_text().strip() == "0"
    paths = [read(run / f"train-path-{rank}.json") for rank in range(16)]
    aligned = arm == "aligned"
    suite = ALIGNED_SUITE if aligned else SUITE
    master_port = (suite / f"{arm}-rendezvous-port").read_text().strip()
    for rank, path in enumerate(paths):
        assert path["arm"] == arm and path["rank"] == rank and path["layers"] == 24
        assert path["master_port"] == master_port
        assert path["gpt_oss_case2"] == path["batch_invariant"] == aligned
        if not aligned:
            assert path["vllm_batch_invariant"] is None
            assert read(run / f"runtime-{rank}.json")["vllm_bi"] is None
    updates, timing = ([json.loads(line) for line in (run / file).read_text().splitlines()]
                       for file in ("updates.jsonl", "timings.jsonl"))
    assert [r["step"] for r in updates] == [r["step"] for r in timing] == list(STEPS)
    rows = []
    for step, update, clock in zip(STEPS, updates, timing, strict=True):
        score = read(run / f"old-step-{step}.json")
        assert score["step"] == step and score["tokens"] > 0
        assert 0 <= score["mismatch"] <= score["tokens"] and math.isfinite(score["k3_kl"])
        assert math.isfinite(score["max_abs"])
        if aligned:
            assert score["mismatch"] == score["k3_kl"] == score["max_abs"] == 0
        assert 0 <= update["nonzero_adv"] <= update["responses"] <= 128
        assert clock["seconds"]["step"] > 0
        rows.append({**score, "arm": arm, "nonzero_adv": update["nonzero_adv"]})

    os.environ["CASE2_AUDIT_DIR"] = str(run)
    sys.path.insert(0, str(CASE / "audit"))
    from gate0_audit import check_sync

    checks = []
    for step in range(11):
        check_sync(step)
        receipt = read(run / f"sync-{step}-verified.json")
        assert receipt["payload_tensors_per_replica"] == 411
        assert receipt["post_load_parameters_per_replica"] == 315
        checks.append(receipt)
    write(run / "c9-receipts.json", json.dumps({
        "rows": rows, "timing_seconds": [r["seconds"]["step"] for r in timing],
        "training_path_rank0": paths[0], "syncs": checks,
    }, indent=2) + "\n")


def verify_aligned_archive():
    run = arm_dir("aligned")
    assert (run / f"DONE.{run.name}").read_text().strip() == "0"
    archive = read(run / "c9-receipts.json")
    assert [row["step"] for row in archive["rows"]] == list(STEPS)
    assert len(archive["timing_seconds"]) == len(STEPS)
    assert all(math.isfinite(t) and t > 0 for t in archive["timing_seconds"])
    for step, row in zip(STEPS, archive["rows"], strict=True):
        assert row["arm"] == "aligned" and 0 <= row["nonzero_adv"] <= 128
        assert row["tokens"] > 0 and row["mismatch"] == row["k3_kl"] == row["max_abs"] == 0
        score = read(run / f"old-step-{step}.json")
        assert all(row[key] == value for key, value in score.items())
        assert (run / f"old-step-{step}.pt").stat().st_size > 0
    for step, proof in zip(range(11), archive["syncs"], strict=True):
        assert proof == read(run / f"sync-{step}-verified.json")
        assert proof["step"] == step and proof["replicas"] == 16
        assert proof["payload_mismatches"] == proof["post_load_replica_mismatches"] == 0
        sender = read(run / f"sync-{step}-sender.json")
        loaded = read(run / f"sync-{step}-loaded-0.json")
        assert len(sender) == proof["payload_tensors_per_replica"] == 411
        assert len(loaded) == proof["post_load_parameters_per_replica"] == 315
        for replica in range(16):
            assert sender == read(run / f"sync-{step}-receiver-{replica}.json")
            assert loaded == read(run / f"sync-{step}-loaded-{replica}.json")
    print(f"{run.name}: 10 steps, 0/{sum(row['tokens'] for row in archive['rows']):,}, "
          "11 verified SHA rounds across 16 replicas", flush=True)


def progress(state, arm=None):
    path = CASE / "PROGRESS.md"
    text = path.read_text()
    prefix, _ = text.split("## 下一步\n", 1)
    start = prefix.index("## 当前位置\n") + len("## 当前位置\n")
    end = prefix.index("\n", start)
    work = f"\n## 运行中的任务\n无。\n"
    if state == "running":
        title = f"C9 · 进行中。正在运行 {arm}，每组 10 步。"
        if ALIGNED_SUITE != SUITE:
            title = "C9 · 进行中。只运行 baseline 的 10 步，复用已归档的 aligned。"
            next_step = f"baseline 完成并通过 SHA 核对后，与 `tic/results/{arm_dir('aligned').name}/c9-receipts.json` 汇总。"
        else:
            next_step = "aligned 通过逐步对齐及 SHA 核对后启动 baseline。两组完成后自动汇总。"
        work = (f"\n## 运行中的任务\n`{SUITE.name}`，预计超过 30 分钟。"
                f"状态 `tic/results/{SUITE.name}/STATUS`；完成标记 `DONE.{SUITE.name}`。\n")
    elif state == "complete":
        title = "C9 · 通过。aligned 和 baseline 各完成 10 步，汇总及 SHA 核对完成。"
        next_step = "查看 `results.csv`、`report_tic.md` 和各组 `c9-receipts.json`。"
        if ALIGNED_SUITE == SUITE:
            prefix = prefix.replace("\n## C5 分歧表", f"- C9 · 全模型 10+10 步。证据 `results.csv`、`tic/results/{SUITE.name}/STATUS`。\n\n## C5 分歧表", 1)
    else:
        run = arm_dir(arm)
        fault = run / "alignment-failure.json"
        if fault.exists():
            row = read(fault)
            title = (f"C9 · 卡住。aligned step {row['step']} 位差 {row['mismatch']}/{row['tokens']}，"
                     f"k3 KL {row['k3_kl']:.9g}，当步更新与 baseline 均未启动。")
            frozen = row["frozen_batch"]
            evidence = f"冻结输入 `tic/results/{run.name}/{frozen}`。" if (run / frozen).exists() else "冻结输入写入未完成。"
            next_step = f"{evidence} 查看 `tic/results/{run.name}/alignment-failure.json` 和同目录 old-step 收据。"
        else:
            title = f"C9 · 卡住。{arm} 运行或收据验收失败。"
            next_step = f"查看 `tic/results/{SUITE.name}/STATUS`、`tic/results/{run.name}/STATUS` 和失败摘要。"
    prefix = prefix[:start] + title + prefix[end:]
    new_text = prefix + f"## 下一步\n{next_step}\n" + work + "\n## 待人确认\n无。\n"
    assert len(new_text.splitlines()) <= 60
    write(path, new_text)
    with (CASE / "tic" / "events.jsonl").open("a") as events:
        events.write(json.dumps({"utc": datetime.now(timezone.utc).isoformat(), "gate": "C9",
                                 "event": state, "suite": SUITE.name, "arm": arm}) + "\n")
    if state != "running":
        with (CASE.parent / "STATE.md").open("a") as doc:
            doc.write(f"\n{title} {next_step} suite `case2/tic/results/{SUITE.name}`。\n")


def publish():
    checks = {arm: read(arm_dir(arm) / "c9-receipts.json") for arm in ARMS}
    for file in ("sync-0-sender.json", "sync-0-loaded-0.json"):
        assert read(arm_dir("aligned") / file) == read(arm_dir("baseline") / file), file

    csv_path = CASE / "results.csv"
    temporary = csv_path.with_suffix(".tmp")
    with temporary.open("w", newline="") as output:
        writer = csv.DictWriter(output, FIELDS)
        writer.writeheader()
        for arm in ARMS:
            writer.writerows(checks[arm]["rows"])
    temporary.replace(csv_path)

    report = CASE / "report_tic.md"
    original = report.read_text()
    setup = original[original.index("## Setup"):original.index("## 进度")].rstrip()
    differences = original[original.index("## C5 分歧与修补"):original.index("原始收据为")].rstrip()
    previous = original[original.index("## C6 ·"):original.index("## C9 ·")].rstrip()
    baseline = checks["baseline"]["training_path_rank0"]
    results = [
        "## C9 · 全模型结果", "",
        "两组各完成 10 步。初始 sender 和 vLLM 加载后的 SHA 跨组相同。"
        "每组 round 0–10 的发送端与 16 个 IPC 接收副本逐项一致，16 个 vLLM 副本的加载后 SHA 逐项一致。"
        "每副本覆盖 411 个 payload 张量和 315 个物理参数。", "",
        f"Baseline 的 16 个训练 rank 均记录 `gpt_oss_case2=false`、`batch_invariant=false`，`VLLM_BATCH_INVARIANT` 未设。"
        f"实际 backbone 为 `{baseline['backbone']}`，decoder 为 `{baseline['decoder'][0]}`，"
        f"attention 为 `{baseline['attention']}`（`{baseline['attention_backend']}`），norm 为 `{baseline['norm']}`。", "",
        "`nonzero_adv` 按 response 计数，任一有效 token 的 GRPO advantage 非零即计 1。"
        "两组均使用 GSM8K 正确性 reward。耗时包括采样、训练及权重同步。", "",
        "| 组别 | step | 位差 / token | k3 KL | max abs | nonzero_adv | 耗时（s） |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for arm in ARMS:
        for row, seconds in zip(checks[arm]["rows"], checks[arm]["timing_seconds"], strict=True):
            results.append(f"| {arm} | {row['step']} | {row['mismatch']}/{row['tokens']} | "
                           f"{row['k3_kl']:.9g} | {row['max_abs']:.9g} | {row['nonzero_adv']} | {seconds:.3f} |")
    results += ["", "逐步数据见 `results.csv`。两组的配置、rank 路径、评分与同步收据分别位于 "
                + "、".join(f"`tic/results/{arm_dir(arm).name}/`" for arm in ARMS) + "。"]
    upstream = """## 可上游化的改动

| 改了什么 | 对应仓库 | 是否通用 |
| --- | --- | --- |
| FSDP2 的训练侧对齐前向与 old-logprob 条件接入 | verl | 接入模式通用，本次适配 GPT-OSS BF16/TP1 |
| 对齐 TRITON_ATTN 的 GQA、滑窗和可训练 sink，及 TritonExperts 的带 bias、SWIGLUOAI 和 top-k VJP | verl | GPT-OSS 配方专用，反向接口可复用 |
| YaRN 和 add+RMSNorm 复用 vLLM 前向，提供 FP32 VJP | verl | kernel 数学相同的模型可复用 |
| V2 评分复用完整词表宽度、屏蔽 tokenizer 外列并提供 FP32 VJP | verl | padded vocab 的 V2 评分可复用 |
| CUDA Graph FULL 与 eager kernel 选择配方 | verl（vLLM 配置） | 需按模型和硬件验证 |
| 有效 token 位比较、失败冻结、逐轮 IPC SHA 和 native 路径审计 | verl | 审计方法通用，当前为 Case 2 脚本 |
"""
    write(report, "\n\n".join(("# 案例 2 · 训练与推理一致性", setup, "\n".join(results),
                               differences, previous, upstream)))
    progress("complete")


if __name__ == "__main__":
    command, *args = sys.argv[1:]
    {"verify": verify_arm, "archive": verify_aligned_archive, "progress": progress, "publish": publish}[command](*args)
