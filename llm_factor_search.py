"""
LLM驱动的因子挖掘系统
流程：LLM生成因子 → 本地回测 → 结果反馈给LLM → 迭代优化
"""

import json
import os
import warnings
from datetime import datetime

import anthropic
import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────
# 配置
# ─────────────────────────────────────────────────────────

TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META",
    "TSLA", "NVDA", "JPM", "GS", "BAC",
    "JNJ", "PFE", "XOM", "WMT", "HD",
]

START = "2024-01-01"
END   = "2026-04-01"

N_ROUNDS      = 3   # 迭代轮数
N_FACTORS     = 3   # 每轮生成几个因子
IC_THRESHOLD  = 0.01  # IC低于这个视为无效因子

# Anthropic API key：建议放在环境变量里，不要硬编码
API_KEY = os.environ.get("ANTHROPIC_API_KEY", "你的API_KEY")


# ─────────────────────────────────────────────────────────
# 1. 下载数据
# ─────────────────────────────────────────────────────────

def load_data(tickers, start, end):
    print(f"正在下载数据 {start} ~ {end} ...")
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True)
    close  = raw["Close"].dropna(how="all")
    volume = raw["Volume"].dropna(how="all")
    print(f"完成：{close.shape[0]} 个交易日，{close.shape[1]} 只股票\n")
    return close, volume


# ─────────────────────────────────────────────────────────
# 2. 因子回测
# ─────────────────────────────────────────────────────────

# eval 可用的安全函数白名单
SAFE_FUNCS = {
    "pd": pd,
    "np": np,
    "abs": abs,
    "round": round,
}

def evaluate_factor(expr: str, close: pd.DataFrame, volume: pd.DataFrame) -> dict:
    """
    执行因子表达式，返回回测指标。
    expr 示例："-close.diff(5)"
    """
    try:
        local_vars = {"close": close, "volume": volume, **SAFE_FUNCS}
        factor: pd.DataFrame = eval(expr, {"__builtins__": {}}, local_vars)

        if not isinstance(factor, pd.DataFrame):
            return {"error": "因子表达式必须返回 DataFrame"}

        returns_next = close.pct_change(1).shift(-1)
        daily_ic = factor.corrwith(returns_next, axis=1).dropna()

        if len(daily_ic) < 20:
            return {"error": "有效数据太少"}

        mean_ic  = daily_ic.mean()
        std_ic   = daily_ic.std()
        icir     = mean_ic / std_ic if std_ic > 1e-8 else 0.0
        ic_pos   = (daily_ic > 0).mean()

        # 逐年IC
        yearly_ic = (
            daily_ic.groupby(daily_ic.index.year)
            .mean()
            .round(4)
            .to_dict()
        )

        return {
            "mean_ic":  round(float(mean_ic), 4),
            "icir":     round(float(icir), 4),
            "ic_pos":   round(float(ic_pos), 3),
            "yearly_ic": yearly_ic,
        }

    except Exception as e:
        return {"error": str(e)}


# ─────────────────────────────────────────────────────────
# 3. LLM 交互
# ─────────────────────────────────────────────────────────

def build_prompt(hypothesis: str, previous_results: list, round_num: int) -> str:
    """构建发给 LLM 的 prompt"""

    # 把上一轮的有效结果整理成简洁的文字
    prev_summary = ""
    if previous_results:
        lines = []
        for r in previous_results:
            if "error" not in r:
                lines.append(
                    f"  - {r['name']} | IC={r['mean_ic']} ICIR={r['icir']} "
                    f"IC>0={r['ic_pos']:.0%} | 逻辑:{r['logic']}"
                )
            else:
                lines.append(f"  - {r['name']} | 报错: {r['error']}")
        prev_summary = "上一轮结果：\n" + "\n".join(lines)

        # 找出最好的因子，告诉 LLM 往哪个方向改进
        valid = [r for r in previous_results if "error" not in r]
        if valid:
            best = max(valid, key=lambda x: abs(x["icir"]))
            prev_summary += (
                f"\n\n最优因子是 [{best['name']}]，ICIR={best['icir']}。"
                f"请在此基础上改进，同时尝试其他方向。"
            )

    guidance = prev_summary if prev_summary else "这是第一轮，请自由探索。"

    return f"""你是一个量化因子研究员，正在为美股做因子挖掘。

市场假设：{hypothesis}

{guidance}

请生成 {N_FACTORS} 个新的因子。规则：
1. 变量只能用 close（收盘价 DataFrame）和 volume（成交量 DataFrame）
2. 可用的方法：.diff(n) .pct_change(n) .rolling(n).mean() .rolling(n).std() .rank(axis=1, pct=True)
3. 可用函数：np.log() np.abs() np.sign() pd.Series 运算
4. 表达式必须返回和 close 同形状的 DataFrame
5. 因子值越高 → 预测该股票未来1天收益越高
6. 不同因子之间逻辑要有区别，不要只改参数

第 {round_num} 轮，请在上一轮基础上改进或探索新方向。

只返回 JSON，不要任何其他文字：
{{
  "factors": [
    {{
      "name": "简短英文名",
      "expr": "python表达式字符串",
      "logic": "一句话解释市场逻辑"
    }}
  ]
}}"""


def llm_generate_factors(
    client: anthropic.Anthropic,
    hypothesis: str,
    previous_results: list,
    round_num: int,
) -> list:
    """调用 LLM 生成因子，返回因子列表"""

    prompt = build_prompt(hypothesis, previous_results, round_num)

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = response.content[0].text.strip()

    # 清理 markdown 代码块
    raw = raw.replace("```json", "").replace("```", "").strip()

    try:
        parsed = json.loads(raw)
        return parsed.get("factors", [])
    except json.JSONDecodeError:
        print(f"  [警告] LLM 返回内容无法解析为 JSON:\n{raw[:300]}")
        return []


# ─────────────────────────────────────────────────────────
# 4. 主循环
# ─────────────────────────────────────────────────────────

def run_factor_search(hypothesis: str, n_rounds: int = N_ROUNDS):
    """
    完整的 LLM 因子搜索循环。
    返回所有轮次的结果列表，按 ICIR 排序。
    """
    client = anthropic.Anthropic(api_key=API_KEY)
    close, volume = load_data(TICKERS, START, END)
    returns_next = close.pct_change(1).shift(-1)

    all_results   = []   # 所有轮次的所有因子
    prev_results  = []   # 上一轮的结果，传给下一轮 LLM

    print("=" * 60)
    print(f"市场假设：{hypothesis}")
    print(f"迭代轮数：{n_rounds}  每轮因子数：{N_FACTORS}")
    print("=" * 60)

    for round_num in range(1, n_rounds + 1):
        print(f"\n【第 {round_num} 轮】生成因子中...")

        factors = llm_generate_factors(client, hypothesis, prev_results, round_num)

        if not factors:
            print("  LLM 未返回有效因子，跳过本轮")
            continue

        round_results = []

        for f in factors:
            name  = f.get("name", "unnamed")
            expr  = f.get("expr", "")
            logic = f.get("logic", "")

            metrics = evaluate_factor(expr, close, volume)

            result = {
                "round": round_num,
                "name":  name,
                "expr":  expr,
                "logic": logic,
                **metrics,
            }
            round_results.append(result)
            all_results.append(result)

            # 打印结果
            if "error" not in metrics:
                flag = "✓" if abs(metrics["mean_ic"]) >= IC_THRESHOLD else "✗"
                print(
                    f"  {flag} {name:<30} "
                    f"IC={metrics['mean_ic']:+.4f}  "
                    f"ICIR={metrics['icir']:+.4f}  "
                    f"IC>0={metrics['ic_pos']:.0%}"
                )
                print(f"    逻辑: {logic}")
                print(f"    逐年IC: {metrics['yearly_ic']}")
            else:
                print(f"  ✗ {name:<30} 报错: {metrics['error']}")
                print(f"    表达式: {expr}")

        # 只把有效结果传给下一轮
        prev_results = [r for r in round_results if "error" not in r]

    return all_results


# ─────────────────────────────────────────────────────────
# 5. 结果汇总
# ─────────────────────────────────────────────────────────

def summarize(all_results: list):
    """打印最终排行榜"""
    valid = [r for r in all_results if "error" not in r]

    if not valid:
        print("\n没有找到有效因子。")
        return

    # 按 |ICIR| 排序
    ranked = sorted(valid, key=lambda x: abs(x["icir"]), reverse=True)

    print("\n" + "=" * 60)
    print("最终排行榜（按 |ICIR| 降序）")
    print("=" * 60)
    print(f"{'排名':<4} {'名称':<30} {'IC':>7} {'ICIR':>7} {'IC>0':>6} {'轮次':>4}")
    print("-" * 60)

    for i, r in enumerate(ranked, 1):
        print(
            f"{i:<4} {r['name']:<30} "
            f"{r['mean_ic']:>+7.4f} "
            f"{r['icir']:>+7.4f} "
            f"{r['ic_pos']:>6.0%} "
            f"R{r['round']:>3}"
        )

    print("\n最优因子详情：")
    best = ranked[0]
    print(f"  名称  : {best['name']}")
    print(f"  表达式: {best['expr']}")
    print(f"  逻辑  : {best['logic']}")
    print(f"  IC    : {best['mean_ic']:+.4f}")
    print(f"  ICIR  : {best['icir']:+.4f}")
    print(f"  逐年IC: {best['yearly_ic']}")

    # 保存结果到 JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"factor_search_{timestamp}.json"
    with open(fname, "w", encoding="utf-8") as fp:
        json.dump(ranked, fp, ensure_ascii=False, indent=2)
    print(f"\n结果已保存到 {fname}")


# ─────────────────────────────────────────────────────────
# 入口
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    HYPOTHESIS = (
        "特朗普关税政策引发市场震荡，"
        "短期内超跌股票存在均值回归机会，"
        "同时成交量异常可能预示机构资金流向"
    )

    results = run_factor_search(HYPOTHESIS, n_rounds=N_ROUNDS)
    summarize(results)