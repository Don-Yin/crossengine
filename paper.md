---
title: "crossengine: a multi-asset backtesting engine with cross-engine concordance testing"
tags:
  - python
  - finance
  - backtesting
  - portfolio
  - concordance
  - implementation risk
authors:
  - name: Don Yin
    orcid: 0000-0002-8971-1057
    affiliation: 1
  - name: Takeshi Miki
    orcid: 0009-0000-2063-6274
    affiliation: 1
  - name: Vladislav Lesnichenko
    orcid: 0009-0004-9023-3613
    affiliation: 2
affiliations:
  - name: University of Cambridge, United Kingdom
    index: 1
  - name: Independent researcher, United Kingdom
    index: 2
date: 20 March 2026
bibliography: paper.bib
---

# summary

`crossengine` is a Python package that provides (1) a multi-asset portfolio backtesting engine and (2) a cross-engine concordance testing API. the engine implements the proportional-cost specification from Algorithm 1 of the companion paper [@yin2026implementation]. the concordance API runs the same strategy through five independent backtesting engines and quantifies how much they disagree.

we built this tool after discovering that identical strategies, data, and cost parameters fed to five engines produce equity curves that diverge by over 3%. this is not a bug in any single engine but a structural property of how each implements execution order, commission timing, and cash settlement. we call this "implementation risk."

![equity curves for all 15 benchmark strategies grouped by category, each run on 30 stratified asset buckets (180 S&P 500 stocks) with the S&P 500 as reference. `crossengine` measures whether all five engines agree on these curves.](figures/equity-overlay.png)

this software accompanies the paper "Implementation Risk in Portfolio Backtesting" [@yin2026implementation], submitted to Financial Innovation and available on arXiv (q-fin.CP).

# statement of need

no existing tool allows a practitioner to validate backtesting results across engines. each engine operates as an isolated system, and a strategy producing 14.5% CAGR in one engine may show 14.2% in another. the user has no way to detect this without manually reimplementing their strategy in every engine, a process that takes days per engine and is itself error-prone.

the divergence is silent: every engine returns a confident equity curve and precise Sharpe ratio, with nothing in the output to signal disagreement. during forensic analysis, we found that a single undocumented default in vectorbt (`call_seq` order) caused a 31% equity divergence, and that Backtrader's default commission handling silently undercharges by 100x due to an unintuitive `percabs` parameter. these are the defaults most users encounter, not exceptional configurations.

![divergence scales with transaction costs. each point is one of 15 benchmarks averaged over 30 stratified asset buckets. zero-cost strategies produce 0.0% divergence; high-turnover strategies with costs produce up to 3.8%.](figures/divergence-vs-cost.png)

# state of the field

the Python ecosystem offers several mature backtesting engines: Backtrader [@backtrader2024] pioneered event-driven simulation, bt [@pmorissette2024bt] introduced composable algo trees, vectorbt [@vbt2024] brought vectorised speed, and cvxportfolio [@boyd2017cvxportfolio] added convex cost models. adjacent tools such as PyPortfolioOpt [@martin2021pyportfolioopt] handle portfolio construction rather than execution simulation. academic work on backtesting pitfalls has focused on overfitting [@bailey2017probability] and on the correctness of individual engines [@low2017correctness], but no prior work compares outputs across engines to quantify implementation-level disagreement.

`crossengine` fills this gap by providing automated cross-engine concordance testing. rather than improving any single engine, it orchestrates five of them simultaneously to measure divergence, a capability that cannot be added to any one engine alone.

# software design

![engine sensitivity (ES) across 15 benchmarks and 10 metrics (5 base metrics with ranges), computed over 30 independent asset buckets each. most strategies show near-zero sensitivity; rotation and concentrated strategies show material divergence in return and CAGR.](figures/engine-concordance.png)

the package uses a two-layer architecture. the backtesting engine accepts a signals DataFrame (with optional STAY sentinels) and produces a `BacktestResult` with portfolio value, positions, trades, and metrics. the concordance layer uses a `SignalSchedule` dict as its interchange format, which each engine wrapper translates into its native API.

the STAY signal is a first-class "hold and drift" instruction that freezes the share count rather than the weight. engines with access to live portfolio state (ours, bt, Backtrader) resolve STAY natively at runtime. engines that receive signals upfront (vectorbt, cvxportfolio) receive pre-resolved drifted weights via forward simulation, which introduces at most 0.045% additional divergence. the tool measures divergence, not correctness: when engines disagree, the concordance API quantifies the disagreement but does not determine which engine is right.

# research impact statement

the companion paper [@yin2026implementation] uses `crossengine` to validate 15 benchmark strategies (12 rule-based plus 4 ML approaches) across 30 non-overlapping asset buckets drawn from 180 S&P 500 stocks. all 150 equity curves (5 engines x 30 buckets) match the saved results to sub-penny tolerance ($0.01). the package includes 44 tests. the engine sensitivity (ES) metric introduced by this tool provides a reproducible, quantitative measure of implementation risk for any portfolio strategy.

# AI usage disclosure

generative AI tools (Claude, Anthropic) were used during development for code generation and editing. all outputs were reviewed and validated by the authors.

# acknowledgements

this work was conducted at the University of Cambridge. we thank the developers of bt, vectorbt, Backtrader, and cvxportfolio for making their engines open source. the forensic case studies in the companion paper would not have been possible without access to these codebases.

# references
