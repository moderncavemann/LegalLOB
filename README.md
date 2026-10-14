<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/logo-dark.png">
    <img src="assets/logo.png" alt="LegalLOB" width="60%">
  </picture>
</p>

<h1 align="center">LegalLOB: Generating Structurally Valid Level-2 Limit Order Book Trajectories with Diffusion Models</h1>

<p align="center">
  <a href="#-citation"><img src="https://img.shields.io/badge/ICAIF-2026-1F4E9E.svg" alt="ICAIF 2026"></a>
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB.svg?logo=python&amp;logoColor=white" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg?logo=pytorch&amp;logoColor=white" alt="PyTorch 2.x">
  <a href="REPRODUCE.md"><img src="https://img.shields.io/badge/Reproduce-4%20Nasdaq%20settings-2E8B57.svg" alt="Reproduction guide"></a>
  <a href="https://github.com/moderncavemann/LegalLOB/stargazers"><img src="https://img.shields.io/github/stars/moderncavemann/LegalLOB?style=flat&amp;logo=github&amp;color=1F4E9E" alt="GitHub stars"></a>
  <a href="https://github.com/moderncavemann/LegalLOB/forks"><img src="https://img.shields.io/github/forks/moderncavemann/LegalLOB?style=flat&amp;logo=github&amp;color=1F4E9E" alt="GitHub forks"></a>
  <a href="https://hitscounter.dev/history?url=https%3A%2F%2Fgithub.com%2Fmoderncavemann%2FLegalLOB"><img src="https://hitscounter.dev/api/hit?url=https%3A%2F%2Fgithub.com%2Fmoderncavemann%2FLegalLOB&amp;label=Views&amp;icon=graph-up&amp;color=%231f4e9e&amp;message=&amp;style=flat&amp;tz=Australia%2FSydney" alt="Page views: today / total"></a>
</p>

<p align="center">
  <a href="README.md"><strong>English</strong></a> | <a href="README.zh-CN.md">简体中文</a>
</p>

> **Authors:** Guanghao Wu<sup>1,\*</sup>, Yixuan Xu<sup>2,\*</sup>, Shoujin Wang<sup>1,†</sup><br>
> <sup>1</sup>University of Technology Sydney &nbsp; <sup>2</sup>Independent Researcher, Sydney<br>
> <sup>\*</sup>Equal contribution &nbsp; <sup>†</sup>Corresponding author<br>
> 7th ACM International Conference on AI in Finance (ICAIF '26), Milan, Italy

LegalLOB is a conditional diffusion model for Level-2 limit order book (LOB) trajectories whose generated books are **valid by construction**. A valid book obeys three rules: the best ask stays above the best bid, prices stay ordered across levels, and displayed depth is non-negative. LegalLOB describes each book by the quantities these rules constrain and decodes them with softplus links and cumulative sums, so every book it decodes, including every clean-state prediction during sampling, is valid without penalties, rejection, or projection.

- 🧱 **Constraint-aligned representation:** a free midpoint plus the spread, the gaps between adjacent levels, and the depths, so every rule becomes a lower bound on one number.
- ➕ **Softplus decoder:** softplus links keep each number above its bound, and cumulative sums rebuild the price ladders; the diffusion objective and sampler stay unchanged.
- 🔁 **Valid along the whole sampling path:** guidance, early stopping, or a recursive simulation can read any decoded prediction without repairing it.

<sub>The name refers to structural validity, not to legal compliance.</sub>

## 🧭 Why LegalLOB?

Diffusion models can generate plausible future Level-2 books for simulation and stress testing, but they output unconstrained prices and depths. Even a model with low denoising error can produce crossed quotes, inverted levels, or negative depth, and such books must be rejected or repaired before risk calculations, downstream models, or the next step of a recursive simulation can use them. Training penalties make violations rarer without ruling them out. Exact projection with the pool-adjacent-violators algorithm (PAVA) repairs a book cheaply, but only the books it is given: every intermediate prediction needs its own projection when it is read, and projecting inside the sampler changes the sampling path.

<p align="center">
  <img src="assets/overview.png" alt="Validity along the sampling path" width="80%">
</p>

<p align="center"><sub>Validity along the sampling path (schematic). Each row shows the decoded clean-state prediction at reverse steps <i>N</i>, <i>k</i> and 1 and the output, and the number of exact projections (Π) per 100-step GOOG-Event rollout. Post-hoc PAVA makes only the output valid; in-loop PAVA projects every prediction and feeds the repair back into sampling; LegalLOB is valid everywhere without projection.</sub></p>

### 🚫 Violations Are the Norm

Before any repair, the unconstrained variants produce invalid books in about half or more of their snapshots on every dataset. The constraint-aligned coordinates alone do not remove the violations; the softplus decoder does.

| Variant | AAPL-1Hz | AMZN-1Hz | GOOG-Event | GOOG-1Hz |
| --- | ---: | ---: | ---: | ---: |
| **LegalLOB** (coordinates + softplus decoder) | **0** | **0** | **0** | **0** |
| Coord-only (coordinates, identity links) | 99.9 | 48.3 | 78.5 | 59.3 |
| Raw (prices and depths) | 100.0 | 59.9 | 84.9 | 70.1 |
| Raw+Penalty (training penalty) | 100.0 | 59.6 | 84.4 | 69.4 |

<p align="center"><sub>Invalid snapshots before repair (%), mean over three independently trained checkpoints; each evaluation covers 6,400 histories with ten future snapshots.</sub></p>

### 🎯 Three Challenges

- **Coupled constraints.** In raw prices, whether a book is crossed depends jointly on two prices, and whether a ladder is ordered depends on every adjacent pair.
- **Validity without external repair.** A network can always output a value below its bound. Validity must hold without changing the diffusion objective, adding a projection to sampling, or collapsing the generated distribution.
- **The source and the value of the guarantee.** Zero violations could also come from easier coordinates or a stronger network, and exact projection also yields valid final books.

## ⚙️ How It Works

LegalLOB keeps the conditional diffusion model, a DiffLOB-style denoiser trained with the usual noise-prediction objective and sampled with DDIM, and changes only how its clean-state prediction is decoded into a book.

<p align="center">
  <img src="assets/framework.png" alt="Framework of LegalLOB and the compared output interfaces" width="100%">
</p>

<p align="center"><sub>All variants share the denoiser architecture, the noise-prediction objective and the DDIM sampler (top) and differ in how the clean-state prediction is decoded into a book (bottom): (A) identity links; (B) identity links followed by exact projection, once after decoding or at every reverse step; (C) the LegalLOB softplus decoder.</sub></p>

**1. Constraint-aligned representation.** Relative to the midpoint of the last observed book, each future book is described by its midpoint offset, the bid–ask spread $s$, the gaps $g^a_k$ and $g^b_k$ between adjacent ask and bid levels, and the depths $v$. Every ordering rule becomes a lower bound on a single coordinate, while the midpoint stays free.

**2. Softplus decoder.** With the softplus function $\rho(u)=\log(1+e^u)$, a minimum spread $\delta$, a minimum gap $\gamma$ and a depth floor $\epsilon_v$, the decoder maps the network output $u$ to

$$
s=\delta+\rho(u_s),\qquad g^a_k=\gamma+\rho(u^a_k),\qquad g^b_k=\gamma+\rho(u^b_k),\qquad v=\epsilon_v+\rho(u_v),
$$

$$
a^{(1)}=m+\tfrac{s}{2},\qquad b^{(1)}=m-\tfrac{s}{2},\qquad a^{(k)}=a^{(1)}+\sum_{j=2}^{k}g^a_j,\qquad b^{(k)}=b^{(1)}-\sum_{j=2}^{k}g^b_j .
$$

where $m$ is the anchor midpoint plus the generated midpoint offset. In exact arithmetic, this is a smooth bijection onto the interior of the set of valid books, so every network output decodes to a strictly valid book, and its inverse encodes the training targets. In the experiments, $\delta=\gamma$ is one exchange tick and $\epsilon_v=0$.

**3. Controlled comparison.** The baselines use the same denoiser architecture, hyperparameters, data and training budget, and impose validity nowhere (Raw, Coord-only), through a training penalty (Raw+Penalty), by exact PAVA projection after sampling (+Post), or by exact PAVA projection at every denoising step (+In-loop).

The representation and the decoder are `LegalLOBTransform` in [`genlob/utils/legal_lob.py`](genlob/utils/legal_lob.py), next to `MatchedLOBTransform`, the coordinates-only baseline. Validity checks and exact PAVA projection are in [`genlob/utils/physics.py`](genlob/utils/physics.py).

## 📊 Results

LegalLOB is the only learned model with no violations before repair on all four datasets, and in a 100-step GOOG-Event rollout it is the only method whose decoded predictions can all be used without projection. It keeps 99.6% of the unconstrained model's throughput. The guarantee is not free in fidelity: post-hoc PAVA on the coordinates-only model (Coord-only+Post), the strongest repair baseline, leads on 24 of the 36 fidelity comparisons, while LegalLOB's probabilistic scores on GOOG-Event stay within a few percent of it.

### 🔁 Intermediate States in a Recursive Rollout

A fail-closed monitor reads the decoded prediction at every denoising step and every block endpoint of a 100-step GOOG-Event rollout and rejects any invalid snapshot.

<table>
  <thead>
    <tr>
      <th align="left">Method</th>
      <th align="center">Projection calls<br>per path ↓</th>
      <th align="center">Cumulative<br>repair ↓</th>
      <th align="center">Rejected snapshot<br>requests (%) ↓</th>
      <th align="center">Paths with a<br>rejection (%) ↓</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>Coord-only+Post</td><td align="center">10</td><td align="center">0.077</td><td align="center">51.9</td><td align="center">100</td></tr>
    <tr><td>Coord-only+In-loop</td><td align="center">200</td><td align="center">159.1</td><td align="center"><ins>1.0 × 10<sup>−4</sup></ins></td><td align="center"><ins>0.19</ins></td></tr>
    <tr><td><b>LegalLOB</b></td><td align="center"><b>0</b></td><td align="center">no projection</td><td align="center"><b>0</b></td><td align="center"><b>0</b></td></tr>
  </tbody>
</table>

<p align="center"><sub>Mean over three checkpoints, each with 640 histories and three sampling seeds (11.5 million snapshot requests per method). Rejections are counted after each method's own projection. Cumulative repair: total normalized L2 change that projection makes along a path. Coord-only+Post repairs only block endpoints, so about half of the intermediate requests stay invalid; Coord-only+In-loop repairs every request but changes the sampling path.</sub></p>

### ⏱️ Cost and Probabilistic Accuracy

<table>
  <tr>
    <td width="45%" valign="top">

| Method | Snapshots/s ↑ |
| --- | ---: |
| Raw (unconstrained) | 1300.0 |
| Raw+Post | <ins>1237.9</ins> |
| Raw+In-loop | 709.4 |
| Coord-only+Post | 1237.8 |
| **LegalLOB** | **1294.9** |

<sub>Generation of 1,280 future snapshots on the 1-Hz datasets (DDIM-20, Apple M3 Ultra). Ranked among methods with valid outputs.</sub>

  </td>
  <td width="55%" valign="top">

| Method | Energy score ↓ | CRPS ↓ | Variogram ↓ |
| --- | ---: | ---: | ---: |
| Persistence | 0.4567 | 0.1832 | 0.1128 |
| Raw+Post | 0.5867 | 0.2768 | 0.0874 |
| Coord-only+Post | **0.3095** | **0.1452** | **0.0826** |
| **LegalLOB** | <ins>0.3157</ins> | <ins>0.1513</ins> | <ins>0.0832</ins> |

<sub>GOOG-Event, 512 histories with 32 samples each. LegalLOB has the most dispersed ensemble; neither LegalLOB nor Coord-only+Post is perfectly calibrated.</sub>

  </td>
  </tr>
</table>

<sub><b>Bold</b>: best; <ins>underlined</ins>: second best. Mean over three checkpoints.</sub>

### 📈 What the Guarantee Costs and Keeps

<table>
  <tr>
    <td width="50%" align="center" valign="top">
      <a href="assets/burden.png"><img src="assets/burden.png" alt="Cumulative repair per path" width="100%"></a><br>
      <sub><strong>How much repair does each method need?</strong> Cumulative repair per 100-step GOOG-Event path (log scale) and projection calls. In-loop projection moves each state much further than post-hoc projection; LegalLOB needs none.</sub>
    </td>
    <td width="50%" align="center" valign="top">
      <a href="assets/marginals.png"><img src="assets/marginals.png" alt="GOOG-Event marginal distributions" width="100%"></a><br>
      <sub><strong>Does the guarantee distort the book?</strong> GOOG-Event spread, depth imbalance, Liq@5 and slippage: the generated and real (gray) distributions closely overlap.</sub>
    </td>
  </tr>
</table>

LegalLOB has the lowest microprice distance among the stochastic methods on three datasets and the lowest Liq@5 distance on both GOOG datasets, but Coord-only+Post is better on most other fidelity metrics, by a wide margin on some temporal diagnostics. LegalLOB also has the highest point error among the stochastic methods on the three 1-Hz datasets.

### 🤔 When to Use LegalLOB

If only the final book is used, post-hoc PAVA is the natural default: it is exact, adds about 5% to the runtime, and, applied to the coordinates-only model, leads on most fidelity metrics. LegalLOB matters when earlier states are read, for example by guidance that scores the clean-state prediction at every denoising step, by anytime sampling that stops early under a latency budget, or by monitors of recursive simulations. Projection then needs one call per read, and feeding the repaired states back to the sampler changes the generated path, whereas LegalLOB's predictions are valid as generated and pass gradients through a smooth decoder.

## 🧪 Datasets

| Setting | Data | Dataset ID |
| --- | --- | --- |
| **AAPL-1Hz, AMZN-1Hz, GOOG-1Hz** | Nasdaq public TotalView-ITCH samples: the last valid regular-hours book in each second on six days in 2019 (training Jan 30, Mar 27 and Jul 30; validation Aug 30; test Oct 30 and Dec 30); 140,400 snapshots per ticker. | `PUBLIC_ITCH_2019_SAMPLES_<TICKER>_1s` |
| **GOOG-Event** | Databento XNAS.ITCH MBP-10: every valid regular-hours GOOG book update on Jan 30, 2019, split chronologically 70/15/15 with 100-row purged gaps; 856,984 snapshots. | `GOOG_databento_20190130_mbp10_rth_v2` |

All settings use the top ten levels per side, 50 history and 10 future snapshots, prices in units of ten exchange ticks, and square-root depth clipped at the 99th percentile of the training split. Market data are not redistributed: the scripts download the public ITCH samples and fetch the Databento sample with your own API key.

## 🚀 Quick Start

Python 3.10+ and PyTorch 2 (CUDA, MPS or CPU); install the PyTorch build for your platform first. This example trains and evaluates LegalLOB on GOOG-1Hz.

```bash
git clone https://github.com/moderncavemann/LegalLOB.git
cd LegalLOB
pip install -r requirements.txt

# Data: the six public Nasdaq ITCH sample days of 2019 (about 25 GB in total)
bash scripts/download_nasdaq_2019_samples.sh
python scripts/prepare_public_itch_multisymbol.py \
  --itch-file data/raw/nasdaq_itch/public/01302019.NASDAQ_ITCH50.gz \
  --itch-file data/raw/nasdaq_itch/public/03272019.NASDAQ_ITCH50.gz \
  --itch-file data/raw/nasdaq_itch/public/07302019.NASDAQ_ITCH50.gz \
  --itch-file data/raw/nasdaq_itch/public/08302019.NASDAQ_ITCH50.gz \
  --itch-file data/raw/nasdaq_itch/public/10302019.NASDAQ_ITCH50.gz \
  --itch-file data/raw/nasdaq_itch/public/12302019.NASDAQ_ITCH50.gz \
  --symbols AAPL,AMZN,GOOG --sampling both --purge-gap 100 --update-manifest

# Train LegalLOB (50 epochs; about 1.5 hours on an Apple M3 Ultra)
python scripts/train_v2.py --dataset PUBLIC_ITCH_2019_SAMPLES_GOOG_1s --data-dir data \
  --representation legal_lob --hist-len 50 --pred-len 10 --batch-size 128 --epochs 50 \
  --lr 1e-4 --cfg-drop 0.1 --early-stopping-patience 10 --action-conditioning none \
  --physics-loss-weight 0 --action-contrastive-weight 0 \
  --save-dir checkpoints/goog_1hz_legal_seed0 --num-workers 0 --seed 0 --device mps

# Generate 6,400 test windows with DDIM-20 and evaluate them
python scripts/evaluate_v2.py --checkpoint checkpoints/goog_1hz_legal_seed0/best.pt --weights ema \
  --dataset PUBLIC_ITCH_2019_SAMPLES_GOOG_1s --data-dir data --split test \
  --n-batches 50 --batch-size 128 --ddim-steps 20 --projection-mode none \
  --representation auto --output-dir results/goog_1hz_legal_seed0 --device mps
```

`evaluate_v2.py` saves the generated arrays and writes validity rates, point error and summary statistics to `metrics.json` in the output directory. `--representation raw` or `matched_lob` trains the Raw or Coord-only baseline, and `--projection-mode exact` or `--sampler-projection-mode exact` adds post-hoc or in-loop PAVA at evaluation. The [reproduction guide](REPRODUCE.md) gives the commands behind every table and figure of the paper.

## 🗂️ Repository Layout

| Path | Purpose |
| --- | --- |
| [`genlob/utils/legal_lob.py`](genlob/utils/legal_lob.py) | LegalLOB representation and softplus decoder; the coordinates-only transform. |
| [`genlob/utils/physics.py`](genlob/utils/physics.py) | Validity checks, exact PAVA projection and the training penalty. |
| [`genlob/models/`](genlob/models/) | Conditional denoiser, VP diffusion schedule, and DDIM sampling with an optional in-loop projection. |
| [`genlob/data/`](genlob/data/), [`genlob/eval/`](genlob/eval/) | Snapshot loader, training-split normalization and evaluation metrics. |
| [`scripts/`](scripts/) | Data preparation, training, evaluation, rollout and intermediate-state studies, probabilistic scores, cost benchmark and paper summaries. |
| [`experiments/multi_asset_event_time/`](experiments/multi_asset_event_time/) | A separate fixed-budget study on AAPL, AMZN, GOOG and INTC event-time books, not part of the ICAIF paper. |
| [`data/`](data/) | Layout and manifests of the processed datasets; market data are not included. |
| [`tests/`](tests/) | Unit and workflow tests (`python -m pytest -q`). |
| [`assets/`](assets/) | Logo and figures of this README. |

## 📚 Documentation

- Paper: *LegalLOB: Generating Structurally Valid Level-2 Limit Order Book Trajectories with Diffusion Models*, ICAIF '26, DOI [10.1145/3858063.3858881](https://doi.org/10.1145/3858063.3858881) (active once the proceedings are published).
- [Reproduction guide](REPRODUCE.md): the four settings, the commands behind every table and figure, and where their outputs are written.
- [`data/README.md`](data/README.md): layout of the processed datasets.

LegalLOB is released under the [MIT License](LICENSE).

## 🖋 Citation

If this repository is useful for you, please cite as:

```bibtex
@inproceedings{wu2026legallob,
      title={LegalLOB: Generating Structurally Valid Level-2 Limit Order Book Trajectories with Diffusion Models},
      author={Guanghao Wu and Yixuan Xu and Shoujin Wang},
      booktitle={Proceedings of the 7th ACM International Conference on AI in Finance (ICAIF '26)},
      year={2026},
      address={Milan, Italy},
      publisher={ACM},
      doi={10.1145/3858063.3858881},
      url={https://doi.org/10.1145/3858063.3858881},
}
```

If you have any questions, please open an issue.
