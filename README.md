# Token-Aware Joint Optimization for Ground–LEO Collaborative Multimodal AIGC Inference

Simulation code and calibration data for the correspondence
*Token-Aware Joint Optimization for Ground–LEO Collaborative Multimodal AIGC Inference*
(submitted to IEEE Transactions on Vehicular Technology).

The repository contains three parts:

| Part | Directory | What it reproduces |
|---|---|---|
| Per-slot joint optimization and baselines | `src/`, `config.py` | Closed-form inner solutions (Theorem 1), dual updates with certified stopping (Algorithm 1), and the four baselines of Sec. IV-A |
| Simulation sweeps and figures | `experiments/`, `figures/` | Table II, Fig. 3(a)–(d), and the supplementary tables (S-IV to S-VIII) |
| Profile calibration on Qwen3-VL | `calibration/` | The task-conditioned rate–distortion profiles $(\delta, \theta)$ of Sec. II-B, Fig. 2, and Tables S-I to S-III |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q tests
```

Python 3.10 or newer, NumPy, SciPy, NetworkX, and Matplotlib are required.
PyTorch is used by the TD3-RL baseline only. The `openai` package is
optional and unused in the reported results.

## Reproducing the simulation results

All scripts are run from the repository root.

```bash
# Main sweep: five schemes, U = 10..40, 5 seeds x 50 slots (about one hour on 10 cores)
python -m experiments.run_all --U 10,15,20,25,30,35,40 --seeds 5 --slots 50 --procs 10

# Auxiliary sweeps used in Sec. IV-B and the supplementary material
python -m experiments.aux_experiments w1     # Q_eff against w_1 at U = 30 (Fig. 3(b), Table S-VIII)
python -m experiments.aux_experiments eta    # eta* against nu for the four task profiles (Fig. 3(c))
python -m experiments.aux_experiments dual   # dual function and recovered objective per iteration
python -m experiments.aux_experiments isl    # ISL slice sensitivity (Table S-V)
python -m experiments.aux_margins            # return-link margin check (Sec. S2)

# Figures and headline numbers
python -m experiments.make_fig2              # Fig. 2 (fig_calib) and Fig. 3 (fig1_combined) into figures/,
                                             # and Table II with the percentage gains printed to stdout
```

The outputs of the sweeps used in the paper are included so that the
figures and tables can be regenerated without rerunning anything:
`figures/data/run_all.json.gz` holds the per-user records of the main
sweep (Table II, Fig. 3(a), Fig. 3(d), Tables S-IV, S-VI, S-VII), and
`figures/data/aux_*.json` hold the auxiliary sweeps. `make_fig2.py` reads
the compressed file directly.

The TD3-RL baseline uses the trained policy in `data/td3_checkpoint.pt`
(training log in `data/td3_train_log.npz`, hyperparameters in Sec. S5).
It can be retrained with

```bash
python -m src.baselines.td3_rl --train --n_episodes 5000
```

### Scheme names

| Paper | Code |
|---|---|
| Proposed | `src/optimizer.py` (`solve_per_slot`) |
| Profile-Oblivious | `src/baselines/profile_oblivious.py` |
| Rule-Based Offloader | `src/baselines/icl_llm.py` (rule-based path; the optional LLM path is not used) |
| TD3-RL | `src/baselines/td3_rl.py` |
| Fixed-η Dijkstra | `src/baselines/fixed_eta_dijkstra.py` |

## Profile calibration

The profiles in `data/calib_table.json` and
`calibration/data/profile_lookup_final.json` were calibrated on 440
image–text questions from VQAv2, TextVQA, DocVQA, and ChartQA with
Qwen3-VL-32B-Instruct (MLX 4-bit, satellite model) answering under
importance-ranked visual-token retention, and Qwen3-VL-2B-Instruct
(FP16, on-device model) providing the ranking from the query-to-visual
attention at layer index 12. Both models were run on an Apple M3 Max.

The dataset images and question texts are not redistributed here. The
exact question set is identified by `calibration/data/question_ids.json`
(source, image id, question id, and calibration/held-out split), and
`calibration/data.py` rebuilds the questions and images from the Hugging
Face copies of the datasets. The model outputs are included:

| File | Content |
|---|---|
| `calibration/data/answers_final.jsonl` | 32B answers and VQA accuracy per question, retention ratio, and selection mode |
| `calibration/data/scores_final.npz` | Per-question importance distributions, normalized entropy $H_u$, and token counts from the 2B scoring pass |
| `calibration/data/profile_lookup_final.json` | Fitted $(\delta, \theta)$ per task class with calibration and held-out RMSE |
| `calibration/data/selected_rule.json` | Selection-rule comparison that fixed the ranking rule before the final run |
| `calibration/data/timing_sweep.json` | Measured prefill and decode times of the satellite model against context length |

The pipeline is `calibration/pipeline_final.sh` (scoring pass, answers at
eight retention ratios, timing sweep, and profile fitting). Running it
requires the `transformers`, `mlx-vlm`, and `datasets` packages and about
a day on the hardware above; `calibration/final_analysis.py` alone
refits the profiles from the included outputs in seconds.

## Layout

```
config.py                 system parameters (Table I) and symbol names of the paper
src/inner_solve.py        closed-form bandwidth/compute allocation and eta* (Theorem 1)
src/optimizer.py          dual decomposition with certified stopping and recovery (Algorithm 1)
src/data_gen.py           request, channel, and constellation generation
src/semantic_profile.py   rate-distortion profile S(eta) and effective quality
src/baselines/            the four baselines
experiments/              sweeps and figure scripts (Table II is printed by make_fig2)
figures/data/             outputs of the sweeps used in the paper
calibration/              Qwen3-VL calibration pipeline and its outputs
tests/                    checks of the closed forms and of the monotonicity of eta*
```

## License

MIT. See `LICENSE`.
