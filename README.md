# Safety Limits of Capped Potential-Game Flocking: numerical experiments

Code and data for the numerical results of "Safety Limits of Capped Potential-Game Flocking" (A. Sreejith).
All experiments use the Flockwork law in code/swarm_model.py with two parameter sets:
reference (Olfati-Saber 2006, Virágh et al. 2014) and implementation (deployed controller defaults).

Tested with Python 3.12 and the versions in requirements.txt. Every run is deterministic given its seed.
Run all commands from the repository root.

```bash
pip install -r requirements.txt
```

## Layout

- code: model, experiments, and plotting scripts.
- results: saved sweeps (.npy) and run logs.
- figures: paper figures as PDF with PNG previews.

## Paper results

| Result | Command | Output |
|---|---|---|
| Cutoff check, implementation (Sec. VII-B) | python code/exp_cutoff_time.py | stdout |
| Cutoff and sampled-data checks, reference (Sec. VII-B, VII-C) | python code/run_reference.py | stdout |
| Sampled-data descent, implementation (Sec. VII-C) | python code/exp_sampled_late.py > results/sampled_late.log | sampled_late.log |
| Critical size N*, 2D (Sec. VII-D) | python code/exp_crush.py reference 100 > results/crush_reference_spread.log | crush_reference_spread.npy, .log |
| | python code/exp_crush.py implementation 40 > results/crush_implementation_spread.log | crush_implementation_spread.npy, .log |
| Settling recheck of unsettled packs (Sec. VII-A) | python code/exp_crush_settle.py reference 1.5:8 3:24 3:48 3:60 4:24 4:60 4:75 > results/crush_reference_settle.log | crush_reference_settle.log |
| Faithful-descent check (Sec. VII-A) | python code/exp_crush_monotone.py > results/crush_monotone.log | crush_monotone.log |
| Safe ring equilibria (Sec. VII-D) | python code/exp_rings.py reference 10 | stdout |
| Ring stability under perturbation (Sec. VII-D) | python code/exp_rings_stability.py | stdout |
| Critical size N*, 3D (Sec. VII-D) | python code/exp_crush3d.py 1 1.5 2 3 4 > results/crush_reference_3d.log | crush_reference_3d.npy, .log |
| Critical size figure, 2D | python code/plot_crush.py | fig_crush.pdf |
| Settled 3D pack and 3D N* figures | python code/plot_crush3d.py | fig_pack3d.pdf, fig_crush3d.pdf |

## Code

- swarm_model.py: pair force, potentials, control law, parameter sets, output folders.
