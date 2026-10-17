# Flockwork theory: numerical experiments

Code for the numerical results of "Safety Limits of Capped Potential-Game Flocking" (A. Sreejith).
All experiments use the idealized Flockwork law in `swarm_model.py` with two parameter sets:
`reference` (Olfati-Saber 2006, Virágh et al. 2014) and `implementation` (deployed controller defaults).

Tested with Python 3.12 and the versions in `requirements.txt`. Every run is deterministic given its seed.

```bash
pip install -r requirements.txt
```

## Paper results

| Result | Command | Output |
|---|---|---|
| Cutoff check, implementation (Sec. VII-B) | `python exp_cutoff_time.py` | stdout |
| Cutoff and sampled-data checks, reference (Sec. VII-B, VII-C) | `python run_reference.py` | stdout |
| Sampled-data descent, implementation (Sec. VII-C) | `python exp_sampled_late.py` | `sampled_late.log` |
| Critical size N*, 2D (Sec. VII-D, Fig. 2) | `python exp_crush.py reference 100`, `python exp_crush.py implementation 40` | `crush_*_spread.npy`, `crush_*_spread.log` |
| Settling recheck of unsettled packs (Sec. VII-A) | `python exp_crush_settle.py reference 1.5:8 3:24 3:48 3:60 4:24 4:60 4:75` | `crush_reference_settle.log` |
| Faithful-descent check of the step (Sec. VII-A) | `python exp_crush_monotone.py` | `crush_monotone.log` |
| Safe ring equilibria (Thm. 5, Sec. VII-D) | `python exp_rings.py reference 10` | stdout |
| Ring stability under perturbation (Sec. VII-D) | `python exp_rings_stability.py` | stdout |
| Critical size N*, 3D (Remark 9) | `python exp_crush3d.py 1 1.5 2 3 4` | `crush_reference_3d.npy`, `crush_reference_3d.log` |
| Fig. 2 | `python plot_crush.py` | `fig_crush.png` |

## Files

- `swarm_model.py`: pair force, potentials, control law, parameter sets.
- `plot_style.py`: shared figure style.
