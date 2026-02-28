# ReadMe: Multi-Agent Surround Experiments

## 1. Introduction
This project uses QMIX (Multi-Agent Deep Reinforcement Learning) to train agents in the SISL Pursuit environment. The experiments primarily vary **grid size**, **capture constraints** (Surround vs. Touch), and **evader dynamics** (Freeze vs. Active). Parameters such as `n_pursuers` and `n_evaders` are automatically scaled based on the grid size.

---


## 2. Execution Guide

You can run the training and evaluation via the main entry point:

```bash
python main.py
```

### Options:
- **Rendering**: You will be asked if you want to view the simulation window (`y/n`).
- **Experiment Selection**: 
    - `1-12`: Run a specific experiment configuration.
    - `0`: **Run All Experiments** - Executes all 12 experiments sequentially.

---

## 3. Study Design (12 Experiments)

| ID | Grid | Kill Enemies with | Enemy Movement |
| :--- | :--- | :--- | :--- |
| **1-4** | 8x8 | Surround/Touch | Freeze/Active |
| **5-8** | 12x12 | Surround/Touch | Freeze/Active |
| **9-12** | 16x16 | Surround/Touch | Freeze/Active |

---

## 4. Results & Data Structure

All results are stored in the `trained_agents/` directory:
- **`metrics.csv`**: Full training progress log (Episode, Steps, Reward, Epsilon, Loss).
- **`evaluation_summary.csv`**: Summary of periodic evaluation phases.
- **`evaluations/`**: Detailed episode logs for each specific evaluation run.
- **`model.pt`**: The **best model** found so far (updated whenever an evaluation beats the previous best).
- **`config.json`**: The frozen configuration used for the run.

---

## 5. Automated Analysis & Visualization

Once experiments are complete, use the visualization script to generate outcome charts:

```bash
python graph.py
```

This script generates plots in the `graphs/` directory, organized as:
- **`mapsize/`**: Compares task difficulty (Surround vs. Touch) for a fixed grid size.
- **`task/`**: Compares scaling (8x8 -> 16x16) for a fixed task type.
- **`stability/`**: Individual loss plots for every experiment to verify convergence.

---
*Created for the RoboticSensing Module.*

---Grade nutze ich noch nen value, der für die execution phase trotzdem random values nutzt. Inwiefern ist das so richtig??? 

- in der präsi nochmal genauer das konzept qmix erkläeren und wie genau die agenten miteinander reden?? 

- global state??? 

- kleiner report 4-8 seiten im märz abgeben! 

- Sturktur der Neuronalen Netzwerken, wie genau mixer netzwerk vs agenten netzwerke ??? 

- probieren zu visualiesern vllt sogar während des trainigs mit web tools

- Future work im paper inkludieren! 

- Präsentation 30 min etwa 

500 vs 200 mal laufen?? 


ANDERE IDEEN: 

- distrubution shift -> Andere umgebung als die auf der ich trainiert habe, wie gut performt der agent dann? 
Wie hoch ist die performanceeinbußung? 

-