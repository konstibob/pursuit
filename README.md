# ReadMe: Multi-Agent Surround Experiments

## 1. Einleitung
Alle Experimente verwenden ein Q-Learning-/Deep RL-basiertes Training. Die Experimente variieren hauptsächlich die **Mapgröße**, die **Surround-Option**, **Freeze Evaders**, und **n_catch**. Andere Parameter wie `n_pursuers` und `n_evaders` werden automatisch an die Mapgröße angepasst.

---

## 2. Parameter Erklärung

| Parameter        | Beschreibung                                                                                  |
| ---------------- | -------------------------------------------------------------------------------------------- |
| **Mapgröße**       | Größe des Spielfelds in X × Y (z. B. 8×8, 12×12, 16×16). Beeinflusst die Anzahl der Agenten. |
| **Surround**       | Wenn `True`, müssen die Verfolger die Evader umzingeln, um sie zu fangen.                   |
| **Freeze Evaders** | Wenn `True`, bewegen sich die Evaders nicht. Wenn `False`, bewegen sie sich zufällig.      |
| **n_pursuers**     | Anzahl der Verfolger (Pursuers). Wird automatisch nach Mapgröße angepasst.                  |
| **n_evaders**      | Anzahl der zu fassenden Evaders. Wird automatisch nach Mapgröße angepasst.                   |
| **n_catch**        | Anzahl an Verfolgern, die nötig sind, um einen Evader zu fangen. (z. B. 2 oder 4)          |
| **Mapgröße** | Größe des Grids in X- und Y-Richtung.                                                     |
---

## 3. Experimente

| Experiment # | Mapgröße | Surround | Freeze Evaders | n_pursuers | n_evaders | n_catch |
| ------------ | -------- | -------- | -------------- | ---------- | --------- | ------- |
| 1            | 8×8      | True     | True           | 4          | 15        | 2       |
| 2            | 8×8      | True     | True           | 4          | 15        | 4       |
| 3            | 8×8      | True     | False          | 4          | 15        | 2       |
| 4            | 8×8      | True     | False          | 4          | 15        | 4       |
| 5            | 8×8      | False    | True           | 4          | 15        | 2       |
| 6            | 8×8      | False    | True           | 4          | 15        | 4       |
| 7            | 8×8      | False    | False          | 4          | 15        | 2       |
| 8            | 8×8      | False    | False          | 4          | 15        | 4       |
| 9            | 12×12    | True     | True           | 6          | 22        | 2       |
| 10           | 12×12    | True     | True           | 6          | 22        | 4       |
| 11           | 12×12    | True     | False          | 6          | 22        | 2       |
| 12           | 12×12    | True     | False          | 6          | 22        | 4       |
| 13           | 12×12    | False    | True           | 6          | 22        | 2       |
| 14           | 12×12    | False    | True           | 6          | 22        | 4       |
| 15           | 12×12    | False    | False          | 6          | 22        | 2       |
| 16           | 12×12    | False    | False          | 6          | 22        | 4       |
| 17           | 16×16    | True     | True           | 8          | 30        | 2       |
| 18           | 16×16    | True     | True           | 8          | 30        | 4       |
| 19           | 16×16    | True     | False          | 8          | 30        | 2       |
| 20           | 16×16    | True     | False          | 8          | 30        | 4       |
| 21           | 16×16    | False    | True           | 8          | 30        | 2       |
| 22           | 16×16    | False    | True           | 8          | 30        | 4       |
| 23           | 16×16    | False    | False          | 8          | 30        | 2       |
| 24           | 16×16    | False    | False          | 8          | 30        | 4       |


**Hinweise zu Experimenten:**
- `n_pursuers` und `n_evaders` passen sich automatisch an die Mapgröße an, keine extra Experimente nötig.  
- Jedes Experiment wird für x viele Episoden durchgeführt, um Lerntrends sichtbar zu machen.  
- `n_catch` wird in jedem Szenario entweder 2 oder 4 sein, um die Effektivität von Surrounding unter verschiedenen Bedingungen zu vergleichen.  

---

## 4. Ideen für Grafiken

### Reward über Alle Experimente
- **Lineplot:**  
- **X-Achse:** Episode  
- **Y-Achse:** Reward pro Episode  
- **Linien:** Verschiedene Experimente (z. B. unterschiedliche Mapgrößen, Surround True/False, Freeze True/False)  
- **Zweck:** Zeigt, wie schnell die Agenten lernen und ob Surround/Taktik überhaupt effektiv ist.  
- **Tip:** Mehrere Linien pro Plot, Legende mit Experiment-Nummern oder Parameter-Kombinationen.
---

### Vergleich von n_catch = 2 vs n_catch = 4
- **Lineplot:**  
  - **X-Achse:** Episode  
  - **Y-Achse:** Reward gemittelt über alle experimente? 
- **Zweck:** Zeigt, wie viel einfacher oder schwieriger Surround wird, wenn mehr Agenten nötig sind, um einen Evader zu fangen.
---

### Vergleich Freeze Evaders True vs False
- **Lineplot:**  
  - **X-Achse:** Episode 
  - **Y-Achse:** Reward   
- **Zweck:** Zeigt, wie die Bewegung der Evaders das Lernen beeinflusst – lernen Agenten schneller, wenn Evaders stillstehen?
