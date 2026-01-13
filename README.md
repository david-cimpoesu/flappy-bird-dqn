# Raport Proiect: Agent Flappy Bird cu Deep Q-Learning

## 1. Introducere

Acest proiect implementează un agent bazat pe învățare prin întărire (Deep Q-Learning) capabil să joace Flappy Bird folosind exclusiv input vizual (pixeli). Obiectivul principal este maximizarea scorului prin navigarea precisă printre obstacole, învățând direct din experiența vizuală.

---

## 2. Arhitectura Rețelei Neuronale (FlappyCNN)

Am utilizat o arhitectură de tip Convolutional Neural Network (CNN), optimizată pentru procesarea datelor spațiale din imagini:

**Input:**  
Un "stack" de 4 cadre consecutive (84x84 pixeli) pentru a oferi agentului informații temporale despre viteza și direcția păsării.

**Straturi Convoluționale:**
- Conv1: 32 filtre, kernel 8x8, stride 4.
- Conv2: 64 filtre, kernel 4x4, stride 2.
- Conv3: 64 filtre, kernel 3x3, stride 1.

**Straturi Fully Connected (FC):**  
Un strat ascuns de 512 unități cu activare ReLU, urmat de un strat de ieșire de 2 unități (corespunzătoare acțiunilor: nu sări și sari).

---

## 3. Implementarea Algoritmului Q-Learning

Algoritmul se bazează pe ecuația lui Bellman:

$$
Q(s, a) = r + \gamma \max_{a'} Q(s', a')
$$

**Componente Cheie:**

**Experience Replay:**  
Utilizăm un buffer circular (ReplayBuffer) de 100.000 de tranziții pentru a elimina corelațiile temporale din date și a stabiliza procesul de antrenare.

**Target Network:**  
Folosim o rețea separată pentru calculul valorilor Q-țintă, actualizată la fiecare 1000 de cadre, pentru a evita oscilațiile în timpul învățării.

**Strategy Pattern:**  
Selecția acțiunilor urmează o strategie $\epsilon$-greedy, care balansează explorarea (mișcări aleatoare) și exploatarea (deciziile optime ale modelului).

---

## 4. Preprocesarea Datelor

Pentru a optimiza performanța și a respecta cerințele de punctaj maxim, input-ul vizual este procesat astfel:

- **Grayscale:** Reducerea complexității prin conversia cadrelor RGB la un singur canal de intensitate.
- **Resize:** Redimensionarea imaginilor la 84x84 pixeli.
- **Normalizare:** Valorile pixelilor sunt scalate în intervalul $[0, 1]$.
- **Frame Skipping:** Agentul ia o decizie la fiecare 4 cadre, accelerând astfel procesul de învățare și execuție.

---

## 5. Experimente și Hiperparametri

### Detalii Hardware

- **GPU:** NVIDIA GeForce RTX 4060.
- **Timp de antrenare:** Aproximativ 30 de minute pentru ca agentul să înceapă să treacă constant de primul obstacol. 
- Antrenarea direct din pixeli este costisitoare computațional, motiv pentru care utilizarea GPU a fost importantă pentru rularea eficientă a experimentelor.

### Configurația Hiperparametrilor (Final)

| Parametru | Valoare |
|----------|----------|
| BATCH_SIZE | 258 |
| LEARNING_RATE | $10^{-4}$ |
| GAMMA (Discount Factor) | 0.99 |
| EPSILON_DECAY | 300.000 cadre |
| LEARNING_STARTS | 10.000 cadre |

### Istoricul Experimentelor

**Încercarea Inițială:**

- Configurație: Recompensă moarte: -1.0; Epsilon Decay: 10.000 (prea rapid).
- Rezultat: Agentul nu a reușit să învețe corect riscul de coliziune, obținând un reward mediu de 11.3 și eșuând să navigheze printre pipe-uri. Acest rezultat a indicat o sensibilitate ridicată la alegerea strategiei de explorare și a funcției de recompensare.

**Varianta Finală (Curentă):**

- Ajustare Reward Shaping: Am setat penalizarea de moarte la -1.0, un bonus semnificativ pentru trecerea pipe-urilor (+10.0) și un bonus mic de supraviețuire (0.1).
- Explorare: Creșterea EPSILON_DECAY la 300.000 a permis o explorare mult mai detaliată a spațiului stărilor.  
- O decădere mai lentă a explorării a permis evitarea convergenței premature către politici suboptime.

---

## 6. Rezultate Obținute

În faza de testare, agentul a demonstrat o performanță modestă, atingând scoruri de până la 70.3, ceea ce corespunde trecerii a aproximativ 3-5 obstacole consecutive.  
Acest rezultat sugerează că agentul a învățat o politică coerentă bazată exclusiv pe informație vizuală.

**Exemplu Log Testare:**

- Test Episode 1: 29.1  
- Test Episode 4: 42.6  
- Test Episode 5: 70.3  
