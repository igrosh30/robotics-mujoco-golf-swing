# robotics-mujoco-golf-swing
O ponto de entrada da simulação atual é `main.py`

| Ficheiro | Função |
|---|---|
| `main.py` | Simulação do jogador, controlo PD, perturbações e viewer |
| `swing_trajectory.py` | Referência do swing e cinemática inversa dos braços |
| `export_visuals.py` | Exportação dos GIFs e folhas de imagens |
| `planar_kinematics.py` | Cálculos do exemplo anterior de um braço planar de 2 elos |
| `planar_arm_demo.py` | Demonstração desse exemplo planar no MuJoCo |
| `human.xml` | Modelo físico do jogador, taco, bola, joints e motores |

Com as dependências instaladas, executar na pasta do projeto

```bash
python main.py
python main.py --headless
python export_visuals.py
```

Os comandos de instalação estão em `RUN_LAB.txt` e a explicação do modelo em
`UNDERSTAND_THE_PROJECT.txt`

O exemplo planar depende de `models/arm2.xml`, que não está atualmente na pasta
O módulo `planar_kinematics.py` pode ser executado sozinho, sem esse XML

O texto abaixo descreve a etapa planar anterior e conserva as instruções históricas

# robotics-mujoco-golf-swing

A 2-link planar model of a golf swing in **MuJoCo**, for the IST MEEC Robotics
course (Lab 1, Prof. João Silva Sequeira). The player + club are modelled as a
two-joint arm; we derive its kinematics by hand, solve inverse kinematics to
place the club head on a target path, and drive it in MuJoCo.

---

## The idea in one picture

```
        shoulder ● θ1          two revolute joints (θ1, θ2) in one vertical plane
                 |             player's arm  = link 1  (length L1)
                 | L1          club          = link 2  (length L2)
          elbow  ● θ2          club head     = the tip that hits the ball
                 |
                 | L2
                 ● club head  →  ● ball
```

- **Forward kinematics**: given the angles (θ1, θ2), where is the club head?
- **Inverse kinematics**: given where we want the club head (x, y), what angles?

MuJoCo does forward kinematics and dynamics for us. The **inverse kinematics is
ours to compute** (in `planar_kinematics.py`) and feed to the simulator.

---

## What's in the repo

```
.
├── planar_kinematics.py        pure math: forward + inverse kinematics, trajectories.
│                        NO MuJoCo import — testable on its own with numpy.
├── planar_arm_demo.py           glue: loads the model, solves IK, writes the joint
│                        angles, and checks the tip lands on the target.
├── models/
│   ├── arm2.xml         the 2-link arm (Stage 1: gravity OFF, kinematics only)
│   ├── pendulum.xml     warm-up single pendulum
│   ├── car.xml          MuJoCo reference example (study material)
│   ├── starting.xml     scratch / experimentation
│   ├── golf_club.stl    club mesh
│   └── golf_club_head.stl collision mesh do taco
├── requirements.txt     exact package versions — the environment recipe
├── .gitignore           keeps .venv/ and junk out of git
└── README.md            this file
```

**Design choice — why two Python files, not one:** `planar_kinematics.py` is pure
geometry and imports only numpy. `planar_arm_demo.py` is the only file that touches
MuJoCo and the XML. So the math can be tested, plotted and debugged with no
simulator, and the simulator can be swapped without touching the math.

---

## Setup (do this once after cloning)

You need **Python 3.10+**. Everyone builds their own virtual environment from
`requirements.txt` — we do NOT share the installed environment (`.venv/` is
gitignored because it's large and machine-specific).

```bash
git clone https://github.com/igrosh30/robotics-mujoco-golf-swing.git
cd robotics-mujoco-golf-swing

python -m venv .venv                 # create the environment
source .venv/bin/activate            # macOS / Linux
# .venv\Scripts\activate             # Windows

pip install -r requirements.txt      # install the exact same versions
```

Your prompt should now show `(.venv)`.

---

## Running it

```bash
# 1. The math alone — solves IK for several targets, checks the round trip.
#    If IK is correct, every "roundtrip error" is ~1e-16.
python planar_kinematics.py

# 2. Math + MuJoCo — solves IK, writes the angles, asks MuJoCo where the tip
#    landed, compares to the target. Worst error should be ~1e-16.
python planar_arm_demo.py

# 3. The viewer — animates the arm along an arc.
#    macOS needs mjpython (ships with the mujoco package), NOT python:
mjpython planar_arm_demo.py --view
```

`planar_arm_demo.py --branch -1` switches to the mirrored elbow configuration.

**Plane note:** the hand derivation is in the x–y plane; MuJoCo is z-up, so the
arm lives in the **x–z** plane. The mapping is `derivation y → MuJoCo z`. An IK
target `(x, y)` becomes the world point `(x, 0, y)`.

---

## Working on it together

```bash
git pull            # get the latest before you start
# ... make your changes ...
git add -A
git commit -m "short description of what you did"
git push            # share it
```

If `git push` is rejected with "behind", someone else pushed first —
run `git pull` to merge their work in, then push again.

**Never commit `.venv/`** — it's gitignored for a reason (huge, and the binaries
only work on the machine that built them). If you add a new package, run
`pip freeze > requirements.txt` and commit that so everyone stays in sync.

---

## Where the project is going

- **Stage 1 (done):** kinematics — arm places its tip on any target, gravity off.
- **Stage 2 (next):** dynamics — gravity on, apply torques/forces to drive a real
  swing, add the ball and the impact.
- **Stage 3:** validation + report — check the physics, plot results, write it up.

## The three work streams

- **Modelling** — the arm/club model and its parameters (XML, masses, lengths).
- **Simulation** — the Python that drives MuJoCo and runs the disturbance tests.
- **Validation + report** — physics sanity checks, plots, and the writeup.
