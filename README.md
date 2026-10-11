# robotics-mujoco-golf-swing

Simulação de um swing de golfe em **MuJoCo + Python** — jogador (parte superior
do corpo), taco e bola. IST MEEC, Robótica Lab 1 (Prof. João Silva Sequeira).

O jogador tem 11 graus de liberdade atuados (tronco, ombros, cotovelos, pulso
direito). O movimento é definido por uma trajetória de referência calculada por
cinemática inversa e seguida por um controlador PD com compensação dinâmica —
os motores aplicam **torques**, nunca ângulos diretos, por isso a física é real.

## Ficheiros

| Ficheiro | Função |
|---|---|
| `human.xml` | Modelo físico: jogador, taco, bola, joints e motores |
| `swing_trajectory.py` | Referência do swing e cinemática inversa dos braços |
| `main.py` | Simulação, controlo PD, perturbações e viewer |
| `disturbance_sweep.py` | Varrimento de perturbações em cada DOF (Task 2) |
| `analysis.py` | Varrimento do tempo de estabelecimento do PD |
| `export_visuals.py` | Exportação dos GIFs e folhas de imagens |
| `models/golf_club.stl` | Malha visual do taco |
| `models/golf_club_head.stl` | Malha de colisão da cabeça do taco |
| `requirements.txt` | Versões exatas dos pacotes |

## Instalação (uma vez)

Requer **Python 3.10+**. Cada pessoa cria o seu ambiente a partir de
`requirements.txt` (`.venv/` está no `.gitignore`).

```bash
git clone https://github.com/igrosh30/robotics-mujoco-golf-swing.git
cd robotics-mujoco-golf-swing

python -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

## Como correr

No macOS o viewer precisa de `mjpython` (vem com o pacote mujoco), não `python`.

```bash
# Ver o swing na janela interativa
mjpython main.py
#   R  -> reinicia a simulação
#   P  -> reduz os torques a 10% (mostra que o movimento vem da dinâmica)

# Correr sem janela e imprimir as métricas (impacto, velocidade, erro, etc.)
python main.py --headless

# Teste de perturbações (Task 2): baseline + cada DOF perturbado.
# Guarda disturbance_results_<A>Nm.json e imprime a tabela comparativa.
python disturbance_sweep.py --amplitude 30
python disturbance_sweep.py --amplitude 60
python disturbance_sweep.py --amplitude 300

# Varrimento do tempo de estabelecimento do PD (gera GIFs + summary.json)
python analysis.py --all

# Exportar GIFs e folhas de imagens (front/side) para exports/
python export_visuals.py
```

## Testes realizados (Task 2)

- **Swing nominal** — baseline sem perturbação: impacto ≈ 2,25 s, velocidade da
  cabeça do taco ≈ 9,44 m/s, erro máximo de seguimento 1,03°, 0% de saturação.
- **Perturbação por DOF** — torque externo sinusoidal adicionado a
  `qfrc_applied` (não a `ctrl`), com janela `sin²` suave, aplicado a cada uma das
  11 joints a 30, 60 e 300 N·m. Resultados em `disturbance_results_*.json`.
- **Detuning do PD** — tempo de estabelecimento de 0,3 a 20 s (`analysis.py`).
- **Verificação de torque reduzido** — tecla `P` a 10%: o corpo não consegue
  seguir a referência, confirmando que o movimento vem da dinâmica.

## Trabalho em equipa

```bash
git pull
# ... alterações ...
git add -A && git commit -m "descrição curta"
git push
```

Nunca fazer commit de `.venv/`. Se adicionares um pacote,
`pip freeze > requirements.txt` e faz commit.

## Documentação

`RUN_LAB.txt` — comandos de instalação · `UNDERSTAND_THE_PROJECT.txt` —
explicação do modelo e das equações · relatório em `MuJoCo_lab1.pdf`.
