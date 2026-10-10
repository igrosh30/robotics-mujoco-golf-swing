"""Swing de golfe com motores, inércia, contactos e bola livre no MuJoCo.

Executa ``python starting.py`` para ver a cena ou ``python starting.py --headless``
para medir o resultado sem abrir a janela.
"""

import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np


SETTLE_TIME = 0.35
BACKSWING_TIME = 0.9
DOWNSWING_TIME = 0.32
FOLLOW_TIME = 0.35
FPS = 60

# Todos os DOFs de rotação têm motor. Os dois free joints (corpo e bola)
# evoluem pela dinâmica e pelas forças de contacto com o chão e o taco.
JOINTS = (
    "torso_turn",
    "right_shoulder1",
    "right_shoulder2",
    "right_wrist_x",
    "right_wrist_y",
    "left_shoulder1",
    "left_shoulder2",
)

# Cada valor descreve o movimento da articulação correspondente ao longo do swing.
# O tronco produz a maior parte da rotação; os braços e pulsos participam com
# ângulos pequenos para manter a cabeça do taco perto do plano horizontal.
AMPLITUDE = np.deg2rad([35.0, 3.0, -2.0, 1.5, -1.5, -3.0, 2.0])
# O pulso inclina o taco para que a cabeça passe junto à bola após o corpo assentar.
REST = np.deg2rad([0.0, 0.0, 0.0, 0.0, 7.0, 0.0, 0.0])
KP = np.array([650.0, 220.0, 220.0, 100.0, 100.0, 220.0, 220.0])
KD = np.array([70.0, 28.0, 28.0, 12.0, 12.0, 28.0, 28.0])


def smoothstep(u):
    """Posição e derivada de uma interpolação com velocidade nula nas pontas."""
    u = np.clip(u, 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u), 6.0 * u * (1.0 - u)


def swing_phase(t):
    """Devolve fase e velocidade da fase: 0 -> -1 -> +1."""
    if t < SETTLE_TIME:
        return 0.0, 0.0

    t -= SETTLE_TIME
    if t < BACKSWING_TIME:
        p, dp = smoothstep(t / BACKSWING_TIME)
        return -p, -dp / BACKSWING_TIME

    t -= BACKSWING_TIME
    if t < DOWNSWING_TIME:
        p, dp = smoothstep(t / DOWNSWING_TIME)
        return -1.0 + 2.0 * p, 2.0 * dp / DOWNSWING_TIME

    t -= DOWNSWING_TIME
    p, dp = smoothstep(t / FOLLOW_TIME)
    return 1.0 + 0.1 * p, 0.1 * dp / FOLLOW_TIME


class SwingSimulation:
    def __init__(self):
        self.model = mujoco.MjModel.from_xml_path(
            str(Path(__file__).with_name("starting.xml"))
        )
        self.data = mujoco.MjData(self.model)
        self.qpos_ids = np.array([self.model.joint(n).qposadr[0] for n in JOINTS])
        self.qvel_ids = np.array([self.model.joint(n).dofadr[0] for n in JOINTS])
        self.motor_ids = np.array([self.model.actuator(f"{n}_motor").id for n in JOINTS])
        self.club_id = self.model.geom("golf_club").id
        self.ball_id = self.model.geom("golf_ball_geom").id
        self.ball_dof = self.model.joint("golf_ball_free").dofadr[0]
        self.ball_start = self.data.body("golf_ball").xpos.copy()
        self.head_reference_z = None
        self.impact_time = None
        self.max_ball_speed = 0.0
        self.max_head_height_error = 0.0
        mujoco.mj_forward(self.model, self.data)
        self.ball_start = self.data.body("golf_ball").xpos.copy()

    def step(self):
        phase, phase_speed = swing_phase(self.data.time)
        target = REST + AMPLITUDE * phase
        target_speed = AMPLITUDE * phase_speed

        # PD em torque: erro de ângulo + erro de velocidade. A compensação
        # de bias/passive retira o peso e as molas; mj_step calcula a aceleração.
        torque = (
            KP * (target - self.data.qpos[self.qpos_ids])
            + KD * (target_speed - self.data.qvel[self.qvel_ids])
            + self.data.qfrc_bias[self.qvel_ids]
            - self.data.qfrc_passive[self.qvel_ids]
        )
        limits = self.model.actuator_ctrlrange[self.motor_ids]
        self.data.ctrl[self.motor_ids] = np.clip(torque, limits[:, 0], limits[:, 1])
        mujoco.mj_step(self.model, self.data)

        if self.impact_time is None:
            for contact in self.data.contact:
                if {contact.geom1, contact.geom2} == {self.club_id, self.ball_id}:
                    self.impact_time = self.data.time
                    break

        speed = np.linalg.norm(self.data.qvel[self.ball_dof:self.ball_dof + 3])
        self.max_ball_speed = max(self.max_ball_speed, float(speed))
        if self.head_reference_z is None and self.data.time >= SETTLE_TIME:
            self.head_reference_z = float(self.data.site("club_head").xpos[2])
        if SETTLE_TIME <= self.data.time <= SETTLE_TIME + BACKSWING_TIME + DOWNSWING_TIME:
            error = abs(float(self.data.site("club_head").xpos[2]) - self.head_reference_z)
            self.max_head_height_error = max(self.max_head_height_error, error)

    def report(self):
        ball = self.data.body("golf_ball").xpos
        displacement = np.linalg.norm(ball[:2] - self.ball_start[:2])
        print(f"Contacto taco-bola: {self.impact_time if self.impact_time is not None else 'não'}")
        print(f"Velocidade máxima da bola: {self.max_ball_speed:.2f} m/s")
        print(f"Deslocamento horizontal da bola: {displacement:.2f} m")
        print(f"Desvio máximo de altura do taco: {self.max_head_height_error:.3f} m")
        print("Ângulos finais (graus):", np.round(np.rad2deg(self.data.qpos[self.qpos_ids]), 1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true", help="simular sem janela")
    parser.add_argument("--duration", type=float, default=3.0, help="segundos de simulação")
    args = parser.parse_args()
    sim = SwingSimulation()

    if args.headless:
        for _ in range(round(args.duration / sim.model.opt.timestep)):
            sim.step()
        sim.report()
        return

    with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
        with viewer.lock():
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            viewer.cam.fixedcamid = sim.model.camera("front_fixed").id
        viewer.sync()
        start = time.monotonic()
        while viewer.is_running():
            frame_start = time.monotonic()
            target_time = frame_start - start
            while sim.data.time < target_time:
                sim.step()
            viewer.sync()
            time.sleep(max(0.0, 1.0 / FPS - (time.monotonic() - frame_start)))
    sim.report()


if __name__ == "__main__":
    main()
