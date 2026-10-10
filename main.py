import argparse
import csv
import json
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from scipy.optimize import brentq


from swing_trajectory import (SwingTrajectory, JOINTS, SETTLE_TIME, BACKSWING_TIME,
                              DOWNSWING_TIME, FOLLOW_TIME, IMPACT_TIME, FINISH_TIME)

FPS = 60
# tempo de acomodação pretendido no modelo simplificado de cada joint, em segundos
# escolhemos 0,30 s por ser inferior aos 0,75 s do downswing, não é uma medição do jogador, embora tenhamos feito testes com 5s 10s,também correram bem
# mas com 15s ou mais o jogador não consegue adptar-se a tempo, então bate com o taco no chão e os braços ficam out of control 
PD_SETTLING_TIME = 0.30


def calculate_pd_gains(model, trajectory, settling_time=PD_SETTLING_TIME):
    """Calcula Kp e Kd na ordem de JOINTS para uma resposta nominal criticamente amortecida"""
    if not np.isfinite(settling_time) or settling_time <= 0:
        raise ValueError('settling_time must be finite and positive')

    # começamos por estimar a resistência de cada joint a acelerar
    # na rotação temos torque = J * aceleração angular, o equivalente a F = m*a
    # J depende da distribuição da massa em relação ao eixo, não apenas da massa do segmento
    qids = np.array([model.joint(name).qposadr[0] for name in JOINTS])
    vids = np.array([model.joint(name).dofadr[0] for name in JOINTS])
    reference_data = mujoco.MjData(model)
    mass = np.empty((model.nv, model.nv))
    joint_inertia = np.zeros(len(JOINTS))
    for t in np.linspace(0., FINISH_TIME, 201):
        # usamos um estado auxiliar para este cálculo não mexer no jogador da simulação
        reference_data.qpos[qids] = trajectory.sample(t)[0]
        mujoco.mj_forward(model, reference_data)
        mujoco.mj_fullM(model, reference_data, mass)
        # Mii é a inércia desta coordenada com as outras joints imobilizadas, em kg m²
        # inclui os corpos a jusante e a armature, por isso o taco pesa no cálculo do braço direito
        # usamos o maior valor entre 201 amostras da referência para obter ganhos fixos
        # é uma aproximação diagonal, não a inércia efetiva exata com a pega fechada
        joint_inertia = np.maximum(joint_inertia, mass.diagonal()[vids])

    # definimos e = q_ref-q e e_dot = v_ref-v
    # P funciona como uma mola virtual que puxa para a referência
    # D corrige a diferença de velocidade, ajudando a evitar oscilações
    # com compensação ideal e antecipação da aceleração, uma joint isolada dá
    # J*e_ddot + Kd*e_dot + Kp*e = 0
    # dividimos por J e comparamos com e_ddot + 2*zeta*wn*e_dot + wn²*e = 0
    # daí saem Kp = J*wn² e Kd = 2*zeta*J*wn
    # wn define a rapidez da resposta e zeta o amortecimento relativo
    # escolhemos zeta=1 para amortecimento crítico, sem oscilar no degrau nominal

    # queremos que o erro do degrau desça até 2% do valor inicial no tempo Ts
    # para zeta=1 e derivada inicial do erro nula, e(t)/e(0) = (1+wn*t)*exp(-wn*t)
    # resolvemos (1+x)*exp(-x)=0,02 com x=wn*Ts e obtemos x perto de 5,834
    # com Ts=0,30 s dá wn perto de 19,446 rad/s, atenção que não são Hz
    # a aproximação habitual Ts=4/(zeta*wn) não é exata para esta resposta crítica
    scaled_time = brentq(lambda x: (1.+x)*np.exp(-x)-.02, 0., 20.)
    natural_frequency = scaled_time / settling_time
    kp = joint_inertia * natural_frequency**2       # N m/rad
    kd = 2. * joint_inertia * natural_frequency     # N m s/rad, zeta fica fixo em 1

    # não descontamos o amortecimento passivo em Kd porque step() compensa qfrc_passive
    # as fórmulas dimensionam uma aproximação contínua de joints isoladas
    # o acoplamento, a pega, os contactos, a saturação e o passo temporal exigem ensaios
    # escolher Ts=0,30 s não garante esse tempo de acomodação no jogador completo
    # referência: https://modernrobotics.northwestern.edu/nu-gm-book-resource/11-4-motion-control-with-torque-or-force-inputs-part-1-of-3/
    return kp, kd


class SwingSimulation:
    def __init__(self, *, timestep=None, disturbance_joint=None,
                 disturbance_amplitude=3., disturbance_frequency=8.,
                 settling_time=PD_SETTLING_TIME):
        self.model = mujoco.MjModel.from_xml_path(str(Path(__file__).with_name('human.xml')))
        if timestep is not None:
            self.model.opt.timestep = timestep
        self.data = mujoco.MjData(self.model)
        self.trajectory = SwingTrajectory(self.model) # prepara a referência de ângulos, velocidades e acelerações
        # temos 11 coordenadas de rotação no jogador e 6 DOF livres na bola
        # qpos tem 18 valores e qvel tem 17 porque a bola usa um quaternion para a orientação
        # qposadr e dofadr evitam confundir os índices de posição e de velocidade
        self.qpos_ids = np.array([self.model.joint(n).qposadr[0] for n in JOINTS])
        self.qvel_ids = np.array([self.model.joint(n).dofadr[0] for n in JOINTS])
        self.motor_ids = np.array([self.model.actuator(f'{n}_motor').id for n in JOINTS])
        # calculamos uma vez para este modelo e referência, os ganhos ficam fixos durante o swing
        # estes ganhos substituem os antigos valores manuais, não são a justificação desses valores
        self.kp, self.kd = calculate_pd_gains(
            self.model, self.trajectory, settling_time=settling_time
        )
        self.club_id = self.model.geom('club_head_collision').id
        self.ball_id = self.model.geom('golf_ball_geom').id
        self.ball_dof = self.model.joint('golf_ball_free').dofadr[0]
        self.inertia = np.empty((self.model.nv, self.model.nv))
        if disturbance_joint is not None and disturbance_joint not in JOINTS:
            raise ValueError(f'Unknown upper-body joint: {disturbance_joint}')
        self.disturbance_joint = disturbance_joint
        self.disturbance_amplitude = disturbance_amplitude
        self.disturbance_frequency = disturbance_frequency
        self.pd_disabled = False # a tecla P alterna entre comando completo e comando reduzido
        self.reset()
        

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        # só impomos os ângulos diretamente no reset para colocar o jogador na pose inicial
        # durante o swing o movimento resulta dos torques e da dinâmica
        self.data.qpos[self.qpos_ids] = self.trajectory.sample(0.)[0]
        mujoco.mj_forward(self.model, self.data)
        self.ball_start = self.data.body('golf_ball').xpos.copy()
        self.impact_time = None
        self.launch_velocity = None
        self.separation_velocity = None
        self.club_speed_at_impact = None
        self.head_was_touching_ball = False
        self.max_ball_height = float(self.data.body('golf_ball').xpos[2])
        self.max_ball_speed = 0.
        self.max_grip_error = 0.
        self.max_tracking_error = 0.
        self.saturated_steps = 0
        self.steps = 0
        self.trace = []

    def step(self):
        # atualizamos a cinemática e os termos dinâmicos antes de calcular o controlo
        # esta divisão em step1 e step2 usa o integrador Euler escolhido por omissão
        mujoco.mj_step1(self.model, self.data)
        t = self.data.time
        # ângulo em rad, velocidade em rad/s e aceleração em rad/s²
        target, target_speed, target_acceleration = self.trajectory.sample(t)
        mujoco.mj_fullM(self.model, self.data, self.inertia)
        # o cálculo dos ganhos usa inércias diagonais, o controlo usa o bloco atuado 11x11
        # os termos fora da diagonal representam o acoplamento entre as joints
        mass = self.inertia[np.ix_(self.qvel_ids, self.qvel_ids)]
        dt = self.model.opt.timestep
        # P corrige o erro de ângulo e D o erro de velocidade, não temos termo integral
        # mass @ target_acceleration antecipa a aceleração pedida pela trajetória
        # os termos com dt tratam o feedback implicitamente e ajudam na estabilidade numérica
        # quando dt tende para zero aproximamo-nos de M*a_cmd = Kp*e + Kd*e_dot + M*a_ref
        acceleration = np.linalg.solve(mass + np.diag(dt*self.kd + dt*dt*self.kp),
            self.kp*(target-self.data.qpos[self.qpos_ids])
            + (self.kd+dt*self.kp)*(target_speed-self.data.qvel[self.qvel_ids])
            + mass@target_acceleration)
        # bias compensa a gravidade e os efeitos de Coriolis e centrífugos
        # subtraímos as forças passivas porque o MuJoCo também as aplica na dinâmica
        # por isso aumentar damping não simula automaticamente atrito desconhecido pelo controlo
        # as forças da pega e dos contactos continuam a ser resolvidas pelo simulador
        torque = (mass@acceleration + self.data.qfrc_bias[self.qvel_ids]
                  - self.data.qfrc_passive[self.qvel_ids])
        limits = self.model.actuator_ctrlrange[self.motor_ids]

        pd_constant = 0
        # o botão P reduz o comando total para 25%, incluindo a compensação da gravidade
        # não desliga apenas P e D, por isso não é um ensaio puro de PD ligado ou desligado
        if self.pd_disabled:
            pd_constant = 0.25
        else:
            pd_constant = 1
        
        # enviamos um valor real para cada motor através de data.ctrl
        # como gear=1, ctrl=10 corresponde a 10 N m nessa joint, não a graus ou PWM
        # o passo de 0,0005 s dá 2000 atualizações por segundo simulado, FPS só afeta a janela
        # o clip limita os torques a ±250 no tronco, ±180 nos ombros e ±90 nos cotovelos e pulso
        # se o motor saturar, aumentar Kp não lhe dá mais torque disponível
        self.data.ctrl[self.motor_ids] = np.clip(torque, limits[:, 0], limits[:, 1]) * pd_constant
        # contamos passos em que pelo menos um pedido excede o limite, antes da redução pela tecla P
        self.saturated_steps += int(np.any((torque < limits[:, 0]) | (torque > limits[:, 1])))
        self.steps += 1

        # este ensaio aplica um torque externo só na joint escolhida
        # a amplitude A está em N m e a frequência f em Hz, por omissão A=3 e f=8
        # sin(pi*u)² liga e desliga suavemente a oscilação entre 0,50 e 3,10 s
        # A limita a amplitude, o máximo amostrado pode ser ligeiramente inferior
        # sem disturbance_joint a perturbação fica desligada mesmo que A seja diferente de zero
        # isto testa rejeição de perturbações, não altera o atrito nem permite aos pés escorregar
        self.data.qfrc_applied[:] = 0.
        if self.disturbance_joint is not None and SETTLE_TIME <= t <= FINISH_TIME:
            u = (t - SETTLE_TIME) / (FINISH_TIME - SETTLE_TIME)
            disturbance = (self.disturbance_amplitude * np.sin(np.pi*u)**2
                           * np.sin(2*np.pi*self.disturbance_frequency*(t-SETTLE_TIME)))
            dof = self.model.joint(self.disturbance_joint).dofadr[0]
            self.data.qfrc_applied[dof] = disturbance

        # verificamos o contacto no instante t antes de integrar o passo seguinte
        # IMPACT_TIME é o instante planeado, o contacto real depende da dinâmica
        contact = any({c.geom1, c.geom2} == {self.club_id, self.ball_id}
                      for c in self.data.contact)
        if contact and self.impact_time is None:
            self.impact_time = float(t)
            jac = np.zeros((3,self.model.nv))
            mujoco.mj_jacSite(self.model,self.data,jac,None,self.model.site('club_head').id)
            self.club_speed_at_impact = float(np.linalg.norm(jac@self.data.qvel))
        if self.head_was_touching_ball and not contact and self.separation_velocity is None:
            self.separation_velocity = self.data.qvel[self.ball_dof:self.ball_dof+3].copy()
        self.head_was_touching_ball = contact
        mujoco.mj_step2(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

        velocity = self.data.qvel[self.ball_dof:self.ball_dof+3].copy()
        speed = float(np.linalg.norm(velocity))
        self.max_ball_height = max(self.max_ball_height,float(self.data.body('golf_ball').xpos[2]))
        if speed > self.max_ball_speed:
            self.max_ball_speed = speed
            if self.impact_time is not None:
                self.launch_velocity = velocity
        # distância entre as duas pegas, mede o erro da restrição e não deslizamento da mão
        grip_error = float(np.linalg.norm(self.data.site('shared_grip').xpos
                                         - self.data.site('left_grip').xpos))
        self.max_grip_error = max(self.max_grip_error, grip_error)
        target, _, _ = self.trajectory.sample(self.data.time)
        error = self.data.qpos[self.qpos_ids] - target
        if self.data.time >= SETTLE_TIME:
            # maior erro absoluto entre todas as joints depois da preparação, não é erro médio
            self.max_tracking_error = max(self.max_tracking_error, float(np.max(np.abs(error))))
        self.trace.append(np.r_[self.data.time, self.data.qpos[self.qpos_ids], target,
                                self.data.ctrl[self.motor_ids],
                                self.data.site('club_head').xpos,
                                self.data.body('golf_ball').xpos, velocity, grip_error])

    def run(self, duration=3.5):
        for _ in range(int(np.ceil((duration-self.data.time)/self.model.opt.timestep))):
            self.step()
        return self.metrics()

    def metrics(self):
        # comparar erro, saturação, pega e resultado do impacto entre os ensaios
        # zero warnings por si só não prova robustez
        # ball_displacement_m é o deslocamento horizontal até ao instante final, não até a bola parar
        ball = self.data.body('golf_ball').xpos
        trace = np.asarray(self.trace)
        return {
            'duration_s': float(self.data.time),
            'impact_time_s': self.impact_time,
            'club_head_speed_at_impact_m_s': self.club_speed_at_impact,
            'ball_velocity_after_impact_m_s': None if self.separation_velocity is None else self.separation_velocity.tolist(),
            'max_ball_height_m': self.max_ball_height,
            'peak_ball_speed_m_s': self.max_ball_speed,
            'ball_displacement_m': float(np.linalg.norm(ball[:2]-self.ball_start[:2])),
            'ball_final_xyz_m': ball.tolist(),
            'ball_velocity_at_peak_m_s': None if self.launch_velocity is None else self.launch_velocity.tolist(),
            'max_grip_error_mm': self.max_grip_error * 1000.,
            'max_tracking_error_deg': float(np.rad2deg(self.max_tracking_error)),
            'saturated_step_fraction': self.saturated_steps/max(1, self.steps),
            'joint_excursion_deg': (np.rad2deg(np.ptp(trace[:, 1:1+len(JOINTS)], axis=0)).tolist()
                                    if len(trace) else [0.]*len(JOINTS)),
            'final_angles_deg': np.rad2deg(self.data.qpos[self.qpos_ids]).tolist(),
            'warnings': {str(mujoco.mjtWarning(i)): int(w.number)
                         for i, w in enumerate(self.data.warning) if w.number},
        }

    def report(self):
        print(json.dumps(self.metrics(), indent=2))

    def save_trace(self, path):
        header = (['time_s'] + [f'{n}_rad' for n in JOINTS]
                  + [f'{n}_target_rad' for n in JOINTS]
                  + [f'{n}_torque_Nm' for n in JOINTS]
                  + ['head_x_m', 'head_y_m', 'head_z_m', 'ball_x_m', 'ball_y_m', 'ball_z_m',
                     'ball_vx_m_s', 'ball_vy_m_s', 'ball_vz_m_s', 'grip_error_m'])
        with Path(path).open('w', newline='', encoding='utf-8') as stream:
            writer = csv.writer(stream)
            writer.writerow(header)
            writer.writerows(self.trace)


def positive_float(value):
    number = float(value)
    if not np.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError('must be a finite positive number')
    return number

def main():
    start = 0
    reset_flag = [False]
    pd_disabled = [False]

    # R reinicia o movimento e P alterna a redução do comando dos motores
    def on_key(key_code):
        if key_code == 82:
            reset_flag[0] = True
            #print("reset!")
        if key_code == 80:
            pd_disabled[0] = True

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--headless', action='store_true', help='run without opening a window')
    parser.add_argument('--duration', type=positive_float, default=3.5, help='simulation seconds')
    parser.add_argument('--slow-motion', type=positive_float, default=1., metavar='FACTOR',
                        help='playback speed divisor, e.g. 3 (physics is unchanged)')
    parser.add_argument('--repeat', action='store_true', help='replay after duration in the viewer')
    parser.add_argument('--csv', type=Path, help='save the measured motion trace')
    parser.add_argument('--disturb-joint', choices=JOINTS, help='apply an external shaky-joint torque')
    parser.add_argument('--disturb-amplitude', type=positive_float, default=3., help='peak disturbance in N m')
    args = parser.parse_args()
    sim = SwingSimulation(disturbance_joint=args.disturb_joint,
                          disturbance_amplitude=args.disturb_amplitude)
    if args.headless:
        sim.run(args.duration)
    else:
        with mujoco.viewer.launch_passive(sim.model, sim.data, key_callback=on_key) as viewer:
            with viewer.lock():
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                viewer.cam.fixedcamid = sim.model.camera('front_fixed').id
            start = time.monotonic()
            while viewer.is_running():
                if reset_flag[0]:
                    with viewer.lock():
                        sim.reset()
                    start = time.monotonic()
                    reset_flag[0] = False
                if pd_disabled[0]:
                    sim.pd_disabled = not sim.pd_disabled
                    pd_disabled[0] = False
                
                frame_start = time.monotonic()
                target_time = min(args.duration, (frame_start - start) / args.slow_motion)
                with viewer.lock():
                    while sim.data.time + 1e-10 < target_time:
                        sim.step()
                viewer.sync()
                if args.repeat and frame_start - start > (args.duration + .8) * args.slow_motion:
                    with viewer.lock():
                        sim.reset()
                    start = time.monotonic()
                time.sleep(max(0., 1. / FPS - (time.monotonic() - frame_start)))
    if args.csv:
        sim.save_trace(args.csv)
    sim.report()


if __name__ == '__main__':
    main()
