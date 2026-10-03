#!/usr/bin/env python3
"""AR Shadow Fighter: fight a shadow with your body through your webcam.

Run:   python ar_fighter.py [--camera 0] [--complexity 1] [--debug]

Controls:
    D      toggle hitbox / skeleton debug overlay
    R      restart the fight
    C      recalibrate (stand ~6 ft / 2 m back, full body in view)
    F      toggle fullscreen
    Esc/Q  quit

Threads:
    main   : Pygame events, game logic, collisions and rendering, locked to 60 FPS.
    camera : OpenCV capture + MediaPipe Pose (fighter/camera.py). The main loop only
             reads the newest snapshot and never blocks on the camera.
"""
from __future__ import annotations

import argparse
import sys
import time

import pygame

from fighter import config as C
from fighter.camera import CameraThread
from fighter.effects import Effects
from fighter.enemy import EnemyState, ShadowEnemy
from fighter.geometry import intersects
from fighter.player import PlayerTracker

MODE_CALIBRATE, MODE_FIGHT, MODE_OVER = "calibrate", "fight", "over"


class FramePacer:
    """Holds a steady frame rate against absolute deadlines.

    pygame's Clock.tick() oversleeps on macOS (about 53 FPS at a 60 target), so
    sleep until shortly before each deadline (the sleep releases the GIL for the
    camera thread), then spin for the last few milliseconds.
    """
    SPIN_MARGIN = 0.003

    def __init__(self, fps: int):
        self.period = 1.0 / fps
        self.fps = float(fps)
        self._last = time.perf_counter()
        self._deadline = self._last + self.period

    def tick(self) -> float:
        remaining = self._deadline - time.perf_counter()
        if remaining > self.SPIN_MARGIN:
            time.sleep(remaining - self.SPIN_MARGIN)
        while time.perf_counter() < self._deadline:
            pass
        now = time.perf_counter()
        self._deadline += self.period
        if now > self._deadline:  # fell behind (e.g. window drag): don't try to catch up
            self._deadline = now + self.period
        dt, self._last = now - self._last, now
        if dt > 0:
            self.fps += (1.0 / dt - self.fps) * 0.05
        return dt


class Game:
    def __init__(self, camera_index: int, complexity: int, debug: bool):
        # Hand the GIL between threads more often than the default 5 ms so the
        # render thread never waits long behind the camera thread's Python code.
        sys.setswitchinterval(0.0005)
        pygame.init()
        pygame.display.set_caption(C.WINDOW_TITLE)
        self.size = (C.SCREEN_W, C.SCREEN_H)
        try:
            self.screen = pygame.display.set_mode(self.size, pygame.SCALED)
        except pygame.error:  # e.g. no GPU renderer available
            self.screen = pygame.display.set_mode(self.size)
        self.world = pygame.Surface(self.size).convert()
        self.pacer = FramePacer(C.FPS)

        self.font_small = pygame.font.Font(None, 22)
        self.font = pygame.font.Font(None, 32)
        self.font_big = pygame.font.Font(None, 96)

        self.camera = CameraThread(self.size, camera_index, complexity)
        self.camera.start()

        self.player = PlayerTracker(*self.size)
        self.enemy = ShadowEnemy(*self.size)
        self.effects = Effects(self.size)

        self.mode = MODE_CALIBRATE
        self.debug = debug
        self.running = True
        self.background = None
        self.last_frame_id = -1
        self.pose_ms = 0.0
        self.over_timer = 0.0
        self.result = ""
        self.trail_player = float(C.PLAYER_MAX_HP)
        self.trail_enemy = float(C.ENEMY_MAX_HP)

    # --- main loop -----------------------------------------------------------
    def run(self) -> None:
        try:
            while self.running:
                dt = min(self.pacer.tick(), 1.0 / 20.0)
                now = time.perf_counter()
                self._handle_events()
                self._poll_camera(now)
                self._update(dt, now)
                self._draw()
        finally:
            self.camera.stop()
            self.camera.join(timeout=2.0)
            pygame.quit()

    def _handle_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    self.running = False
                elif event.key == pygame.K_d:
                    self.debug = not self.debug
                elif event.key == pygame.K_r and self.player.calibrated:
                    self._begin_fight()
                elif event.key == pygame.K_c:
                    self.player.reset_calibration()
                    self.effects.clear()
                    self.mode = MODE_CALIBRATE
                elif event.key == pygame.K_f:
                    pygame.display.toggle_fullscreen()

    def _poll_camera(self, now: float) -> None:
        """Non-blocking: picks up the newest camera snapshot if there is one."""
        frame = self.camera.latest()
        if frame is None or frame.frame_id == self.last_frame_id:
            return
        self.last_frame_id = frame.frame_id
        self.pose_ms = frame.process_ms
        self.background = pygame.image.frombuffer(frame.image.tobytes(), self.size, "RGB").convert()
        self.player.ingest(frame.landmarks, frame.capture_time, now)

    # --- game logic ----------------------------------------------------------
    def _begin_fight(self) -> None:
        p = self.player
        p.reset_fight()
        com = p.com_x() if p.tracked else self.size[0] * 0.5
        spawn_x = self.size[0] * (0.82 if com < self.size[0] * 0.5 else 0.18)
        self.enemy.reset(ground_y=p.ground_y, height=p.body_height, x=spawn_x)
        self.effects.clear()
        self.trail_player = float(C.PLAYER_MAX_HP)
        self.trail_enemy = float(C.ENEMY_MAX_HP)
        self.mode = MODE_FIGHT
        self.effects.text("FIGHT!", (self.size[0] / 2, self.size[1] * 0.35), (255, 255, 255), 110, 1.2)
        if not p.full_body_visible:
            self.effects.text("Feet not visible: step back to dodge sweeps by jumping",
                              (self.size[0] / 2, self.size[1] * 0.47), (255, 200, 80), 32, 3.0)

    def _update(self, dt: float, now: float) -> None:
        self.player.update(dt, now)
        self.effects.update(dt)
        self.trail_player += (self.player.hp - self.trail_player) * min(1.0, dt * 3)
        self.trail_enemy += (self.enemy.hp - self.trail_enemy) * min(1.0, dt * 3)

        if self.mode == MODE_CALIBRATE:
            if self.player.calibrated:
                self._begin_fight()
            return

        self.enemy.update(dt, self.player)
        if self.mode == MODE_FIGHT:
            if self.player.tracked:
                self._resolve_player_punches()
                self._resolve_enemy_strike()
            if self.enemy.state is EnemyState.DEAD:
                self._end("VICTORY")
            elif self.player.hp <= 0:
                self._end("DEFEAT")
        elif self.mode == MODE_OVER:
            self.over_timer += dt

    def _end(self, result: str) -> None:
        self.mode = MODE_OVER
        self.result = result
        self.over_timer = 0.0

    def _resolve_player_punches(self) -> None:
        enemy, player = self.enemy, self.player
        if enemy.state is EnemyState.DEAD or not player.in_range(C.PLAYER_PUNCH_MIN_DEPTH):
            return
        hurtboxes = enemy.hurtboxes()
        ec = enemy.center()
        for side, fist in player.fists().items():
            if player.punch_cooldown[side] > 0:
                continue
            speed = player.fist_speed(side)
            if speed < C.PUNCH_SPEED_THRESHOLD:
                continue
            vx, vy = player.fist_vel[side]
            if vx * (ec[0] - fist.center[0]) + vy * (ec[1] - fist.center[1]) <= 0:
                continue  # fist moving away from the enemy (pulling back), not a punch
            if not any(intersects(fist, hb) for hb in hurtboxes):
                continue

            power = min(1.0, (speed - C.PUNCH_SPEED_THRESHOLD) / C.PUNCH_SPEED_THRESHOLD)
            damage = round(C.PUNCH_DAMAGE_MIN + (C.PUNCH_DAMAGE_MAX - C.PUNCH_DAMAGE_MIN) * power)
            player.punch_cooldown[side] = C.PUNCH_COOLDOWN
            result = enemy.take_hit(damage)

            self.effects.burst(fist.center, (255, 240, 200), 22, 520)
            self.effects.burst(fist.center, (255, 140, 40), 12, 320)
            self.effects.text(f"-{damage}", (fist.center[0], fist.center[1] - 40), (255, 220, 120), 40)
            self.effects.shake(6 + 6 * power)
            if result == "ko":
                sk = enemy.sk
                self.effects.smoke([sk[k] for k in ("head", "shoulder", "hip", "f_hand", "r_hand",
                                                    "f_elbow", "r_elbow", "f_knee", "r_knee",
                                                    "f_ankle", "r_ankle")], per_point=10)
                self.effects.text("K.O.", (ec[0], ec[1] - enemy.H * 0.5), (255, 255, 255), 120, 1.6)
                self.effects.shake(18)
                return

    def _strike_contact(self, spec, strike):
        """Geometric contact of the swept strike. Forearms are tested before head/torso so blocks always win."""
        player = self.player
        if spec.unblockable:
            if any(intersects(strike, leg) for leg in player.legs()):
                return "legs"
            # Feet are lifted while jumping, so the sweep can pass under them. Still
            # report contact so the dodge registers.
            if player.is_jumping and abs(strike.b[0] - player.com_x()) < 0.8 * player.unit + strike.radius:
                return "legs"
            return None
        if any(intersects(strike, fa) for fa in player.forearms()):
            return "block"
        head = player.head()
        if head is not None and intersects(strike, head):
            return "head"
        torso = player.torso()
        if torso is not None and intersects(strike, torso):
            return "torso"
        return None

    def _resolve_enemy_strike(self) -> None:
        enemy, player = self.enemy, self.player
        strike = enemy.strike_collider()
        if strike is None:
            return
        spec = enemy.attack
        contact = self._strike_contact(spec, strike)
        if contact is None:
            return
        enemy.attack_resolved = True  # each attack resolves at most once
        pos = strike.b

        if not player.in_range(spec.min_depth):
            self.effects.text("OUT OF RANGE", (pos[0], pos[1] - 30), (120, 255, 160), 36)
            return
        if spec.unblockable and player.is_jumping:
            self.effects.text("JUMPED!", (pos[0], pos[1] - 40), (120, 255, 160), 44)
            self.effects.burst(pos, (180, 180, 200), 10, 200)
            return
        if contact == "block":
            chip = spec.damage * spec.block_chip
            player.hp = max(0.0, player.hp - chip)
            enemy.on_blocked()
            self.effects.burst(pos, (120, 200, 255), 18, 380)
            self.effects.text("BLOCK" if chip < 0.5 else f"BLOCK -{chip:.0f}", (pos[0], pos[1] - 30),
                              (140, 210, 255), 40)
            self.effects.shake(4)
            return

        damage = spec.damage * (C.HEAD_HIT_MULTIPLIER if contact == "head" else 1.0)
        player.hp = max(0.0, player.hp - damage)
        self.effects.burst(pos, (255, 60, 60), 20, 420)
        self.effects.text(f"{spec.label}! -{damage:.0f}", (pos[0], pos[1] - 30), (255, 90, 90), 42)
        self.effects.flash((180, 0, 0), 110)
        self.effects.shake(10 + damage * 0.5)

    # --- rendering -----------------------------------------------------------
    def _draw(self) -> None:
        world = self.world
        if self.background is not None:
            world.blit(self.background, (0, 0))
        else:
            world.fill((12, 10, 18))

        if self.mode != MODE_CALIBRATE:
            self.enemy.draw(world)
        self.effects.draw_world(world)
        if self.debug or self.mode == MODE_CALIBRATE:
            self.player.draw_debug(world, self.font_small, colliders=self.debug)
        if self.debug and self.mode != MODE_CALIBRATE:
            self.enemy.draw_debug(world)

        self.screen.fill((0, 0, 0))
        self.screen.blit(world, self.effects.shake_offset())
        self.effects.draw_overlay(self.screen)
        self._draw_hud()
        pygame.display.flip()

    def _text(self, msg, pos, font=None, color=(255, 255, 255), anchor="topleft") -> None:
        font = font or self.font
        shadow = font.render(msg, True, (0, 0, 0))
        surf = font.render(msg, True, color)
        rect = surf.get_rect(**{anchor: pos})
        self.screen.blit(shadow, rect.move(2, 2))
        self.screen.blit(surf, rect)

    def _bar(self, x, y, w, h, value, trail, max_value, color, right_to_left=False) -> None:
        pygame.draw.rect(self.screen, (20, 20, 28), (x - 3, y - 3, w + 6, h + 6), border_radius=4)
        for v, c in ((trail, (240, 240, 240)), (value, color)):
            fw = int(w * max(0.0, v) / max_value)
            rx = x + w - fw if right_to_left else x
            pygame.draw.rect(self.screen, c, (rx, y, fw, h), border_radius=3)

    def _draw_hud(self) -> None:
        w, h = self.size
        cam_err = self.camera.error

        if self.mode != MODE_CALIBRATE:
            bw = int(w * 0.38)
            self._bar(30, 30, bw, 22, self.player.hp, self.trail_player, C.PLAYER_MAX_HP, C.PLAYER_BAR)
            self._bar(w - 30 - bw, 30, bw, 22, self.enemy.hp, self.trail_enemy, C.ENEMY_MAX_HP, C.ENEMY_BAR, True)
            self._text("YOU", (30, 58), self.font)
            self._text("SHADOW", (w - 30, 58), self.font, anchor="topright")
            in_range = self.player.in_range(C.PLAYER_PUNCH_MIN_DEPTH)
            self._text(f"RANGE {'IN' if in_range else 'OUT'}  ({self.player.depth_ratio:.2f})", (30, 86),
                       self.font_small, (120, 255, 160) if in_range else (255, 200, 80))
            if self.player.is_jumping:
                self._text("AIRBORNE", (30, 106), self.font_small, (120, 200, 255))

        stats = (f"FPS {self.pacer.fps:4.0f}   CAM {self.camera.camera_fps:4.1f}   "
                 f"POSE {self.pose_ms:4.1f} ms   [D] debug {'ON' if self.debug else 'off'}   "
                 "[R] restart   [C] calibrate   [F] fullscreen   [Esc] quit")
        self._text(stats, (14, h - 26), self.font_small, (200, 200, 210))

        if cam_err:
            self._text("Camera / pose error", (w / 2, h / 2 - 40), self.font_big, (255, 90, 90), "center")
            self._text(cam_err, (w / 2, h / 2 + 30), self.font, anchor="center")
            return
        if self.background is None:
            self._text("Starting camera...", (w / 2, h / 2), self.font_big, anchor="center")
            return

        if self.mode == MODE_CALIBRATE:
            self._text("CALIBRATION", (w / 2, h * 0.12), self.font_big, anchor="center")
            self._text("Stand about 6 ft (2 m) back with your whole body in view, then hold still.",
                       (w / 2, h * 0.21), self.font, anchor="center")
            if not self.player.tracked:
                self._text("No body detected: step into view", (w / 2, h * 0.27), self.font,
                           (255, 200, 80), "center")
            bw = int(w * 0.4)
            self._bar(int(w / 2 - bw / 2), int(h * 0.31), bw, 16, self.player.calibration_progress,
                      self.player.calibration_progress, 1.0, (120, 255, 160))
        elif self.mode == MODE_FIGHT and not self.player.tracked:
            self._text("STEP INTO VIEW", (w / 2, h * 0.45), self.font_big, (255, 200, 80), "center")
        elif self.mode == MODE_OVER and self.over_timer > 0.8:
            color = (120, 255, 160) if self.result == "VICTORY" else (255, 90, 90)
            self._text(self.result, (w / 2, h * 0.4), self.font_big, color, "center")
            self._text("Press R to fight again, C to recalibrate, Esc to quit", (w / 2, h * 0.5),
                       self.font, anchor="center")


def main() -> None:
    parser = argparse.ArgumentParser(description="AR Shadow Fighter")
    parser.add_argument("--camera", type=int, default=C.CAMERA_INDEX, help="webcam index")
    parser.add_argument("--complexity", type=int, choices=(0, 1, 2), default=C.POSE_MODEL_COMPLEXITY,
                        help="MediaPipe Pose model complexity (0 = fastest)")
    parser.add_argument("--debug", action="store_true", help="start with the hitbox overlay on")
    args = parser.parse_args()
    Game(args.camera, args.complexity, args.debug).run()


if __name__ == "__main__":
    main()
