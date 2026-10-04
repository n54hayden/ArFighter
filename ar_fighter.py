#!/usr/bin/env python3
"""AR Shadow Fighter: fight a shadow with your body through your webcam.

Run:   python ar_fighter.py [--camera 0] [--complexity 1] [--debug]

Flow: menu -> 3 s countdown to get into position -> calibration -> bosses 1-5 in order
(or pick any boss from Select Boss; beating one moves on to the next).

Controls:
    Enter  start (menu) / fight again (after a fight)
    D      toggle hitbox / skeleton debug overlay
    R      restart the current fight
    C      recalibrate (stand ~6 ft / 2 m back, full body in view)
    F      toggle fullscreen
    M      mute / unmute music and sound effects
    Esc    back to the menu (quits from the menu)
    Q      quit

Threads:
    main   : Pygame events, game logic, collisions and rendering, locked to 60 FPS.
    camera : OpenCV capture + MediaPipe Pose (fighter/camera.py). The main loop only
             reads the newest snapshot and never blocks on the camera.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import pygame

from fighter import config as C
from fighter.audio import Music
from fighter.sfx import SoundEffects
from fighter.camera import CameraThread
from fighter.effects import Effects
from fighter.enemy import EnemyState, ShadowEnemy
from fighter.geometry import intersects
from fighter.player import PlayerTracker
from fighter.projectiles import ThrowingStar
from fighter.spells import FireWall, GroundStrike, capsule_hits_rect, circle_hits_rect, make_fire_wall, \
    make_icicles, make_meteor

MODE_MENU, MODE_COUNTDOWN, MODE_CALIBRATE = "menu", "countdown", "calibrate"
MODE_FIGHT, MODE_OVER = "fight", "over"


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
        self._veil = pygame.Surface(self.size).convert()  # black, for dimming behind the menu
        self.pacer = FramePacer(C.FPS)

        self.font_small = pygame.font.Font(None, 22)
        self.font = pygame.font.Font(None, 32)
        self.font_big = pygame.font.Font(None, 96)
        self.font_menu = pygame.font.Font(None, 44)
        self.font_huge = pygame.font.Font(None, 150)

        self.camera = CameraThread(self.size, camera_index, complexity)
        self.camera.start()

        self.player = PlayerTracker(*self.size)
        self.enemy = ShadowEnemy(*self.size)
        self.effects = Effects(self.size)
        here = Path(__file__).resolve().parent
        self.save_path = here / C.SAVE_FILE
        self.checkpoint = self._load_checkpoint()  # furthest boss reached by beating the one before
        self.music = Music(here / C.MUSIC_DIR)
        self.sfx = SoundEffects(here / C.SFX_DIR)

        self.mode = MODE_MENU
        self.debug = debug
        self.running = True
        self.background = None
        self.last_frame_id = -1
        self.pose_ms = 0.0
        self.boss_index = 0
        self.enemy_ready = False      # enemy already placed for the upcoming fight
        self.countdown_t = 0.0
        self.over_timer = 0.0
        self.result = ""
        self.stars = []
        self.hazards = []  # Mage spells: GroundStrike / FireWall
        self.menu_index = 0
        self.menu_page = "main"
        self.menu_rects = []
        self.trail_player = float(C.PLAYER_MAX_HP)
        self.trail_enemy = self.enemy.max_hp

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
            self.music.stop()
            self.camera.stop()
            self.camera.join(timeout=2.0)
            pygame.quit()

    def _handle_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif self.mode == MODE_MENU and event.type == pygame.MOUSEMOTION:
                for i, rect in enumerate(self.menu_rects):
                    if rect.collidepoint(event.pos):
                        self.menu_index = i
            elif self.mode == MODE_MENU and event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                for i, rect in enumerate(self.menu_rects):
                    if rect.collidepoint(event.pos):
                        self._menu_select(i)
            elif event.type == pygame.KEYDOWN:
                self._handle_key(event.key)

    def _handle_key(self, key: int) -> None:
        if key == pygame.K_q:
            self.running = False
        elif key == pygame.K_f:
            pygame.display.toggle_fullscreen()
        elif key == pygame.K_m:
            self.music.toggle_mute()
            self.sfx.muted = self.music.muted if self.music.available else not self.sfx.muted
        elif key == pygame.K_d:
            self.debug = not self.debug
        elif self.mode == MODE_MENU:
            if key == pygame.K_ESCAPE and self.menu_page == "bosses":
                self.menu_page, self.menu_index = "main", 0
            elif key == pygame.K_ESCAPE:
                self.running = False
            elif key in (pygame.K_UP, pygame.K_w):
                self.menu_index = (self.menu_index - 1) % len(self._menu_items())
            elif key in (pygame.K_DOWN, pygame.K_s):
                self.menu_index = (self.menu_index + 1) % len(self._menu_items())
            elif key in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_SPACE):
                self._menu_select(self.menu_index)
        elif key == pygame.K_ESCAPE:
            self._open_menu()
        elif key == pygame.K_c:
            self.player.reset_calibration()
            self._start_countdown()
        elif key == pygame.K_r or (key in (pygame.K_RETURN, pygame.K_KP_ENTER) and self.mode == MODE_OVER):
            if self.mode in (MODE_FIGHT, MODE_OVER):
                if self.result == "CHAMPION" and self.mode == MODE_OVER:
                    self.boss_index = 0  # beat everyone: start over from the first boss
                self._start_countdown()

    def _menu_items(self):
        """(label, action) pairs for the current menu page.

        Main page: 'Continue' appears once you've beaten at least one boss; 'Select Boss'
        opens a page listing every boss so you can fight any of them directly.
        """
        if self.menu_page == "bosses":
            return [(f"{i + 1}.  {b['name'].title()}", f"boss:{i}") for i, b in enumerate(C.BOSSES)] + \
                   [("Back", "back")]
        items = []
        if self.checkpoint > 0:
            nxt = C.BOSSES[self.checkpoint]
            items.append((f"Continue: Round {self.checkpoint + 1} - {nxt['name'].title()}", "continue"))
        items.append(("New Game" if self.checkpoint > 0 else "Start Game", "new"))
        items += [("Select Boss", "select"), ("Quit", "quit")]
        return items

    def _menu_select(self, index: int) -> None:
        action = self._menu_items()[index][1]
        if action == "quit":
            self.running = False
        elif action == "select":
            self.menu_page, self.menu_index = "bosses", 0
        elif action == "back":
            self.menu_page, self.menu_index = "main", 0
        else:
            if action.startswith("boss:"):
                self.boss_index = int(action.split(":")[1])
            else:
                self.boss_index = self.checkpoint if action == "continue" else 0
            self.menu_page = "main"
            self._start_countdown()

    def _load_checkpoint(self) -> int:
        try:
            saved = int(json.loads(self.save_path.read_text())["checkpoint"])
            return max(0, min(len(C.BOSSES) - 1, saved))
        except (OSError, ValueError, KeyError, TypeError):
            return 0

    def _save_checkpoint(self) -> None:
        try:
            self.save_path.write_text(json.dumps({"checkpoint": self.checkpoint}))
        except OSError as exc:
            print(f"[save] could not save progress: {exc}")

    def _poll_camera(self, now: float) -> None:
        """Non-blocking: picks up the newest camera snapshot if there is one."""
        frame = self.camera.latest()
        if frame is None or frame.frame_id == self.last_frame_id:
            return
        self.last_frame_id = frame.frame_id
        self.pose_ms = frame.process_ms
        self.background = pygame.image.frombuffer(frame.image.tobytes(), self.size, "RGB").convert()
        self.player.ingest(frame.landmarks, frame.capture_time, now)

    # --- game flow -----------------------------------------------------------
    @property
    def boss(self) -> dict:
        return C.BOSSES[self.boss_index]

    def _stop_fight(self) -> None:
        self.music.stop()
        self.effects.clear()
        self.stars.clear()
        self.hazards.clear()

    def _open_menu(self) -> None:
        self._stop_fight()
        self.sfx.stop()
        self.mode = MODE_MENU
        self.menu_index = 0
        self.menu_page = "main"

    def _start_countdown(self) -> None:
        """Gives the player a few seconds to get into position before calibrating or fighting."""
        self._stop_fight()
        self.mode = MODE_COUNTDOWN
        self.countdown_t = float(C.COUNTDOWN_SECONDS)
        self.result = ""
        self.enemy_ready = False
        if self.player.calibrated:
            self._place_enemy()  # show the upcoming boss during the countdown

    def _place_enemy(self) -> None:
        p = self.player
        com = p.com_x() if p.tracked else self.size[0] * 0.5
        spawn_x = self.size[0] * (0.82 if com < self.size[0] * 0.5 else 0.18)
        self.enemy.reset(ground_y=p.ground_y, height=p.body_height, x=spawn_x, boss=self.boss)
        p.shield_enabled = bool(self.boss.get("player_shield"))
        self.enemy.facing = 1 if com > spawn_x else -1
        self.enemy_ready = True

    def _begin_fight(self) -> None:
        p = self.player
        p.reset_fight()
        if not self.enemy_ready:
            self._place_enemy()
        self.enemy_ready = False
        self.effects.clear()
        self.stars.clear()
        self.hazards.clear()
        self.trail_player = float(C.PLAYER_MAX_HP)
        self.trail_enemy = self.enemy.max_hp
        self.mode = MODE_FIGHT
        self.sfx.stop()  # cut off a victory cheer still playing from the last fight
        self.sfx.bell_ring()
        self.music.start()
        self.effects.text("FIGHT!", (self.size[0] / 2, self.size[1] * 0.35), (255, 255, 255), 110, 1.2)
        if p.shield_enabled:
            self.effects.text("You have a SHIELD on your left arm: catch his sword on it to parry!",
                              (self.size[0] / 2, self.size[1] * 0.47), (255, 215, 120), 34, 3.5)
        elif not p.full_body_visible:
            self.effects.text("Feet not visible: step back so you can kick and jump over sweeps",
                              (self.size[0] / 2, self.size[1] * 0.47), (255, 200, 80), 32, 3.0)

    def _update(self, dt: float, now: float) -> None:
        self.player.update(dt, now)
        self.effects.update(dt)
        self.trail_player += (self.player.hp - self.trail_player) * min(1.0, dt * 3)
        self.trail_enemy += (self.enemy.hp - self.trail_enemy) * min(1.0, dt * 3)

        if self.mode == MODE_MENU:
            return
        if self.mode == MODE_COUNTDOWN:
            self.countdown_t -= dt
            if self.countdown_t <= 0:
                if self.player.calibrated:
                    self._begin_fight()
                else:
                    self.mode = MODE_CALIBRATE
                    self.player.calibration_enabled = True
            return
        if self.mode == MODE_CALIBRATE:
            if self.player.calibrated:
                self._begin_fight()
            return

        self.enemy.hazards_active = bool(self.hazards)  # one spell at a time: never overlapping hazards
        self.enemy.update(dt, self.player)
        for cast in self.enemy.pop_casts():
            self._spawn_spell(cast)
        for kind, pos in self.enemy.pop_teleport_events():
            self._teleport_effect(kind, pos)
        if self.enemy.pop_flip_started():  # leaping over the player
            self.sfx.throw(0.8)
            self.effects.burst((self.enemy.x, self.enemy.ground_y), (70, 60, 50), 14, 260, size=(3.0, 6.0))
        if self.enemy.pop_landed():  # boss hit the floor after a knockdown
            self.effects.shake(12)
            self.effects.burst((self.enemy.x - self.enemy.facing * self.enemy.H * 0.4, self.enemy.ground_y),
                               (60, 50, 70), 24, 300, size=(3.0, 7.0), gravity=600)
            self.sfx.punch(0.6)
        self._update_stars(dt)
        self._update_hazards(dt)
        if self.mode == MODE_FIGHT:
            self.music.set_paused(not self.player.tracked)
            if self.player.tracked:
                self._resolve_player_attacks()
                self._resolve_enemy_strike()
            if self.enemy.state is EnemyState.DEAD:
                self._end("VICTORY" if self.boss_index + 1 < len(C.BOSSES) else "CHAMPION")
            elif self.player.hp <= 0:
                self._end("DEFEAT")
        elif self.mode == MODE_OVER:
            self.over_timer += dt
            if self.result == "VICTORY" and self.over_timer >= C.NEXT_BOSS_DELAY:
                self.boss_index += 1
                self._start_countdown()

    def _end(self, result: str) -> None:
        if result == "VICTORY" and self.boss_index + 1 > self.checkpoint:
            self.checkpoint = self.boss_index + 1  # the next boss is now unlocked from the menu
            self._save_checkpoint()
        self.mode = MODE_OVER
        self.result = result
        self.music.set_paused(False)  # a paused track can't fade out
        self.music.fade_out()
        if result != "DEFEAT":
            self.sfx.cheer_crowd()
        self.over_timer = 0.0

    # --- Mage spells -----------------------------------------------------------
    def _spawn_spell(self, cast: dict) -> None:
        """Turn a cast (with the player's position captured at cast time) into locked hazards."""
        spell, w = cast["spell"], self.size[0]
        if spell == "meteor":
            self.hazards += make_meteor(cast)
        elif spell == "icicle_rain":
            self.hazards += make_icicles(cast, w)
        else:
            kind = "low" if spell == "fire_wall_low" else "high"
            self.hazards.append(make_fire_wall(kind, cast, cast["origin_x"], cast["ground_y"], self.enemy.H, w))
        self.sfx.throw(0.6)

    def _teleport_effect(self, kind: str, pos) -> None:
        if kind == "tell":
            self.effects.burst(pos, (170, 140, 255), 14, 200, life=(0.3, 0.6), gravity=-200)
        else:
            self.effects.burst(pos, (150, 110, 255), 34, 420, size=(3.0, 7.0), gravity=-150)
            self.effects.burst(pos, (235, 225, 255), 14, 260, gravity=-150)
            if kind == "out":
                self.sfx.throw(0.7)

    def _update_hazards(self, dt: float) -> None:
        player = self.player
        live = self.mode == MODE_FIGHT and player.tracked
        remaining = []
        for hz in self.hazards:
            if isinstance(hz, GroundStrike):
                if hz.update(dt):
                    self._ground_impact(hz, live)
            else:
                hz.update(dt)
                if live and hz.launched and not hz.resolved:
                    self._check_fire_wall(hz)
            if not hz.finished:
                remaining.append(hz)
        self.hazards = remaining

    def _ground_impact(self, hz: GroundStrike, live: bool) -> None:
        pos = (hz.x, hz.ground_y)
        if hz.kind == "meteor":
            self.effects.burst(pos, (255, 230, 140), 40, 700, size=(3.0, 8.0))
            self.effects.burst(pos, (255, 110, 30), 40, 520, size=(4.0, 9.0))
            self.effects.burst(pos, (60, 45, 40), 26, 300, size=(6.0, 12.0), gravity=-120)
            self.effects.shake(16)
            self.effects.flash((255, 120, 30), 70)
            self.sfx.punch(1.0)
        else:
            self.effects.burst(pos, (200, 240, 255), 14, 360, size=(2.0, 5.0))
            self.effects.shake(3)
            self.sfx.punch(0.3)
        p = self.player
        if live and hz.covers(p.hip_mid()[0], C.PLAYER_HALF_WIDTH * p.unit):  # still standing in the zone
            p.hp = max(0.0, p.hp - hz.damage)
            label = "Meteor!" if hz.kind == "meteor" else "Icicle!"
            self.effects.text(f"{label} -{hz.damage:.0f}", (p.com_x(), hz.ground_y - p.unit * 4), (255, 90, 90), 42)
            self.effects.flash((180, 0, 0), 90)

    def _check_fire_wall(self, hz: FireWall) -> None:
        """Safe if you're airborne (low wall) or ducked under it (high wall) at any moment between
        the wall reaching your body and its middle passing your middle."""
        p = self.player
        px, half = p.com_x(), C.PLAYER_HALF_WIDTH * p.unit
        lead = hz.x + hz.direction * hz.width / 2
        if not hz.window_open and (lead - (px - hz.direction * half)) * hz.direction >= 0:
            hz.window_open = True
        if not hz.window_open:
            return
        if hz.kind == "low":
            safe = p.is_jumping
        else:
            r, head, torso = hz.rect(), p.head(), p.torso()
            safe = not ((head is not None and circle_hits_rect(head, r))
                        or (torso is not None and capsule_hits_rect(torso, r)))
        hz.evaded = hz.evaded or safe
        if (hz.x - px) * hz.direction >= 0:  # the wall's middle has passed yours: decide once
            hz.resolved = True
            y = hz.y_top - 30 if hz.kind == "high" else hz.y_top - 40
            if hz.evaded:
                self.effects.text("JUMPED!" if hz.kind == "low" else "DUCKED!", (px, y), (120, 255, 160), 44)
            else:
                p.hp = max(0.0, p.hp - C.FIRE_WALL_DAMAGE)
                self.effects.burst((px, (hz.y_top + hz.y_bottom) / 2), (255, 140, 40), 24, 420)
                self.effects.text(f"Fire Wall! -{C.FIRE_WALL_DAMAGE}", (px, y), (255, 90, 90), 42)
                self.effects.flash((200, 60, 0), 90)
                self.effects.shake(8)
                self.sfx.punch(0.8)

    def _update_stars(self, dt: float) -> None:
        """Move throwing stars and hit the player's head/torso. Forearms don't stop them: duck!"""
        enemy, player = self.enemy, self.player
        for start, aim in enemy.pop_thrown():
            star = ThrowingStar(start, aim, C.STAR_SPEED * enemy.H, C.STAR_RADIUS * enemy.H)
            star.side = 1 if start[0] > player.com_x() else -1  # which side of the player it came from
            self.stars.append(star)
            self.sfx.throw()
        live = self.mode == MODE_FIGHT and player.tracked
        remaining = []
        for star in self.stars:
            star.update(dt)
            if star.offscreen(*self.size):
                continue
            if live and not star.passed_player:
                hit = None
                head, torso = player.head(), player.torso()
                if head is not None and intersects(star.collider, head):
                    hit = "head"
                elif torso is not None and intersects(star.collider, torso):
                    hit = "torso"
                if hit:
                    damage = C.STAR_DAMAGE * (C.HEAD_HIT_MULTIPLIER if hit == "head" else 1.0)
                    player.hp = max(0.0, player.hp - damage)
                    pos = (star.x, star.y)
                    self.effects.burst(pos, (255, 60, 60), 20, 420)
                    self.effects.burst(pos, (220, 220, 235), 10, 300)
                    self.effects.text(f"Throwing Star! -{damage:.0f}", (pos[0], pos[1] - 30), (255, 90, 90), 42)
                    self.effects.flash((180, 0, 0), 110)
                    self.effects.shake(10)
                    self.sfx.punch(0.9)
                    continue
                if (star.x - player.com_x()) * star.side < -0.5 * player.unit:
                    star.passed_player = True
                    self.effects.text("DUCKED!", (player.com_x(), star.y - 30), (120, 255, 160), 44)
            remaining.append(star)
        self.stars = remaining

    def _resolve_player_attacks(self) -> None:
        """Punches (fast fists) and kicks (fast feet) that reach the shadow's body or head."""
        enemy, player = self.enemy, self.player
        if not enemy.can_be_hit or not player.in_range(C.PLAYER_PUNCH_MIN_DEPTH):
            return
        hurtboxes = enemy.hurtboxes()
        kick_targets = enemy.kick_targets()
        staff = enemy.staff_collider()
        ec = enemy.center()
        strikes = [("punch", side, c, player.fist_vel[side], player.fist_speed(side))
                   for side, c in player.fists().items()]
        strikes += [("kick", side, c, player.foot_vel[side], player.foot_speed(side))
                    for side, c in player.feet().items()]
        for kind, side, limb, (vx, vy), speed in strikes:
            kick = kind == "kick"
            cooldowns = player.kick_cooldown if kick else player.punch_cooldown
            threshold = C.KICK_SPEED_THRESHOLD if kick else C.PUNCH_SPEED_THRESHOLD
            if cooldowns[side] > 0 or speed < threshold:
                continue
            if vx * (ec[0] - limb.center[0]) + vy * (ec[1] - limb.center[1]) <= 0:
                continue  # moving away from the enemy (pulling back), not a strike
            if kick:
                # Head and feet take priority: those are the hits that knock the boss down.
                part = next((name for name, hb in kick_targets if intersects(limb, hb)
                             and (name != "feet" or speed >= threshold * C.LOW_KICK_SPEED_FACTOR)), None)
            else:
                part = "body" if any(intersects(limb, hb) for hb in hurtboxes) else None
            if part is None:
                if staff is not None and intersects(limb, staff):  # missed the body but hit the staff
                    cooldowns[side] = C.KICK_COOLDOWN if kick else C.PUNCH_COOLDOWN
                    self._strike_staff(limb.center)
                    return
                continue

            if enemy.try_dodge():
                cooldowns[side] = C.KICK_COOLDOWN if kick else C.PUNCH_COOLDOWN
                self.effects.text("ROLLED AWAY" if enemy.rolling else "DODGED", (ec[0], ec[1] - enemy.H * 0.4),
                                  (200, 200, 255), 44)
                return
            power = min(1.0, (speed - threshold) / threshold)
            lo, hi = (C.KICK_DAMAGE_MIN, C.KICK_DAMAGE_MAX) if kick else (C.PUNCH_DAMAGE_MIN, C.PUNCH_DAMAGE_MAX)
            damage = round(lo + (hi - lo) * power)
            cooldowns[side] = C.KICK_COOLDOWN if kick else C.PUNCH_COOLDOWN
            result = enemy.take_kick(damage, part) if kick else enemy.take_hit(damage)
            pos = limb.center

            if result == "blocked":
                dealt = round(damage * C.ENEMY_KICK_BLOCK_DAMAGE)
                self.effects.burst(pos, (200, 150, 255), 16, 360)
                self.effects.text(f"BLOCKED -{dealt}", (pos[0], pos[1] - 40), (200, 160, 255), 40)
                self.effects.shake(5)
                self.sfx.punch(0.5)
            else:
                self.effects.burst(pos, (255, 240, 200), 30 if kick else 22, 620 if kick else 520)
                self.effects.burst(pos, (255, 140, 40), 12, 320)
                self.effects.text(f"-{damage}", (pos[0], pos[1] - 40), (255, 220, 120), 48 if kick else 40)
                self.effects.shake((10 if kick else 6) + 6 * power)
                self.sfx.punch(0.85 + 0.15 * power if kick else 0.7 + 0.3 * power)
            if result == "knockdown":
                how = "HEAD KICK!" if part == "head" else "LEG SWEEP!"
                self.effects.text(f"{how} KNOCKDOWN!", (ec[0], ec[1] - enemy.H * 0.45), (255, 200, 80), 60, 1.2)
            elif result == "ko":
                sk = enemy.sk
                self.effects.smoke([sk[k] for k in ("head", "shoulder", "hip", "f_hand", "r_hand",
                                                    "f_elbow", "r_elbow", "f_knee", "r_knee",
                                                    "f_ankle", "r_ankle")], per_point=10)
                self.effects.text("K.O.", (ec[0], ec[1] - enemy.H * 0.5), (255, 255, 255), 120, 1.6)
                self.effects.shake(18)
            if not enemy.can_be_hit:
                return

    def _strike_staff(self, pos) -> None:
        if self.enemy.staff_struck() == "parry":
            self.effects.burst(pos, (255, 220, 120), 26, 520)
            self.effects.text("PARRY!", (pos[0], pos[1] - 40), (255, 220, 120), 56)
            self.effects.shake(8)
            self.sfx.punch(0.8)
        else:
            self.effects.burst(pos, (200, 170, 120), 12, 300)
            self.effects.text("STAFF BLOCK", (pos[0], pos[1] - 40), (220, 190, 140), 36)
            self.sfx.punch(0.4)

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
        shield = player.shield()
        if shield is not None:
            blade = self.enemy.sword_capsules() if spec.limb == "sword" else []
            if intersects(strike, shield) or any(intersects(cap, shield) for cap in blade):
                return "shield"
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
        if contact == "shield":
            if spec.limb == "sword":  # parry: sword flung back, he staggers wide open
                enemy.sword_parried()
                self.effects.burst(pos, (255, 230, 140), 30, 560)
                self.effects.text("PARRY!", (pos[0], pos[1] - 40), (255, 220, 120), 60)
                self.effects.shake(9)
                self.sfx.punch(0.9)
            else:
                enemy.on_blocked()
                self.effects.burst(pos, (220, 180, 100), 16, 380)
                self.effects.text("SHIELD BLOCK", (pos[0], pos[1] - 30), (230, 200, 130), 38)
                self.effects.shake(4)
                self.sfx.punch(0.5)
            return
        if contact == "block":
            chip = spec.damage * spec.block_chip
            player.hp = max(0.0, player.hp - chip)
            enemy.on_blocked()
            self.effects.burst(pos, (120, 200, 255), 18, 380)
            self.effects.text("BLOCK" if chip < 0.5 else f"BLOCK -{chip:.0f}", (pos[0], pos[1] - 30),
                              (140, 210, 255), 40)
            self.effects.shake(4)
            self.sfx.punch(0.45)
            return

        damage = spec.damage * (C.HEAD_HIT_MULTIPLIER if contact == "head" else 1.0)
        player.hp = max(0.0, player.hp - damage)
        self.effects.burst(pos, (255, 60, 60), 20, 420)
        self.effects.text(f"{spec.label}! -{damage:.0f}", (pos[0], pos[1] - 30), (255, 90, 90), 42)
        self.effects.flash((180, 0, 0), 110)
        self.sfx.punch(1.0)
        self.effects.shake(10 + damage * 0.5)

    # --- rendering -----------------------------------------------------------
    def _draw(self) -> None:
        world = self.world
        if self.background is not None:
            world.blit(self.background, (0, 0))
        else:
            world.fill((12, 10, 18))

        show_enemy = self.mode in (MODE_FIGHT, MODE_OVER) or (self.mode == MODE_COUNTDOWN and self.enemy_ready)
        now = time.perf_counter()
        for hz in self.hazards:  # floor warnings / scorch marks go under everyone
            if isinstance(hz, GroundStrike):
                hz.draw_floor(world, now)
        if show_enemy:
            self.enemy.draw(world)
        if self.mode in (MODE_FIGHT, MODE_OVER):
            self.player.draw_shield(world)
        for hz in self.hazards:
            if isinstance(hz, GroundStrike):
                hz.draw_falling(world, -60)
            else:
                hz.draw(world, now)
        for star in self.stars:
            star.draw(world)
        self.effects.draw_world(world)
        if self.mode in (MODE_COUNTDOWN, MODE_CALIBRATE) or (self.debug and self.mode != MODE_MENU):
            self.player.draw_debug(world, self.font_small, colliders=self.debug)
        if self.debug and show_enemy:
            self.enemy.draw_debug(world)

        self.screen.fill((0, 0, 0))
        self.screen.blit(world, self.effects.shake_offset())
        self.effects.draw_overlay(self.screen)
        if self.mode == MODE_MENU:
            self._draw_menu()
        else:
            self._draw_hud()
        pygame.display.flip()

    def _dim(self, alpha: int) -> None:
        self._veil.set_alpha(alpha)
        self.screen.blit(self._veil, (0, 0))

    def _draw_menu(self) -> None:
        w, h = self.size
        self._dim(170)
        self._text("SHADOW FIGHTER", (w / 2, h * 0.17), self.font_huge, (235, 225, 255), "center")
        self._text("An AR fighting game: your body is the controller", (w / 2, h * 0.28), self.font,
                   (200, 190, 230), "center")

        self.menu_rects = []
        items = self._menu_items()
        bosses_page = self.menu_page == "bosses"
        if bosses_page:
            self._text("Select a boss", (w / 2, h * 0.345), self.font_menu, (255, 220, 120), "center")
        bw = max(340, max(self.font_menu.size(label)[0] for label, _ in items) + 60)  # fit the longest label
        top, gap = (h * 0.44, 60) if bosses_page else (h * 0.38, 66)
        for i, (label, _) in enumerate(items):
            rect = pygame.Rect(0, 0, bw, 52)
            rect.center = (w // 2, int(top) + i * gap)
            selected = i == self.menu_index
            pygame.draw.rect(self.screen, (120, 70, 200) if selected else (40, 34, 56), rect, border_radius=12)
            pygame.draw.rect(self.screen, (220, 200, 255) if selected else (90, 80, 120), rect, 3, border_radius=12)
            self._text(label, rect.center, self.font_menu, (255, 255, 255), "center")
            self.menu_rects.append(rect)

        tips = [
            "Stand about 6 ft (2 m) back with your whole body in view.",
            "Punch and kick sideways at the boss.  Raise your forearms to block.",
            "Jump over low sweeps.  Duck under throwing stars.  Step back to get out of range.",
            "Parry staff and sword.  Dodge the Mage's spells, then punish him.  Beat all five bosses!",
        ]
        tips_y = self.menu_rects[-1].bottom + 36  # always below the last button
        for i, tip in enumerate([] if bosses_page else tips):
            self._text(tip, (w / 2, tips_y + i * 30), self.font, (210, 210, 220), "center")
        self._text("Up/Down + Enter or click    [M] mute    [F] fullscreen    "
                   + ("[Esc] back" if bosses_page else "[Esc] quit"),
                   (w / 2, h - 30), self.font_small, (170, 170, 185), "center")
        if self.camera.error:
            self._text(f"Camera error: {self.camera.error}", (w / 2, h - 60), self.font_small, (255, 90, 90),
                       "center")

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

        if self.mode in (MODE_FIGHT, MODE_OVER):
            bw = int(w * 0.38)
            self._bar(30, 30, bw, 22, self.player.hp, self.trail_player, C.PLAYER_MAX_HP, C.PLAYER_BAR)
            self._bar(w - 30 - bw, 30, bw, 22, self.enemy.hp, self.trail_enemy, self.enemy.max_hp, C.ENEMY_BAR,
                      True)
            self._text("YOU", (30, 58), self.font)
            self._text(self.enemy.name, (w - 30, 58), self.font, anchor="topright")
            self._text(f"ROUND {self.boss_index + 1}/{len(C.BOSSES)}", (w / 2, 30), self.font, anchor="midtop")
            in_range = self.player.in_range(C.PLAYER_PUNCH_MIN_DEPTH)
            self._text(f"RANGE {'IN' if in_range else 'OUT'}  ({self.player.depth_ratio:.2f})", (30, 86),
                       self.font_small, (120, 255, 160) if in_range else (255, 200, 80))
            if self.player.is_jumping:
                self._text("AIRBORNE", (30, 106), self.font_small, (120, 200, 255))

        stats = (f"FPS {self.pacer.fps:4.0f}   CAM {self.camera.camera_fps:4.1f}   "
                 f"POSE {self.pose_ms:4.1f} ms   [D] debug {'ON' if self.debug else 'off'}   "
                 "[R] restart   [C] calibrate   [F] fullscreen   "
                 f"[M] {'unmute' if self.music.muted or self.sfx.muted else 'mute'}   [Esc] menu")
        self._text(stats, (14, h - 26), self.font_small, (200, 200, 210))

        if cam_err:
            self._text("Camera / pose error", (w / 2, h / 2 - 40), self.font_big, (255, 90, 90), "center")
            self._text(cam_err, (w / 2, h / 2 + 30), self.font, anchor="center")
            return
        if self.background is None:
            self._text("Starting camera...", (w / 2, h / 2), self.font_big, anchor="center")
            return

        if self.mode == MODE_COUNTDOWN:
            self._text(f"ROUND {self.boss_index + 1}: {self.boss['name']}", (w / 2, h * 0.12), self.font_big,
                       (235, 225, 255), "center")
            self._text("Get into position: about 6 ft (2 m) back, whole body in view", (w / 2, h * 0.21),
                       self.font, anchor="center")
            n = max(1, math.ceil(self.countdown_t))
            frac = n - self.countdown_t  # 0 -> 1 within each second
            size = int(220 + 120 * (1 - frac))
            digit = pygame.font.Font(None, size).render(str(n), True, (255, 255, 255))
            digit.set_alpha(int(255 * min(1.0, 1.4 - frac)))
            self.screen.blit(digit, digit.get_rect(center=(w // 2, int(h * 0.5))))
        elif self.mode == MODE_CALIBRATE:
            self._text("CALIBRATION", (w / 2, h * 0.12), self.font_big, anchor="center")
            self._text("Hold still with your whole body in view.", (w / 2, h * 0.21), self.font, anchor="center")
            if not self.player.tracked:
                self._text("No body detected: step into view", (w / 2, h * 0.27), self.font,
                           (255, 200, 80), "center")
            bw = int(w * 0.4)
            self._bar(int(w / 2 - bw / 2), int(h * 0.31), bw, 16, self.player.calibration_progress,
                      self.player.calibration_progress, 1.0, (120, 255, 160))
        elif self.mode == MODE_FIGHT and not self.player.tracked:
            self._text("STEP INTO VIEW", (w / 2, h * 0.45), self.font_big, (255, 200, 80), "center")
        elif self.mode == MODE_OVER and self.over_timer > 0.8:
            if self.result == "VICTORY":
                nxt = C.BOSSES[self.boss_index + 1]["name"]
                self._text("VICTORY", (w / 2, h * 0.4), self.font_big, (120, 255, 160), "center")
                self._text(f"Next up: {nxt}", (w / 2, h * 0.5), self.font, anchor="center")
            elif self.result == "CHAMPION":
                self._text("CHAMPION!", (w / 2, h * 0.4), self.font_big, (255, 220, 90), "center")
                self._text("You beat every boss.  Enter/R to play again, Esc for the menu", (w / 2, h * 0.5),
                           self.font, anchor="center")
            else:
                self._text("DEFEAT", (w / 2, h * 0.4), self.font_big, (255, 90, 90), "center")
                self._text(f"Enter/R to retry {self.enemy.name}, C to recalibrate, Esc for the menu",
                           (w / 2, h * 0.5), self.font, anchor="center")


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
