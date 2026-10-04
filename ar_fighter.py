#!/usr/bin/env python3
"""ARena: Fit Fighter: fight shadow bosses with your body through your webcam.

Run:   python ar_fighter.py [--camera 0] [--complexity 1] [--debug]

Flow: menu -> (first time: 3 s to get into position -> calibration) -> boss intro
(WARNING, name and title, 3-2-1) -> fight -> workout summary with a Fitness Score, then
Next Boss / Fight Again / Select Boss / Main Menu. Bosses 1-5 in order, or pick any boss
from Select Boss. Wins that make the top 10 go on the Fitness Score leaderboard.

Controls:
    Enter  start (menu) / skip the boss intro / choose an option (Fight Summary)
    D      toggle hitbox / skeleton debug overlay
    R      restart the current fight (also Fight Again from the summary)
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

from fighter import arcade
from fighter import config as C
from fighter.audio import Music
from fighter.sfx import SoundEffects
from fighter.camera import CameraThread
from fighter.effects import Effects, make_vignette, render_outlined
from fighter.enemy import EnemyState, ShadowEnemy
from fighter.geometry import intersects
from fighter.leaderboard import Leaderboard
from fighter.player import SIDES, PlayerTracker
from fighter.projectiles import ThrowingStar
from fighter.spells import FireWall, GroundStrike, capsule_hits_rect, circle_hits_rect, make_fire_wall, \
    make_icicles, make_meteor
from fighter.stats import FightStats, format_duration, rank

MODE_MENU, MODE_COUNTDOWN, MODE_CALIBRATE, MODE_INTRO = "menu", "countdown", "calibrate", "intro"
MODE_FIGHT, MODE_OVER, MODE_SUMMARY = "fight", "over", "summary"
RANK_COLORS = {"S": (255, 220, 90), "A": (120, 255, 160), "B": (110, 210, 255), "C": (255, 170, 90),
               "D": (255, 90, 90), "-": (170, 170, 185)}


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
        self._veil = pygame.Surface(self.size).convert()  # cabinet black, for dimming behind the menu
        self._veil.fill(arcade.INK)
        self._tint = pygame.Surface(self.size).convert()  # red, for the boss intro's WARNING
        self._tint.fill((150, 0, 0))
        self._vignette = make_vignette(self.size, (200, 0, 0))  # low-health pulse at the screen edges
        self.low_health = 0.0         # 0 = fine; >0 = how close to death (0..1) while at or below LOW_HEALTH_FRACTION
        self.pacer = FramePacer(C.FPS)

        # Arcade type: Press Start 2P ('pixel') for titles, numbers and buttons, VT323 ('term') for reading text.
        self.font_tiny = arcade.term(24)
        self.font_small = arcade.term(26)
        self.font = arcade.term(32)
        self.font_row = arcade.term(30)
        self.font_label = arcade.pixel(12)
        self.font_menu = arcade.pixel(16)
        self.font_head = arcade.pixel(22)
        self.font_grade = arcade.pixel(28)
        self.font_stat = arcade.pixel(32)
        self.font_big = arcade.pixel(44)
        self.font_huge = arcade.pixel(64)
        self.font_rank = arcade.pixel(80)
        self.font_title = arcade.pixel(104)
        self.font_count = arcade.pixel(128)

        self.camera = CameraThread(self.size, camera_index, complexity)
        self.camera.start()

        self.player = PlayerTracker(*self.size)
        self.enemy = ShadowEnemy(*self.size)
        self.effects = Effects(self.size)
        here = Path(__file__).resolve().parent
        self.save_path = here / C.SAVE_FILE
        self.checkpoint = self._load_checkpoint()  # furthest boss reached by beating the one before
        self.leaderboard = Leaderboard(here / C.LEADERBOARD_FILE)
        self.name_entry = None        # initials being typed after a top-10 win (None = not entering)
        self.board_pos = None         # 1-based leaderboard place just earned (pop-up showing the board)
        self.music = Music({"fight": here / C.MUSIC_DIR, "menu": here / C.MENU_MUSIC_DIR},
                           optional={f"boss:{b['name']}": here / b["music"] for b in C.BOSSES if b.get("music")})
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
        self.stats = FightStats()     # replaced with a fresh one at the start of every fight
        self._melee = None            # (enemy.attack_seq, stats id) of the melee strike being tracked
        self.summary_t = 0.0
        self.intro_t = 0.0
        self.intro_from_x = self.intro_to_x = 0.0
        self.slowmo_t = 0.0           # real seconds of K.O. hit-stop / slow motion left
        self.combo_pop_t = 9.0        # seconds since the combo counter last went up (drives its pop)
        self._now = 0.0
        self.menu_index = 0
        self.menu_page = "main"
        self.menu_rects = []
        self.trail_player = float(C.PLAYER_MAX_HP)
        self.trail_enemy = self.enemy.max_hp
        self.music.play("menu")

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
            elif self._summary_popup() and event.type in (pygame.KEYDOWN, pygame.MOUSEBUTTONDOWN):
                self._popup_input(event)
            elif self.mode in (MODE_MENU, MODE_SUMMARY) and event.type == pygame.MOUSEMOTION:
                for i, rect in enumerate(self.menu_rects):
                    if rect.collidepoint(event.pos):
                        self.menu_index = i
            elif self.mode in (MODE_MENU, MODE_SUMMARY) and event.type == pygame.MOUSEBUTTONDOWN \
                    and event.button == 1:
                for i, rect in enumerate(self.menu_rects):
                    if rect.collidepoint(event.pos):
                        (self._menu_select if self.mode == MODE_MENU else self._summary_select)(i)
                        break
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
            if key == pygame.K_ESCAPE and self.menu_page != "main":
                self.menu_page, self.menu_index = "main", 0
            elif key == pygame.K_ESCAPE:
                self.running = False
            elif key in (pygame.K_UP, pygame.K_w):
                self.menu_index = (self.menu_index - 1) % len(self._menu_items())
            elif key in (pygame.K_DOWN, pygame.K_s):
                self.menu_index = (self.menu_index + 1) % len(self._menu_items())
            elif key in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_SPACE):
                self._menu_select(self.menu_index)
        elif self.mode == MODE_SUMMARY and key in (pygame.K_LEFT, pygame.K_UP, pygame.K_RIGHT, pygame.K_DOWN,
                                                   pygame.K_TAB):
            step = -1 if key in (pygame.K_LEFT, pygame.K_UP) else 1
            self.menu_index = (self.menu_index + step) % len(self._summary_items())
        elif self.mode == MODE_SUMMARY and key in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_SPACE):
            self._summary_select(self.menu_index)
        elif self.mode == MODE_INTRO and key in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_SPACE):
            self.intro_t = max(self.intro_t, C.INTRO_WARNING_TIME + C.INTRO_TITLE_TIME)  # straight to 3-2-1
        elif key == pygame.K_ESCAPE:
            self._open_menu()
        elif key == pygame.K_c:
            self.player.reset_calibration()
            self._start_countdown()
        elif key == pygame.K_r and self.mode in (MODE_FIGHT, MODE_OVER, MODE_SUMMARY):
            self._start_countdown()  # same boss again
        elif key in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_SPACE) and self.mode == MODE_OVER:
            self._show_summary()     # skip the result banner

    def _menu_items(self):
        """(label, action) pairs for the current menu page.

        Main page: 'Continue' appears once you've beaten at least one boss; 'Select Boss'
        opens a page listing every boss so you can fight any of them directly.
        """
        if self.menu_page == "bosses":
            return [(f"{i + 1}  {b['name']}", f"boss:{i}") for i, b in enumerate(C.BOSSES)] + \
                   [("BACK", "back")]
        if self.menu_page == "leaderboard":
            return [("BACK", "back")]
        items = []
        if self.checkpoint > 0:
            nxt = C.BOSSES[self.checkpoint]
            items.append((f"CONTINUE: {nxt['name']}", "continue"))
        items.append(("NEW GAME" if self.checkpoint > 0 else "START", "new"))
        items += [("SELECT BOSS", "select"), ("HIGH SCORES", "leaderboard"), ("QUIT", "quit")]
        return items

    def _menu_select(self, index: int) -> None:
        action = self._menu_items()[index][1]
        if action == "quit":
            self.running = False
        elif action == "select":
            self.menu_page, self.menu_index = "bosses", 0
        elif action == "leaderboard":
            self.menu_page, self.menu_index = "leaderboard", 0
        elif action == "back":
            self.menu_page, self.menu_index = "main", 0
        else:
            if action.startswith("boss:"):
                self.boss_index = int(action.split(":")[1])
            else:
                self.boss_index = self.checkpoint if action == "continue" else 0
            self.menu_page = "main"
            self._start_countdown()

    def _summary_items(self):
        """(label, action) pairs for the buttons under the Fight Summary."""
        items = []
        if self.result == "VICTORY":
            items.append((f"NEXT: {C.BOSSES[self.boss_index + 1]['name']}", "next"))
        return items + [("REMATCH", "again"), ("PICK BOSS", "select"), ("MENU", "menu")]

    def _summary_select(self, index: int) -> None:
        action = self._summary_items()[index][1]
        if action == "next":
            self.boss_index += 1
            self._start_countdown()
        elif action == "again":
            self._start_countdown()
        elif action == "select":
            self._open_menu()
            self.menu_page, self.menu_index = "bosses", self.boss_index
        else:
            self._open_menu()

    def _show_summary(self) -> None:
        self.board_pos = None
        won = self.result in ("VICTORY", "CHAMPION")
        self.name_entry = "" if won and self.leaderboard.qualifies(self.stats.fitness_score()) else None
        if self.result == "DEFEAT":
            self.sfx.groan_crowd()
        else:
            self.sfx.cheer_crowd()
        self.mode = MODE_SUMMARY
        self.summary_t = 0.0
        self.menu_index = 0
        self.menu_rects = []

    def _summary_popup(self):
        """'entry' (typing initials), 'board' (showing your new place), 'wait' (entry not shown yet) or None."""
        if self.mode != MODE_SUMMARY:
            return None
        if self.board_pos is not None:
            return "board"
        if self.name_entry is not None:
            return "entry" if self.summary_t >= C.SUMMARY_TALLY_TIME + 0.5 else "wait"
        return None

    def _popup_input(self, event) -> None:
        popup = self._summary_popup()
        if popup == "board":
            if event.type == pygame.MOUSEBUTTONDOWN or event.key in (pygame.K_RETURN, pygame.K_KP_ENTER,
                                                                     pygame.K_SPACE, pygame.K_ESCAPE):
                self.board_pos = None
                self.menu_index = 0
        elif popup == "entry" and event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                self.name_entry = None  # skip: the score isn't saved
            elif event.key == pygame.K_BACKSPACE:
                self.name_entry = self.name_entry[:-1]
            elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                if self.name_entry:
                    s = self.stats
                    self.board_pos = self.leaderboard.add(self.name_entry, s.fitness_score(), s.boss_name,
                                                          s.overall_rank()) or None
                    self.name_entry = None
                    self.sfx.bell_ring()
            elif event.unicode and event.unicode.isascii() and event.unicode.isalnum() \
                    and len(self.name_entry) < C.LEADERBOARD_NAME_LEN:
                self.name_entry += event.unicode.upper()
        # 'wait': swallow input so an early key press can't skip the entry or hit a button

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
        self.slowmo_t = 0.0
        self.effects.clear()
        self.stars.clear()
        self.hazards.clear()

    def _open_menu(self) -> None:
        self._stop_fight()
        self.sfx.stop()
        self.mode = MODE_MENU
        self.menu_index = 0
        self.menu_page = "main"
        self.menu_rects = []
        self.music.play("menu")

    def _start_countdown(self) -> None:
        """Start the next fight: the boss intro, or first a few seconds to get into position and calibrate."""
        self._stop_fight()
        self.result = ""
        self.enemy_ready = False
        if self.player.calibrated:
            self._start_intro()
        else:
            self.mode = MODE_COUNTDOWN
            self.countdown_t = float(C.COUNTDOWN_SECONDS)

    def _start_intro(self) -> None:
        """WARNING -> boss name and title while he walks in -> 3-2-1 -> FIGHT! His theme starts here."""
        self._place_enemy()
        e = self.enemy
        self.intro_to_x = e.x
        self.intro_from_x = self.size[0] + 0.3 * e.H if e.facing < 0 else -0.3 * e.H  # walks in from his side
        e.x = self.intro_from_x
        e.walk_in(0.0, e.x, 0.0)
        self.intro_t = 0.0
        self.mode = MODE_INTRO
        self.music.play(f"boss:{self.boss['name']}", fallback="fight")

    def _place_enemy(self) -> None:
        p = self.player
        com = p.com_x() if p.tracked else self.size[0] * 0.5
        spawn_x = self.size[0] * (0.82 if com < self.size[0] * 0.5 else 0.18)
        self.enemy.reset(ground_y=p.ground_y, height=p.body_height, x=spawn_x, boss=self.boss)
        p.shield_enabled = bool(self.boss.get("player_shield"))
        self.enemy.facing = 1 if com > spawn_x else -1
        self.enemy_ready = True

    def _begin_fight(self, now: float) -> None:
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
        self.stats = FightStats(self.boss["name"], now)  # never carried over between fights
        self._melee = None
        self.mode = MODE_FIGHT
        self.sfx.stop()  # cut off a victory cheer still playing from the last fight
        self.sfx.bell_ring()
        self.music.play(f"boss:{self.boss['name']}", fallback="fight")  # keeps playing if the intro started it
        self.effects.text("FIGHT!", (self.size[0] / 2, self.size[1] * 0.35), (255, 255, 255), 110, 1.2)
        if p.shield_enabled:
            self.effects.text("You have a SHIELD on your left arm: catch his sword on it to parry!",
                              (self.size[0] / 2, self.size[1] * 0.47), (255, 215, 120), 34, 3.5)
        elif not p.full_body_visible:
            self.effects.text("Feet not visible: step back so you can kick and jump over sweeps",
                              (self.size[0] / 2, self.size[1] * 0.47), (255, 200, 80), 32, 3.0)

    def _time_scale(self, dt: float) -> float:
        """K.O.: a short freeze-frame, then slow motion; 1.0 the rest of the time."""
        if self.slowmo_t <= 0:
            return 1.0
        elapsed = C.KO_SLOWMO_TIME - self.slowmo_t
        self.slowmo_t -= dt
        return 0.0 if elapsed < C.KO_HITSTOP else C.KO_SLOWMO_SCALE

    def _update_low_health(self, now: float) -> None:
        """Heartbeat + red edge pulse while you're fighting on LOW_HEALTH_FRACTION of your health or less."""
        limit = C.PLAYER_MAX_HP * C.LOW_HEALTH_FRACTION
        hp = self.player.hp
        low = self.mode == MODE_FIGHT and 0 < hp <= limit
        self.low_health = (0.35 + 0.65 * (1.0 - hp / limit)) if low else 0.0  # stronger the closer to 0
        self.sfx.set_heartbeat(low, 0.7 + 0.3 * self.low_health, now)

    def _update(self, dt: float, now: float) -> None:
        self._now = now
        self.player.update(dt, now)
        self._update_low_health(now)
        wdt = dt * self._time_scale(dt)  # "world" time: the boss, projectiles, spells and particles
        self.effects.update(wdt)
        self.combo_pop_t += dt
        self.trail_player += (self.player.hp - self.trail_player) * min(1.0, dt * 3)
        self.trail_enemy += (self.enemy.hp - self.trail_enemy) * min(1.0, dt * 3)

        if self.mode == MODE_SUMMARY:  # the arena stays frozen behind the summary
            before = self.summary_t
            self.summary_t += dt
            if before <= C.SUMMARY_TALLY_TIME < self.summary_t:
                self.sfx.punch(0.9)  # the rank stamps down
            return
        if self.mode == MODE_MENU:
            return
        if self.mode == MODE_COUNTDOWN:
            self.countdown_t -= dt
            if self.countdown_t <= 0:
                if self.player.calibrated:
                    self._start_intro()
                else:
                    self.mode = MODE_CALIBRATE
                    self.player.calibration_enabled = True
            return
        if self.mode == MODE_CALIBRATE:
            if self.player.calibrated:
                self._start_intro()
            return
        if self.mode == MODE_INTRO:
            self.intro_t += dt
            walk_time = C.INTRO_WARNING_TIME + C.INTRO_TITLE_TIME
            speed = abs(self.intro_to_x - self.intro_from_x) / (walk_time * 0.85)
            self.enemy.walk_in(dt, self.intro_to_x, speed)
            if self.intro_t >= walk_time + C.COUNTDOWN_SECONDS:
                self._begin_fight(now)
            return

        self.enemy.hazards_active = bool(self.hazards)  # one spell at a time: never overlapping hazards
        self.enemy.update(wdt, self.player)
        for cast in self.enemy.pop_casts():
            self._spawn_spell(cast)
        for kind, pos in self.enemy.pop_teleport_events():
            self._teleport_effect(kind, pos)
        if self.enemy.pop_flip_started():  # leaping over the player
            self.sfx.throw(0.8)
            self.effects.burst((self.enemy.x, self.enemy.ground_y), (70, 60, 50), 14, 260, size=(3.0, 6.0))
        if self.enemy.pop_landed():  # boss hit the floor after a knockdown (or the K.O.)
            ko = self.enemy.state is EnemyState.DEAD
            self.effects.shake(22 if ko else 12)
            self.effects.burst((self.enemy.x - self.enemy.facing * self.enemy.H * 0.4, self.enemy.ground_y),
                               (60, 50, 70), 44 if ko else 24, 420 if ko else 300, size=(3.0, 7.0), gravity=600)
            self.sfx.punch(1.0 if ko else 0.6)
        self._update_stars(wdt)
        self._update_hazards(wdt)
        if self.mode == MODE_FIGHT:
            self.music.set_paused(not self.player.tracked)
            self._track_enemy_melee()
            if self.player.tracked:
                self._track_player_throws()
                self._resolve_player_attacks()
                self._resolve_enemy_strike()
            if self.enemy.state is EnemyState.DEAD:
                self._end("VICTORY" if self.boss_index + 1 < len(C.BOSSES) else "CHAMPION", now)
            elif self.player.hp <= 0:
                self._end("DEFEAT", now)
        elif self.mode == MODE_OVER:
            self.over_timer += dt
            if self.over_timer >= C.SUMMARY_DELAY:
                self._show_summary()

    def _end(self, result: str, now: float) -> None:
        self.stats.finish(now, result, self.player.hp)
        if result == "VICTORY" and self.boss_index + 1 > self.checkpoint:
            self.checkpoint = self.boss_index + 1  # the next boss is now unlocked from the menu
            self._save_checkpoint()
        self.mode = MODE_OVER
        self.result = result
        self.music.set_paused(False)  # a paused track can't fade out
        self.music.fade_out()
        self.over_timer = 0.0

    # --- Mage spells -----------------------------------------------------------
    def _spawn_spell(self, cast: dict) -> None:
        """Turn a cast (with the player's position captured at cast time) into locked hazards."""
        spell, w = cast["spell"], self.size[0]
        if spell == "meteor":
            new = make_meteor(cast)
        elif spell == "icicle_rain":
            new = make_icicles(cast, w)
        else:
            kind = "low" if spell == "fire_wall_low" else "high"
            new = [make_fire_wall(kind, cast, cast["origin_x"], cast["ground_y"], self.enemy.H, w)]
        attack_id = self.stats.open_attack(len(new))  # the whole cast is one attack, each hazard one part
        for i, hz in enumerate(new):
            hz.attack_id, hz.attack_part = attack_id, i
        self.hazards += new
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
            else:
                if isinstance(hz, FireWall) and not hz.resolved:  # passed while you were out of view
                    self.stats.void_attack(hz.attack_id)
                self.stats.close_attack(hz.attack_id, hz.attack_part)
        self.hazards = remaining

    def _hurt_player(self, amount: float) -> float:
        """Apply damage to the player; returns how much HP was actually lost."""
        p = self.player
        dealt = min(max(0.0, amount), p.hp)
        p.hp -= dealt
        return dealt

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
        if not live:
            self.stats.void_attack(hz.attack_id)  # landed while you were out of view: can't judge it
        elif hz.covers(p.hip_mid()[0], C.PLAYER_HALF_WIDTH * p.unit):  # still standing in the zone
            self.stats.attack_hit(hz.attack_id, self._hurt_player(hz.damage))
            label = "Meteor!" if hz.kind == "meteor" else "Icicle!"
            self.effects.text(f"{label} -{hz.damage:.0f}", (p.com_x(), hz.ground_y - p.unit * 4), (255, 90, 90), 42)
            self.effects.flash((180, 0, 0), 90)
        self.stats.close_attack(hz.attack_id, hz.attack_part)

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
                self.stats.attack_hit(hz.attack_id, self._hurt_player(C.FIRE_WALL_DAMAGE))
                self.effects.burst((px, (hz.y_top + hz.y_bottom) / 2), (255, 140, 40), 24, 420)
                self.effects.text(f"Fire Wall! -{C.FIRE_WALL_DAMAGE}", (px, y), (255, 90, 90), 42)
                self.effects.flash((200, 60, 0), 90)
                self.effects.shake(8)
                self.sfx.punch(0.8)
            self.stats.close_attack(hz.attack_id, hz.attack_part)

    def _update_stars(self, dt: float) -> None:
        """Move throwing stars and hit the player's head/torso. Forearms don't stop them: duck!"""
        enemy, player = self.enemy, self.player
        for start, aim in enemy.pop_thrown():
            star = ThrowingStar(start, aim, C.STAR_SPEED * enemy.H, C.STAR_RADIUS * enemy.H)
            star.side = 1 if start[0] > player.com_x() else -1  # which side of the player it came from
            star.attack_id = self.stats.open_attack()
            self.stars.append(star)
            self.sfx.throw()
        live = self.mode == MODE_FIGHT and player.tracked
        remaining = []
        for star in self.stars:
            star.update(dt)
            if star.offscreen(*self.size):
                self.stats.close_attack(star.attack_id)
                continue
            if not live and not star.passed_player:
                self.stats.void_attack(star.attack_id)  # you were out of view while it was in flight
            if live and not star.passed_player:
                hit = None
                head, torso = player.head(), player.torso()
                if head is not None and intersects(star.collider, head):
                    hit = "head"
                elif torso is not None and intersects(star.collider, torso):
                    hit = "torso"
                if hit:
                    damage = C.STAR_DAMAGE * (C.HEAD_HIT_MULTIPLIER if hit == "head" else 1.0)
                    self.stats.attack_hit(star.attack_id, self._hurt_player(damage))
                    self.stats.close_attack(star.attack_id)
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
                    self.stats.close_attack(star.attack_id)
                    self.effects.text("DUCKED!", (player.com_x(), star.y - 30), (120, 255, 160), 44)
            remaining.append(star)
        self.stars = remaining

    def _track_player_throws(self) -> None:
        """Feed each fist/foot's motion to the stats, using the same speed / direction / cooldown
        rules as _resolve_player_attacks, so one real punch or kick counts as thrown once."""
        enemy, player = self.enemy, self.player
        evaluating = enemy.can_be_hit and player.in_range(C.PLAYER_PUNCH_MIN_DEPTH)
        ec = enemy.center()
        feet = player.feet()
        for kind, limbs, vel, speed_of, threshold, cooldowns in (
                ("punch", player.fists(), player.fist_vel, player.fist_speed, C.PUNCH_SPEED_THRESHOLD,
                 player.punch_cooldown),
                ("kick", feet, player.foot_vel, player.foot_speed, C.KICK_SPEED_THRESHOLD, player.kick_cooldown)):
            for side in SIDES:
                limb = limbs.get(side)
                if limb is None:
                    self.stats.update_limb(kind, side, False, False)
                    continue
                vx, vy = vel[side]
                speed = speed_of(side)
                toward = vx * (ec[0] - limb.center[0]) + vy * (ec[1] - limb.center[1]) > 0
                striking = evaluating and toward and speed >= threshold and cooldowns[side] <= 0
                sustained = toward and speed >= threshold * C.THROW_RELEASE_FACTOR
                # Ducking, jumping or lunging moves the hands and feet fast too; a miss only counts as
                # thrown if the arm/leg itself moved fast. (A strike that lands always counts.)
                qualifies = player.limb_speed_vs_body(kind, side) >= threshold
                if kind == "kick" and qualifies:  # a fast step isn't a kick: the foot has to come up
                    other = feet.get("r" if side == "l" else "l")
                    qualifies = other is not None and \
                        other.center[1] - limb.center[1] >= C.KICK_THROW_MIN_LIFT * player.unit
                self.stats.update_limb(kind, side, striking, sustained, qualifies)

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

            sure_sweep = kick and part == "feet" and enemy.sweep_ready  # a leg sweep always connects
            if not sure_sweep and enemy.try_dodge():
                cooldowns[side] = C.KICK_COOLDOWN if kick else C.PUNCH_COOLDOWN
                self.effects.text("ROLLED AWAY" if enemy.rolling else "DODGED", (ec[0], ec[1] - enemy.H * 0.4),
                                  (200, 200, 255), 44)
                return
            power = min(1.0, (speed - threshold) / threshold)
            lo, hi = (C.KICK_DAMAGE_MIN, C.KICK_DAMAGE_MAX) if kick else (C.PUNCH_DAMAGE_MIN, C.PUNCH_DAMAGE_MAX)
            damage = round(lo + (hi - lo) * power)
            cooldowns[side] = C.KICK_COOLDOWN if kick else C.PUNCH_COOLDOWN
            hp_before = enemy.hp
            result = enemy.take_kick(damage, part) if kick else enemy.take_hit(damage)
            dealt = hp_before - enemy.hp
            pos = limb.center
            if dealt > 0:  # only attacks that actually hurt the boss count as landed
                self.stats.record_damage_dealt(dealt)
                if self.stats.limb_landed(kind, side):
                    self._combo_hit(pos)

            if result in ("blocked", "checked"):
                dealt = round(damage * C.ENEMY_KICK_BLOCK_DAMAGE)
                label = "CHECKED!" if result == "checked" else "BLOCKED"  # checked: sweep still recharging
                self.effects.burst(pos, (200, 150, 255), 16, 360)
                self.effects.text(f"{label} -{dealt}", (pos[0], pos[1] - 40), (200, 160, 255), 40)
                self.effects.shake(5)
                self.sfx.kick(0.5)
            else:
                self.effects.burst(pos, (255, 240, 200), 30 if kick else 22, 620 if kick else 520)
                self.effects.burst(pos, (255, 140, 40), 12, 320)
                self.effects.text(f"-{damage}", (pos[0], pos[1] - 40), (255, 220, 120), 48 if kick else 40)
                self.effects.shake((10 if kick else 6) + 6 * power)
                if kick:
                    self.sfx.kick(0.85 + 0.15 * power)
                else:
                    self.sfx.punch(0.7 + 0.3 * power)
            if result == "knockdown":
                how = "HEAD KICK!" if part == "head" else "LEG SWEEP!"
                self.effects.text(f"{how} KNOCKDOWN!", (ec[0], ec[1] - enemy.H * 0.45), (255, 200, 80), 60, 1.2)
            elif result == "ko":  # freeze-frame, slow motion, and he goes down (see enemy._update_death)
                sk = enemy.sk
                self.slowmo_t = C.KO_SLOWMO_TIME
                self.effects.smoke([sk[k] for k in ("head", "shoulder", "hip", "f_hand", "r_hand",
                                                    "f_elbow", "r_elbow", "f_knee", "r_knee",
                                                    "f_ankle", "r_ankle")], per_point=6)
                self.effects.burst(pos, (255, 255, 255), 50, 900, size=(3.0, 8.0))
                self.effects.burst(pos, (255, 180, 60), 30, 650, size=(3.0, 7.0))
                self.effects.flash((255, 255, 255), 200)
                self.effects.text("K.O.", (self.size[0] / 2, self.size[1] * 0.3), (255, 255, 255), 170, 2.0)
                self.effects.shake(26)
                (self.sfx.kick if kick else self.sfx.punch)(1.0)
            if not enemy.can_be_hit:
                return

    def _combo_hit(self, pos) -> None:
        """A new landed hit: grow the combo and celebrate it once it's 2+."""
        n = self.stats.combo_hit(self._now)
        if n < 2:
            return
        self.combo_pop_t = 0.0
        if n >= C.COMBO_WORDS[0][0]:  # top tier: make it feel huge
            self.effects.burst(pos, C.COMBO_WORDS[0][2], 26, 700, size=(3.0, 7.0))
            self.effects.flash(C.COMBO_WORDS[0][2], 60)

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

    def _track_enemy_melee(self) -> None:
        """One stats entry per melee strike (fist, foot, staff, sword...): opened when its hitbox goes
        live, closed when that window ends. Hit / block are reported by _resolve_enemy_strike; if
        neither happened by the close it was dodged (jumped, ducked, stepped away or out of range)."""
        e, stats = self.enemy, self.stats
        live = e.state is EnemyState.ATTACK and e.attack is not None and e.strike_window_open
        if self._melee is not None and (not live or self._melee[0] != e.attack_seq):
            stats.close_attack(self._melee[1])
            self._melee = None
        if live and self._melee is None:
            self._melee = (e.attack_seq, stats.open_attack())
        if self._melee is not None and not self.player.tracked:
            stats.void_attack(self._melee[1])

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
        attack_id = self._melee[1] if self._melee is not None else None
        pos = strike.b

        if not player.in_range(spec.min_depth):
            self.effects.text("OUT OF RANGE", (pos[0], pos[1] - 30), (120, 255, 160), 36)
            return
        if spec.unblockable and player.is_jumping:
            self.effects.text("JUMPED!", (pos[0], pos[1] - 40), (120, 255, 160), 44)
            self.effects.burst(pos, (180, 180, 200), 10, 200)
            return
        if contact == "shield":
            self.stats.attack_blocked(attack_id)
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
            self.stats.attack_blocked(attack_id, self._hurt_player(chip))
            enemy.on_blocked()
            self.effects.burst(pos, (120, 200, 255), 18, 380)
            self.effects.text("BLOCK" if chip < 0.5 else f"BLOCK -{chip:.0f}", (pos[0], pos[1] - 30),
                              (140, 210, 255), 40)
            self.effects.shake(4)
            self.sfx.punch(0.45)
            return

        damage = spec.damage * (C.HEAD_HIT_MULTIPLIER if contact == "head" else 1.0)
        self.stats.attack_hit(attack_id, self._hurt_player(damage))
        self.effects.burst(pos, (255, 60, 60), 20, 420)
        self.effects.text(f"{spec.label}! -{damage:.0f}", (pos[0], pos[1] - 30), (255, 90, 90), 42)
        self.effects.flash((180, 0, 0), 110)
        (self.sfx.kick if spec.limb == "front_foot" else self.sfx.punch)(1.0)  # his kicks and sweeps
        self.effects.shake(10 + damage * 0.5)

    # --- rendering -----------------------------------------------------------
    def _draw(self) -> None:
        world = self.world
        if self.background is not None:
            world.blit(self.background, (0, 0))
        else:
            world.fill((12, 10, 18))

        show_enemy = self.mode in (MODE_INTRO, MODE_FIGHT, MODE_OVER, MODE_SUMMARY)
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
        if self.low_health > 0:
            pulse = self.sfx.heartbeat_pulse(now)
            self._vignette.set_alpha(int(255 * min(1.0, (0.3 + 0.7 * pulse) * self.low_health)))
            self.screen.blit(self._vignette, (0, 0))
        if self.mode == MODE_MENU:
            self._draw_menu()
        elif self.mode == MODE_SUMMARY:
            self._draw_summary()
        else:
            self._draw_hud()
        if self.mode in (MODE_MENU, MODE_SUMMARY, MODE_INTRO):
            arcade.scanlines(self.screen)
        pygame.display.flip()

    def _dim(self, alpha: int) -> None:
        self._veil.set_alpha(alpha)
        self.screen.blit(self._veil, (0, 0))

    # --- arcade cabinet screens ------------------------------------------------------------
    def _top_bar(self) -> None:
        """1UP / HI-SCORE / FREE PLAY across the top, like a cabinet attract screen."""
        w = self.size[0]
        A, t = arcade, time.perf_counter()
        if int(t * 2) % 2 == 0:
            A.text(self.screen, "1UP", (40, 16), self.font_label, A.RED)
        A.text(self.screen, "HI-SCORE", (w / 2, 16), self.font_label, A.RED, "midtop")
        best = self.leaderboard.entries[0] if self.leaderboard.entries else None
        if best is not None:
            A.text(self.screen, f"{best['score']:,}  {best['name']}", (w / 2, 36), self.font_menu, A.WHITE, "midtop")
        A.text(self.screen, "FREE PLAY", (w - 40, 16), self.font_label, A.CYAN, "topright")

    def _draw_menu(self) -> None:
        w, h = self.size
        A, t = arcade, time.perf_counter()
        self._dim(205)
        self._top_bar()
        self.menu_rects = []
        items = self._menu_items()
        sub_page = self.menu_page != "main"  # boss list or high scores: compact logo, Esc goes back
        if not sub_page:
            self._draw_logo((w / 2, 150))
            A.text(self.screen, "YOUR BODY IS THE CONTROLLER.  EVERY ROUND IS A WORKOUT.", (w / 2, 274),
                   self.font, A.DIM, "center")
            top, gap = 326, 54
        else:
            A.text(self.screen, "ARENA: FIT FIGHTER", (w / 2, 96), self.font_stat, A.YELLOW, "center", shadow=A.MAGENTA)
            if self.menu_page == "bosses":
                A.text(self.screen, "CHOOSE YOUR OPPONENT", (w / 2, 158), self.font_head, A.CYAN, "center")
                top, gap = 222, 60
            else:
                A.text(self.screen, "HIGH SCORES", (w / 2, 158), self.font_head, A.CYAN, "center")
                self._draw_board(200)
                top, gap = 618, 60
        for i, (label, _) in enumerate(items):
            rect = pygame.Rect(0, 0, 470, 46)
            rect.center = (w // 2, int(top) + i * gap)
            self._button(rect, label, i == self.menu_index)
            self.menu_rects.append(rect)

        if not sub_page:
            tips = ["STAND ABOUT 6 FT BACK, WHOLE BODY IN VIEW.  PUNCH & KICK AT THE BOSS, ARMS UP TO BLOCK.",
                    "JUMP THE LOW SWEEPS.  DUCK THE STARS.  STEP BACK TO GET OUT OF REACH."]
            for i, tip in enumerate(tips):
                A.text(self.screen, tip, (w / 2, 612 + i * 26), self.font_small, A.WHITE, "center")
            if int(t * 1.6) % 2 == 0:
                A.text(self.screen, "PRESS ENTER TO SWEAT", (w / 2, 676), self.font_menu, A.YELLOW, "center")
        hint = "UP/DOWN + ENTER (OR CLICK)    M MUTE    F FULLSCREEN    " + ("ESC BACK" if sub_page else "ESC QUIT")
        A.text(self.screen, hint, (w / 2, h - 12), self.font_tiny, A.DIM, "midbottom")
        if self.camera.error:
            A.text(self.screen, f"CAMERA ERROR: {self.camera.error}", (w / 2, h - 36), self.font_small, A.RED,
                   "midbottom")

    def _draw_logo(self, center) -> None:
        """Big extruded 'ARENA' (AR picked out) over 'FIT FIGHTER', with a slapped-on sticker."""
        A = arcade
        cx, cy = center
        word = [("AR", A.CYAN), ("ENA", A.YELLOW)]
        width = sum(self.font_title.size(s)[0] for s, _ in word)
        x = cx - width / 2
        for s, color in word:
            for k in range(8, 0, -1):  # chunky 3D extrude, darker the further back
                shade = tuple(int(c * (0.35 + 0.05 * (8 - k))) for c in A.MAGENTA)
                self.screen.blit(self.font_title.render(s, False, shade), (x + k, cy - 52 + k))
            self.screen.blit(self.font_title.render(s, False, color), (x, cy - 52))
            x += self.font_title.size(s)[0]
        A.text(self.screen, "FIT FIGHTER", (cx, cy + 78), self.font_big, A.WHITE, "center", shadow=A.MAGENTA, depth=5)
        A.sticker(self.screen, "NO CONTROLLER NEEDED!", (cx + width / 2 + 60, cy - 44), self.font_label,
                  angle=-9 + 1.5 * math.sin(time.perf_counter() * 3))

    def _draw_board(self, top: float, highlight=None) -> None:
        """High-score table (best first), each place in its own colour; `highlight` (1-based) blinks."""
        A, t = arcade, time.perf_counter()
        w = self.size[0]
        entries = self.leaderboard.entries
        if not entries:
            A.text(self.screen, "NO SCORES YET.  BE THE FIRST!", (w / 2, top + 80), self.font_menu, A.WHITE, "center")
            return
        x0 = w / 2 - 360
        cols = ((x0, "midleft"), (x0 + 120, "midleft"), (x0 + 400, "midright"), (x0 + 470, "center"),
                (x0 + 720, "midright"))
        for (cx, anchor), label in zip(cols, ("RANK", "NAME", "SCORE", "GRADE", "BOSS")):
            A.text(self.screen, label, (cx, top), self.font_label, A.DIM, anchor)
        for i, e in enumerate(entries):
            y = top + 34 + i * 34
            color = A.ROW_COLORS[min(i, len(A.ROW_COLORS) - 1)]
            if highlight == i + 1:
                pygame.draw.rect(self.screen, A.PANEL, (x0 - 16, y - 16, 752, 32))
                color = A.WHITE if int(t * 6) % 2 == 0 else A.YELLOW
            values = (A.ordinal(i + 1), e["name"], f"{e['score']:,}", e["rank"], e["boss"])
            for j, ((cx, anchor), v) in enumerate(zip(cols, values)):
                A.text(self.screen, v, (cx, y), self.font_menu if j < 4 else self.font_label, color, anchor)

    def _draw_popup(self) -> None:
        """Over the results: type your initials, then see where you landed on the high-score table."""
        popup = self._summary_popup()
        if popup not in ("entry", "board"):
            return
        A = arcade
        w, h = self.size
        self.menu_rects = []  # the results buttons are covered
        self._dim(170)
        box = pygame.Rect(0, 0, 820 if popup == "board" else 640, 500 if popup == "board" else 380)
        box.center = (w // 2, h // 2)
        A.panel(self.screen, box, A.YELLOW)
        if popup == "board":
            A.text(self.screen, "HIGH SCORES", (w / 2, box.top + 40), self.font_head, A.CYAN, "center")
            self._draw_board(box.top + 82, self.board_pos)
            if int(self.summary_t * 2) % 2 == 0:
                A.text(self.screen, f"YOU PLACED {A.ordinal(self.board_pos)}!   PRESS ENTER", (w / 2, box.bottom - 30),
                       self.font_label, A.WHITE, "center")
            return
        score = self.stats.fitness_score()
        place = self.leaderboard.position(score)
        A.text(self.screen, "NEW HIGH SCORE!", (w / 2, box.top + 48), self.font_head, A.rainbow(self.summary_t),
               "center")
        A.text(self.screen, f"{score:,}", (w / 2, box.top + 108), self.font_stat, A.WHITE, "center")
        A.text(self.screen, f"THAT'S {A.ordinal(place)} PLACE.  ENTER YOUR INITIALS:", (w / 2, box.top + 160),
               self.font, A.CYAN, "center")
        n = C.LEADERBOARD_NAME_LEN
        blink = int(self.summary_t * 3) % 2 == 0
        for i in range(n):
            cx, cy = int(w / 2 + (i - (n - 1) / 2) * 110), box.top + 245
            if i < len(self.name_entry):
                A.text(self.screen, self.name_entry[i], (cx, cy), self.font_big, A.YELLOW, "center")
            active = i == len(self.name_entry)
            if not active or blink:
                pygame.draw.rect(self.screen, A.YELLOW if active else A.WHITE, (cx - 34, cy + 36, 68, 8))
        A.text(self.screen, "TYPE LETTERS    BACKSPACE ERASE    ENTER SAVE    ESC SKIP", (w / 2, box.bottom - 28),
               self.font_small, A.DIM, "center")

    def _button(self, rect: pygame.Rect, label: str, selected: bool) -> None:
        arcade.button(self.screen, rect, label, selected, time.perf_counter(), self.font_menu)

    def _draw_summary(self) -> None:
        """Results screen: workout and fight stats, the score tally, grades and a rank stamp. Numbers count up."""
        A = arcade
        w, h = self.size
        s = self.stats
        self._dim(225)
        k = min(1.0, self.summary_t / C.SUMMARY_TALLY_TIME)
        k = 1.0 - (1.0 - k) ** 3  # ease out: numbers race up, then settle

        def num(v) -> str:
            return f"{round(v * k):,}"

        def pct(v) -> str:
            return f"{round(v * k)}%"

        def grade(v):  # good / ok / poor percentages
            return A.LIME if v >= 70 else A.YELLOW if v >= 40 else A.ORANGE

        title, color = {"VICTORY": ("STAGE CLEAR!", A.LIME), "CHAMPION": ("CHAMPION!", A.YELLOW)} \
            .get(s.result, ("GAME OVER", A.RED))
        A.text(self.screen, title, (w / 2, 44), self.font_big, color, "center", shadow=A.MAGENTA, depth=5)
        boss_title = next((b.get("title", "") for b in C.BOSSES if b["name"] == s.boss_name), "")
        A.text(self.screen, f"VS {s.boss_name}" + (f"  -  {boss_title}" if boss_title else ""), (w / 2, 92),
               self.font, A.DIM, "center")
        line = A.coach_line(s)
        cw = self.font_label.size("COACH:")[0] + 16 + self.font.size(line)[0]
        x = w / 2 - cw / 2
        A.text(self.screen, "COACH:", (x, 128), self.font_label, A.MAGENTA, "midleft")
        A.text(self.screen, line, (x + self.font_label.size("COACH:")[0] + 16, 128), self.font, A.WHITE, "midleft")

        top, ph = 158, 432

        def box(x, pw, heading, border):
            rect = pygame.Rect(x, top, pw, ph)
            A.panel(self.screen, rect, border)
            A.text(self.screen, heading, (rect.centerx, rect.top + 30), self.font_menu, border, "center")
            return rect

        def rows(rect, items, y0=64, step=34):
            y = rect.top + y0
            for label, value, value_color in items:
                lw = self.font_row.size(label)[0]
                vw = self.font_row.size(value)[0]
                dots = "." * max(2, int((rect.width - 52 - lw - vw) / self.font_row.size(".")[0]))
                A.text(self.screen, label, (rect.left + 24, y), self.font_row, A.WHITE, "midleft")
                A.text(self.screen, dots, (rect.left + 28 + lw, y), self.font_row, (70, 64, 110), "midleft", shadow=None)
                A.text(self.screen, value, (rect.right - 24, y), self.font_row, value_color, "midright")
                y += step

        r = box(36, 390, "WORKOUT", A.ORANGE)
        rows(r, [
            ("TIME", format_duration(s.duration() * k), A.WHITE),
            ("PUNCHES THROWN", num(s.punches_thrown), A.CYAN),
            ("PUNCHES LANDED", num(s.punches_landed), A.CYAN),
            ("PUNCH ACCURACY", pct(s.punch_accuracy), grade(s.punch_accuracy)),
            ("KICKS THROWN", num(s.kicks_thrown), A.CYAN),
            ("KICKS LANDED", num(s.kicks_landed), A.CYAN),
            ("KICK ACCURACY", pct(s.kick_accuracy), grade(s.kick_accuracy)),
            ("BEST COMBO", f"x{num(s.best_combo)}", A.YELLOW),
            ("EST. CALORIES", f"{num(s.calories())} KCAL", A.YELLOW),
        ])
        A.text(self.screen, f"KCAL IS A ROUGH GUESS (ASSUMES {C.CALORIE_WEIGHT_KG:.0f} KG)",
               (r.centerx, r.bottom - 26), self.font_tiny, A.DIM, "center")

        health = s.health_remaining * 100
        r = box(444, 390, "FIGHT", A.CYAN)
        rows(r, [
            ("TOTAL HITS", num(s.total_hits), A.YELLOW),
            ("DAMAGE DEALT", num(s.damage_dealt), A.CYAN),
            ("ATTACKS FACED", num(s.enemy_attacks_faced), A.CYAN),
            ("DODGED", num(s.enemy_attacks_dodged), A.LIME),
            ("BLOCKED", num(s.enemy_attacks_blocked), A.CYAN),
            ("GOT HIT", num(s.enemy_attacks_hit), A.RED),
            ("DODGE RATE", pct(s.dodge_rate), grade(s.dodge_rate)),
            ("BEST DODGE STREAK", num(s.longest_dodge_streak), A.LIME),
            ("DAMAGE TAKEN", num(s.damage_taken), A.RED),
            ("HEALTH LEFT", pct(health), grade(health)),
        ], step=33)

        r = box(852, 392, "SCORE", A.YELLOW)
        tally = [("POINTS", num(s.points()), A.WHITE)] + [(label, f"+{num(v)}", A.LIME) for label, v in s.bonuses()]
        if not s.won:  # show what a win would have added
            tally.append(("WIN BONUS", "--", A.DIM))
        rows(r, tally, y0=62, step=30)
        y = r.top + 62 + 30 * len(tally) + 2
        pygame.draw.line(self.screen, A.YELLOW, (r.left + 24, y), (r.right - 24, y), 3)
        A.text(self.screen, "TOTAL", (r.left + 24, y + 30), self.font_label, A.YELLOW, "midleft")
        A.text(self.screen, num(s.fitness_score()), (r.right - 24, y + 30), self.font_stat, A.YELLOW, "midright")
        cats = s.category_scores()
        gy = r.top + 286
        for i, (key, label) in enumerate((("accuracy", "ACC"), ("dodging", "DODGE"), ("activity", "MOVE"),
                                          ("combos", "COMBO"))):
            cx = r.left + 52 + i * 96
            letter = rank(cats[key])
            A.text(self.screen, letter, (cx, gy), self.font_grade, RANK_COLORS[letter], "center")
            A.text(self.screen, label, (cx, gy + 28), self.font_tiny, A.DIM, "center")
        overall = s.overall_rank()
        A.text(self.screen, "RANK", (r.left + 40, r.bottom - 54), self.font_menu, A.WHITE, "midleft")
        stamp_t = self.summary_t - C.SUMMARY_TALLY_TIME
        if stamp_t > 0:  # the rank stamps down once the numbers have counted up
            pop = max(0.0, 1.0 - stamp_t / 0.18)
            self._text_scaled(overall, (r.right - 90, r.bottom - 64), self.font_rank, RANK_COLORS[overall],
                              1.0 + 1.2 * pop, 1.0 - 0.6 * pop)

        self.menu_rects = []
        items = self._summary_items()
        bw = [max(200, self.font_menu.size(label)[0] + 48) for label, _ in items]
        x = (w - (sum(bw) + 40 * (len(items) - 1))) // 2
        for i, (label, _) in enumerate(items):
            rect = pygame.Rect(x, top + ph + 22, bw[i], 46)
            self._button(rect, label, i == self.menu_index)
            self.menu_rects.append(rect)
            x += bw[i] + 40
        A.text(self.screen, "LEFT/RIGHT + ENTER (OR CLICK)    R REMATCH    C RECALIBRATE    ESC MENU",
               (w / 2, h - 8), self.font_tiny, A.DIM, "midbottom")
        self._draw_popup()

    def _text_scaled(self, msg, center, font, color, scale: float = 1.0, alpha: float = 1.0) -> None:
        """Centred outlined text, scaled and faded (for pops and stamps)."""
        surf = render_outlined(font, msg, color)
        if abs(scale - 1.0) > 1e-3:
            surf = pygame.transform.scale(surf, (max(1, int(surf.get_width() * scale)),
                                                 max(1, int(surf.get_height() * scale))))
        surf.set_alpha(int(255 * max(0.0, min(1.0, alpha))))
        self.screen.blit(surf, surf.get_rect(center=(int(center[0]), int(center[1]))))

    def _draw_countdown_digit(self, remaining: float) -> None:
        w, h = self.size
        n = max(1, math.ceil(remaining))
        frac = n - remaining  # 0 -> 1 within each second
        self._text_scaled(str(n), (w // 2, int(h * 0.5)), self.font_count, arcade.YELLOW, 1.0 + 0.5 * (1 - frac),
                          min(1.0, 1.4 - frac))

    def _warning_sign(self, center, size: float) -> None:
        cx, cy = center
        pts = [(cx, cy - size * 0.6), (cx + size * 0.62, cy + size * 0.5), (cx - size * 0.62, cy + size * 0.5)]
        pygame.draw.polygon(self.screen, (255, 210, 40), pts)
        pygame.draw.polygon(self.screen, (40, 20, 0), pts, 3)
        pygame.draw.line(self.screen, (40, 20, 0), (cx, cy - size * 0.22), (cx, cy + size * 0.18),
                         max(3, int(size * 0.1)))
        pygame.draw.circle(self.screen, (40, 20, 0), (int(cx), int(cy + size * 0.33)), max(2, int(size * 0.06)))

    def _draw_intro(self) -> None:
        """WARNING / '<BOSS> APPROACHES!', then his name and title, then 3-2-1."""
        A = arcade
        w, h = self.size
        t = self.intro_t
        boss = self.boss
        warn_end = C.INTRO_WARNING_TIME
        title_end = warn_end + C.INTRO_TITLE_TIME
        if t < warn_end:
            pulse = 0.5 + 0.5 * math.sin(t * math.tau * 2.5)
            self._tint.set_alpha(int(35 + 55 * pulse))
            self.screen.blit(self._tint, (0, 0))
            for y in (int(h * 0.29), int(h * 0.60)):  # hazard stripes
                band = pygame.Rect(0, y, w, 16)
                pygame.draw.rect(self.screen, (255, 210, 40), band)
                for x in range(-40 + int(t * 160) % 40, w, 40):
                    pygame.draw.polygon(self.screen, (30, 20, 0), [(x, band.bottom), (x + 16, band.top),
                                                                 (x + 32, band.top), (x + 16, band.bottom)])
            if int(t * 6) % 2 == 0 or t > warn_end - 0.35:
                A.text(self.screen, "WARNING", (w / 2, h * 0.41), self.font_huge, A.RED, "center", depth=6)
            msg = f"{boss['name']} APPROACHES!"
            mw = self.font_menu.size(msg)[0]
            A.text(self.screen, msg, (w / 2, h * 0.535), self.font_menu, A.YELLOW, "center")
            for side in (-1, 1):
                self._warning_sign((w / 2 + side * (mw / 2 + 44), h * 0.535), 40)
        elif t < title_end:
            u = (t - warn_end) / C.INTRO_TITLE_TIME
            appear = min(1.0, u / 0.2)
            name_color = tuple(min(255, c + 100) for c in boss["rim"])
            A.text(self.screen, f"ROUND {self.boss_index + 1} / {len(C.BOSSES)}", (w / 2, h * 0.27), self.font_menu,
                   A.CYAN, "center")
            self._text_scaled(boss["name"], (w / 2, h * 0.39), self.font_huge, name_color,
                              1.0 + 0.35 * (1.0 - appear), appear)
            title = boss.get("title", "")
            if title and u > 0.15:
                a = min(1.0, (u - 0.15) / 0.2)
                tw = self.font_head.size(title)[0]
                for side in (-1, 1):
                    x0 = w / 2 + side * (tw / 2 + 20)
                    pygame.draw.rect(self.screen, A.YELLOW, (min(x0, x0 + side * 90 * a), h * 0.505 - 3, 90 * a, 6))
                self._text_scaled(title, (w / 2, h * 0.505), self.font_head, A.YELLOW, 1.0, a)
        else:
            A.text(self.screen, f"ROUND {self.boss_index + 1}:  {boss['name']}", (w / 2, h * 0.14), self.font_head,
                   A.WHITE, "center")
            A.text(self.screen, "GET READY!", (w / 2, h * 0.21), self.font, A.CYAN, "center")
            self._draw_countdown_digit(C.INTRO_WARNING_TIME + C.INTRO_TITLE_TIME + C.COUNTDOWN_SECONDS - t)
        A.text(self.screen, "ENTER: SKIP", (w - 24, h - 16), self.font_tiny, A.DIM, "bottomright")

    def _draw_combo(self) -> None:
        """'4 HIT COMBO / AMAZING!' near the top while a combo is alive; pops on every new hit."""
        s = self.stats
        if s.combo < 2 or s.last_hit_at is None:
            return
        age = self._now - s.last_hit_at
        alpha = 1.0 if age <= C.COMBO_WINDOW else 1.0 - (age - C.COMBO_WINDOW) / 0.4
        if alpha <= 0:
            return
        _, word, color = next(c for c in C.COMBO_WORDS if s.combo >= c[0])
        pop = 1.0 + 0.55 * max(0.0, 1.0 - self.combo_pop_t / 0.18)
        size = 64 + 8 * min(s.combo - 2, 5)
        w, h = self.size
        self._text_scaled(f"{s.combo} HIT COMBO", (w / 2, h * 0.21), self.effects.font(size), color, pop, alpha)
        self._text_scaled(word, (w / 2, h * 0.21 + size * 0.75 * pop), self.effects.font(int(size * 0.7)), color,
                          pop, alpha)

    def _text(self, msg, pos, font=None, color=(255, 255, 255), anchor="topleft") -> None:
        arcade.text(self.screen, msg, pos, font or self.font, color, anchor)

    def _draw_hud(self) -> None:
        A, t = arcade, time.perf_counter()
        w, h = self.size
        cam_err = self.camera.error

        if self.mode in (MODE_FIGHT, MODE_OVER):
            bw = int(w * 0.36)
            bar_color = A.CYAN
            if self.low_health > 0:  # your bar throbs red with the heartbeat
                k = self.sfx.heartbeat_pulse(t)
                bar_color = tuple(int(a + (b - a) * k) for a, b in zip(A.RED, (255, 200, 200)))
            A.seg_bar(self.screen, pygame.Rect(40, 34, bw, 22), self.player.hp / C.PLAYER_MAX_HP,
                      self.trail_player / C.PLAYER_MAX_HP, bar_color)
            A.seg_bar(self.screen, pygame.Rect(w - 40 - bw, 34, bw, 22), self.enemy.hp / self.enemy.max_hp,
                      self.trail_enemy / self.enemy.max_hp, A.MAGENTA, rtl=True)
            A.text(self.screen, "1UP  YOU", (36, 72), self.font_label, A.CYAN)
            A.text(self.screen, self.enemy.name, (w - 36, 72), self.font_label, A.MAGENTA, "topright")
            A.text(self.screen, f"ROUND {self.boss_index + 1}", (w / 2, 18), self.font_label, A.WHITE, "midtop")
            A.text(self.screen, f"{self.stats.points():06d}", (w / 2, 40), self.font_menu, A.YELLOW, "midtop")
            in_range = self.player.in_range(C.PLAYER_PUNCH_MIN_DEPTH)
            A.text(self.screen, "IN REACH" if in_range else "STEP IN", (36, 96), self.font_tiny,
                   A.LIME if in_range else A.ORANGE)
            if self.player.is_jumping:
                A.text(self.screen, "AIRBORNE!", (36, 116), self.font_tiny, A.CYAN)
            if self.enemy.sweep_ready:  # leg sweep: always knocks him down, then recharges
                A.text(self.screen, "LEG SWEEP READY", (36, 136), self.font_tiny, A.YELLOW)
            else:
                A.text(self.screen, f"LEG SWEEP {math.ceil(self.enemy.sweep_cooldown)}", (36, 136), self.font_tiny,
                       A.DIM)

        if self.debug:  # tech readout only in debug mode
            stats = (f"FPS {self.pacer.fps:4.0f}   CAM {self.camera.camera_fps:4.1f}   POSE {self.pose_ms:4.1f} ms   "
                     f"DEPTH {self.player.depth_ratio:.2f}   [D] debug ON")
            A.text(self.screen, stats, (14, h - 30), self.font_small, A.WHITE)
        elif self.mode in (MODE_FIGHT, MODE_OVER):
            A.text(self.screen, "ESC MENU   R RESTART   M MUTE", (w - 16, h - 12), self.font_tiny, A.DIM, "bottomright")

        if cam_err:
            A.text(self.screen, "CAMERA PROBLEM", (w / 2, h / 2 - 40), self.font_big, A.RED, "center")
            A.text(self.screen, cam_err, (w / 2, h / 2 + 30), self.font, A.WHITE, "center")
            return
        if self.background is None:
            A.text(self.screen, "WARMING UP THE CAMERA...", (w / 2, h / 2), self.font_head, A.WHITE, "center")
            return

        if self.mode == MODE_COUNTDOWN:
            A.text(self.screen, "GET IN POSITION!", (w / 2, h * 0.12), self.font_big, A.YELLOW, "center",
                   shadow=A.MAGENTA)
            A.text(self.screen, "STAND ABOUT 6 FT (2 M) BACK, WHOLE BODY IN VIEW", (w / 2, h * 0.21), self.font,
                   A.WHITE, "center")
            self._draw_countdown_digit(self.countdown_t)
        elif self.mode == MODE_INTRO:
            self._draw_intro()
        elif self.mode == MODE_CALIBRATE:
            A.text(self.screen, "HOLD STILL...", (w / 2, h * 0.12), self.font_big, A.CYAN, "center", shadow=A.MAGENTA)
            A.text(self.screen, "SCANNING YOUR FIGHTER.  WHOLE BODY IN VIEW, PLEASE.", (w / 2, h * 0.21), self.font,
                   A.WHITE, "center")
            if not self.player.tracked and int(t * 2) % 2 == 0:
                A.text(self.screen, "CAN'T SEE YOU - STEP INTO VIEW!", (w / 2, h * 0.27), self.font, A.ORANGE,
                       "center")
            bw = int(w * 0.4)
            p = self.player.calibration_progress
            A.seg_bar(self.screen, pygame.Rect(int(w / 2 - bw / 2), int(h * 0.31), bw, 18), p, p, A.LIME)
        elif self.mode == MODE_FIGHT and not self.player.tracked:
            if int(t * 2) % 2 == 0:
                A.text(self.screen, "STEP BACK INTO VIEW!", (w / 2, h * 0.45), self.font_big, A.ORANGE, "center",
                       shadow=A.INK)
        elif self.mode == MODE_FIGHT:
            self._draw_combo()
        elif self.mode == MODE_OVER and self.over_timer > (0.8 if self.result == "DEFEAT" else 2.2):
            if self.result == "VICTORY":
                A.text(self.screen, "YOU WIN!", (w / 2, h * 0.4), self.font_huge, A.LIME, "center", shadow=A.MAGENTA)
                nxt = C.BOSSES[self.boss_index + 1]["name"]
                A.text(self.screen, f"NEXT UP: {nxt}", (w / 2, h * 0.52), self.font_menu, A.WHITE, "center")
            elif self.result == "CHAMPION":
                A.text(self.screen, "ALL BOSSES DOWN!", (w / 2, h * 0.4), self.font_big, A.YELLOW, "center",
                       shadow=A.MAGENTA)
                A.text(self.screen, "YOU ARE THE CHAMPION", (w / 2, h * 0.52), self.font_menu, A.WHITE, "center")
            else:
                A.text(self.screen, "GAME OVER", (w / 2, h * 0.4), self.font_huge, A.RED, "center", shadow=A.INK)
                A.text(self.screen, f"{self.enemy.name} TAKES THIS ONE", (w / 2, h * 0.52), self.font_menu, A.WHITE,
                       "center")
            if int(t * 2) % 2 == 0:
                A.text(self.screen, "PRESS ENTER", (w / 2, h * 0.6), self.font_label, A.YELLOW, "center")


def main() -> None:
    parser = argparse.ArgumentParser(description="ARena: Fit Fighter")
    parser.add_argument("--camera", type=int, default=C.CAMERA_INDEX, help="webcam index")
    parser.add_argument("--complexity", type=int, choices=(0, 1, 2), default=C.POSE_MODEL_COMPLEXITY,
                        help="MediaPipe Pose model complexity (0 = fastest)")
    parser.add_argument("--debug", action="store_true", help="start with the hitbox overlay on")
    args = parser.parse_args()
    Game(args.camera, args.complexity, args.debug).run()


if __name__ == "__main__":
    main()
