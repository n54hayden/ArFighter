"""Tunable constants.

Player distances are in "torso units" (shoulder-midpoint to hip-midpoint length),
so colliders and speed thresholds scale automatically with distance from the camera.
Enemy distances are fractions of the enemy's height H.
"""

# --- Display -----------------------------------------------------------------
SCREEN_W = 1280
SCREEN_H = 720
FPS = 60
WINDOW_TITLE = "AR Shadow Fighter"

# --- Music -------------------------------------------------------------------
MUSIC_DIR = "music"               # relative to ar_fighter.py; first .mp3/.ogg/.wav is played
MUSIC_VOLUME = 0.6                # 0.0 - 1.0
MUSIC_FADE_IN_MS = 500
MUSIC_FADE_OUT_MS = 1200

# --- Sound effects -----------------------------------------------------------
SFX_DIR = "sound_effects"         # relative to ar_fighter.py
SFX_VOLUME = 0.9                  # 0.0 - 1.0
SFX_KEYWORDS = {                  # a file is used for a role if its name contains any keyword
    "bell": ("bell",),
    "punch": ("punch", "impact", "hit"),
    "cheer": ("cheer", "crowd", "applause"),
    "throw": ("throw", "whoosh", "swish"),
}

# --- Camera / pose thread ----------------------------------------------------
# On this Mac, index 0 is the iPhone (Continuity Camera) when it's nearby and 1 is the
# built-in FaceTime camera. With no phone around, the FaceTime camera becomes 0, so
# the camera thread falls back to 0 when 1 doesn't exist.
CAMERA_INDEX = 1
CAMERA_W, CAMERA_H, CAMERA_FPS = 1280, 720, 30
POSE_INPUT_WIDTH = 640            # frame is downscaled to this before MediaPipe
POSE_MODEL_COMPLEXITY = 1         # 0 = fastest, 2 = most accurate
POSE_MIN_DETECTION_CONFIDENCE = 0.5
POSE_MIN_TRACKING_CONFIDENCE = 0.5
BACKGROUND_BRIGHTNESS = 0.6       # darken the camera feed for the "shadow" mood

# --- Tracking ----------------------------------------------------------------
LANDMARK_VISIBILITY = 0.5         # ignore landmarks below this visibility
LANDMARK_SMOOTHING = 0.7          # EMA weight of the newest sample (1 = no smoothing)
TRACKING_LOST_TIMEOUT = 0.5       # seconds without a pose before the player is "lost"
CALIBRATION_SECONDS = 2.5         # seconds of standing still to record the baseline
DEPTH_SMOOTHING = 0.35
HORIZON_Y_FRAC = 0.5              # assumed camera horizon (0.5 = camera roughly level)

# --- Player colliders (torso units) ------------------------------------------
HEAD_RADIUS = 0.36
TORSO_RADIUS = 0.30
FOREARM_RADIUS = 0.14
FIST_RADIUS = 0.20
FOOT_RADIUS = 0.24
LEG_RADIUS = 0.16

# --- Player offence ----------------------------------------------------------
PUNCH_SPEED_THRESHOLD = 3.5       # torso units / second for a fist to count as a punch
PUNCH_COOLDOWN = 0.30             # per fist, prevents one punch registering many times
PUNCH_DAMAGE_MIN = 4
PUNCH_DAMAGE_MAX = 10
PLAYER_PUNCH_MIN_DEPTH = 0.80     # step back past this and your punches/kicks can't reach
KICK_SPEED_THRESHOLD = 4.0        # torso units / second for a foot to count as a kick
KICK_COOLDOWN = 0.5
KICK_DAMAGE_MIN = 8
KICK_DAMAGE_MAX = 16
LOW_KICK_SPEED_FACTOR = 1.3       # kicks to the boss's feet need this much more speed (so steps don't count)

# --- Player defence ----------------------------------------------------------
PLAYER_MAX_HP = 100
HEAD_HIT_MULTIPLIER = 1.25
JUMP_THRESHOLD = 0.20             # ground signal must rise this many torso units
JUMP_RELEASE = 0.08               # ...and drop back below this to land
JUMP_GRACE = 0.15                 # seconds a jump still counts after landing (latency)
GROUND_ADAPT_TAU = 1.0            # seconds; how fast the floor baseline drifts while grounded

# --- Enemy (shared by every boss) --------------------------------------------
ENEMY_WALK_SPEED = 0.8            # body heights / second
ENEMY_AIM_OFFSET = 0.15           # aim strikes this many torso units in front of the target's centre
ENEMY_FIRST_ATTACK_DELAY = 1.0
ENEMY_APPROACH_TIMEOUT = 2.5
ENEMY_COMBO_CHANCE = 0.5          # chance a jab is followed straight away by a cross
ENEMY_STUN_TIME = 0.55
ENEMY_STUN_IMMUNITY = 0.6         # after a stun, hits still hurt but don't re-stun
ENEMY_KNOCKBACK = 0.9             # body heights / second
ENEMY_KICK_BLOCK_DAMAGE = 0.25    # fraction of kick damage that gets through a block
ENEMY_FALL_TIME = 0.4             # knockdown: falling...
ENEMY_DOWN_TIME = 1.2             # ...lying on the ground (can't be hit)...
ENEMY_GETUP_TIME = 0.6            # ...and getting back up
ENEMY_KNOCKDOWN_KNOCKBACK = 1.4   # body heights / second
ENEMY_DODGE_TIME = 0.35           # how long a dodge lasts (can't be hit meanwhile)
ENEMY_DODGE_SPEED = 2.2           # body heights / second, backward
ENEMY_ROLL_TIME = 0.6             # backward roll (ninja): duration...
ENEMY_ROLL_SPEED = 1.6            # ...and speed in body heights / second
ENEMY_KICK_KNOCKBACK = 1.8        # body kick: pushed back (body heights / second) but stays on its feet
ENEMY_POSE_BLEND = 16.0

# --- Bosses (fought in order) -------------------------------------------------
# dodge_chance / kick_block_chance only apply while the boss is idle or walking,
# never mid-attack, stunned or knocked down.
BOSSES = [
    {
        "name": "SHADOW",
        "max_hp": 200,
        "dodge_chance": 0.10,
        "roll_chance": 0.0,       # fraction of dodges done as a backward roll instead of a hop back
        "dodge_cooldown": 1.0,    # minimum seconds between dodges
        "kick_block_chance": 0.30,
        "attack_speed": 1.0,      # > 1 = faster attacks
        "idle_min": 0.25,         # pause between attacks (shrinks further as the boss gets hurt)
        "idle_max": 0.7,
        "attack_weights": {"jab": 3.0, "cross": 2.0, "uppercut": 1.5, "high_kick": 1.5, "low_sweep": 2.0},
        "rim": (95, 65, 150),
        "headband": None,
    },
    {
        "name": "SHADOW NINJA",
        "max_hp": 300,
        "dodge_chance": 0.55,
        "roll_chance": 0.6,
        "dodge_cooldown": 0.7,
        "kick_block_chance": 0.35,
        "attack_speed": 1.1,
        "idle_min": 0.25,
        "idle_max": 0.6,
        "attack_weights": {"jab": 2.0, "cross": 2.0, "uppercut": 1.5, "high_kick": 1.5, "low_sweep": 1.5,
                           "throwing_star": 3.0},
        "rim": (170, 40, 55),
        "headband": (200, 30, 40),
    },
    {
        "name": "NINJA MONK",
        "max_hp": 400,
        "dodge_chance": 0.35,
        "roll_chance": 0.3,
        "dodge_cooldown": 0.8,
        "kick_block_chance": 0.40,
        "attack_speed": 1.1,
        "idle_min": 0.25,
        "idle_max": 0.6,
        "attack_weights": {"staff_swipe": 2.5, "staff_ground": 2.0, "staff_high": 2.0, "staff_poke": 2.0,
                           "high_kick": 1.0},
        "rim": (235, 150, 40),
        "headband": None,
        "staff": True,            # fights with a staff (arms hold it; it has its own hitbox)
    },
]

# --- Staff (Ninja Monk) ------------------------------------------------------------
STAFF_PARRY_STUN = 0.5            # seconds the monk staggers when you hit his staff mid-attack
STAFF_COLOR = (42, 26, 14)
STAFF_CAP_COLOR = (190, 150, 70)

# --- Throwing stars -----------------------------------------------------------
STAR_SPEED = 1.25                 # body heights / second
STAR_RADIUS = 0.035               # fraction of the boss's height
STAR_DAMAGE = 12
STAR_THROW_DISTANCE = 0.38        # boss backs off to this fraction of the screen width to throw

# --- Flow ---------------------------------------------------------------------
COUNTDOWN_SECONDS = 3
NEXT_BOSS_DELAY = 3.5             # seconds of celebration before the next boss's countdown

# --- Colours -----------------------------------------------------------------
ENEMY_BODY = (8, 6, 12)
ENEMY_BACK = (30, 26, 40)
ENEMY_FLASH = (235, 235, 245)
ENEMY_EYES = (230, 235, 255)
ENEMY_EYES_ANGRY = (255, 70, 40)
PLAYER_BAR = (70, 200, 255)
ENEMY_BAR = (200, 60, 230)
