"""Tunable constants.

Player distances are in "torso units" (shoulder-midpoint to hip-midpoint length),
so colliders and speed thresholds scale automatically with distance from the camera.
Enemy distances are fractions of the enemy's height H.
"""
import sys

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
CAMERA_INDEX = 1 if sys.platform == "darwin" else 0  # elsewhere the built-in webcam is 0
CAMERA_W, CAMERA_H, CAMERA_FPS = 1280, 720, 30
POSE_INPUT_WIDTH = 640            # frame is downscaled to this before MediaPipe
POSE_MODEL_COMPLEXITY = 1         # 0 = fastest, 2 = most accurate
POSE_MIN_DETECTION_CONFIDENCE = 0.5
POSE_MIN_TRACKING_CONFIDENCE = 0.5
BACKGROUND_BRIGHTNESS = 0.6       # darken the camera feed for the "shadow" mood
CAMERA_HFOV_DEG = 65.0            # horizontal field of view of a typical webcam (sets the depth scale)
PERSON_MASK_THRESHOLD = 0.5       # MediaPipe segmentation value treated as "you" (for occlusion)

# --- Floor mapping -----------------------------------------------------------
# The shadow fights on one lane: the floor row where you stood during calibration, at your
# calibrated size, moving only left / right. A SegFormer model (ADE20K classes) marks which
# pixels are floor; along the lane, the floor ends at walls and furniture, and the shadow
# can't go past them. See fighter/floor.py.
FLOOR_MODEL_FILE = "models/segformer-b0-ade.onnx"   # relative to ar_fighter.py; downloaded if missing
FLOOR_MODEL_URL = "https://huggingface.co/Xenova/segformer-b0-finetuned-ade-512-512/resolve/main/onnx/model.onnx"
FLOOR_CLASS_IDS = (3, 6, 9, 11, 13, 28, 29, 46, 52, 91, 94, 101)  # floor, road, grass, sidewalk, earth, rug, ...
FLOOR_MIN_FRACTION = 0.04         # less floor than this (of the image) and the mask is ignored
FLOOR_MAP_FRAMES = 3              # frames segmented after calibration and averaged (steadier walls)
FLOOR_MAP_INTERVAL = 1.0          # seconds between those frames
FLOOR_MIN_LANE = 1.0              # a lane narrower than this many body heights is treated as a bad mask
FLOOR_EDGE_MARGIN = 0.03          # the shadow never gets closer than this fraction of the width to the screen edge
FLOOR_HALF_FOOTPRINT = 0.12       # body heights either side of the shadow's centre that must stay inside the walls
FLOOR_K_MIN = 0.6                 # the floor grid is drawn from your feet back to this scale
FLOOR_HORIZON_MIN_SPREAD = 0.07   # depth-ratio spread needed before your steps refine the horizon
FLOOR_GRID_STEP = 0.25            # grid spacing on the floor, in body heights
FLOOR_GRID_SHOW = 3.5             # seconds the grid and walls are shown after mapping ([D] shows them always)
FLOOR_GRID_COLOR = (90, 200, 255)
FLOOR_GRID_ALPHA = 38
WALL_COLOR = (255, 120, 70)
WALL_SLAM_SPEED = 0.4            # knocked into a wall faster than this (body heights / s): impact effect
CONTACT_SHADOW_ALPHA = 120

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
ENEMY_ROLL_TIME = 0.9             # backward roll (ninja): duration...
ENEMY_ROLL_SPEED = 1.6            # ...and speed in body heights / second
ENEMY_KICK_KNOCKBACK = 1.8        # body kick: pushed back (body heights / second) but stays on its feet
ENEMY_ACCEL = 4.0                 # body heights / second^2 for a boss of mass 1 (heavier = slower to start/stop)
ENEMY_SIZE_FOLLOW_TAU = 1.0       # seconds: how quickly the shadow matches your size when you step in / back
ENEMY_SIZE_DEADZONE = 0.05        # ...ignoring size differences smaller than this (tracking noise)
ENEMY_TURN_TIME = 0.22            # seconds to turn round when you cross over
ENEMY_HITSTOP = 0.07              # the boss freezes this long when a hit lands (sells the contact)
FLINCH_STIFFNESS = 140.0          # additive flinch spring: the body snaps away from a hit and recovers
FLINCH_DAMPING = 14.0
ANIM_FADE = 0.15                  # default cross-fade between clips (seconds)
DODGE_HOP_HEIGHT = 0.06           # hop-back dodge apex, in body heights

# --- Bosses (fought in order) -------------------------------------------------
# dodge_chance / kick_block_chance only apply while the boss is idle or walking,
# never mid-attack, stunned or knocked down.
BOSSES = [
    {
        "name": "SHADOW",
        "mass": 1.0,              # heavier = slower to start and stop, knocked back less
        "max_hp": 200,
        "dodge_chance": 0.10,
        "roll_chance": 0.0,       # fraction of dodges done as a backward roll instead of a hop back
        "dodge_cooldown": 1.0,    # minimum seconds between dodges
        "kick_block_chance": 0.30,
        "attack_speed": 1.0,      # > 1 = faster attacks
        "idle_min": 0.25,         # pause between attacks (shrinks further as the boss gets hurt)
        "idle_max": 0.7,
        "attack_weights": {"jab": 3.0, "cross": 2.0, "high_kick": 1.5, "side_kick": 1.2, "low_sweep": 1.5},
        "rim": (95, 65, 150),
        "headband": None,
    },
    {
        "name": "SHADOW NINJA",
        "mass": 0.8,
        "max_hp": 300,
        "dodge_chance": 0.55,
        "roll_chance": 0.6,
        "dodge_cooldown": 0.7,
        "kick_block_chance": 0.35,
        "attack_speed": 1.1,
        "idle_min": 0.25,
        "idle_max": 0.6,
        "attack_weights": {"jab": 2.0, "cross": 2.0, "high_kick": 1.5, "front_kick": 1.2, "low_sweep": 1.2,
                           "throwing_star": 3.0},
        "rim": (170, 40, 55),
        "headband": (200, 30, 40),
    },
    {
        "name": "NINJA MONK",
        "mass": 1.0,
        "max_hp": 400,
        "dodge_chance": 0.35,
        "roll_chance": 0.3,
        "dodge_cooldown": 0.8,
        "kick_block_chance": 0.40,
        "attack_speed": 1.1,
        "idle_min": 0.25,
        "idle_max": 0.6,
        "attack_weights": {"staff_swipe": 2.5, "staff_ground": 2.0, "staff_lunge": 2.0,
                           "high_kick": 1.0},
        "rim": (235, 150, 40),
        "headband": None,
        "staff": True,            # fights with a staff (arms hold it; it has its own hitbox)
    },
    {
        "name": "EGYPTIAN SOLDIER",
        "mass": 1.25,
        "max_hp": 160,            # the least health, but the hardest hitter
        "dodge_chance": 0.25,
        "roll_chance": 0.0,
        "dodge_cooldown": 0.9,
        "kick_block_chance": 0.30,
        "attack_speed": 1.15,
        "idle_min": 0.25,
        "idle_max": 0.55,
        "attack_weights": {"sword_slash": 3.0, "sword_cut": 2.5, "high_kick": 1.5, "low_sweep": 1.2},
        "flip_chance": 0.25,      # chance, when choosing a move, to flip over you and slash from behind
        "rim": (210, 170, 60),
        "headband": None,
        "sword": True,            # curved khopesh in the front hand
        "headdress": True,
        "player_shield": True,    # you get a shield on your left arm for this fight
    },
    {
        "name": "MAGE",
        "mass": 0.9,
        "max_hp": 260,
        "dodge_chance": 0.15,
        "roll_chance": 0.0,
        "dodge_cooldown": 1.0,
        "kick_block_chance": 0.20,
        "attack_speed": 1.0,
        "idle_min": 0.6,          # a little breathing room between spells
        "idle_max": 1.0,
        "attack_weights": {"meteor": 2.0, "icicle_rain": 2.5, "fire_wall_low": 2.0, "fire_wall_high": 2.0},
        "no_repeat": True,        # never casts the same spell twice in a row
        "teleports": True,        # defensive teleport when you pressure him (see MAGE_TELEPORT_*)
        "rim": (120, 95, 255),
        "headband": None,
        "robe": True,
    },
]

# --- Sword (Egyptian Soldier) and your shield ---------------------------------------
SWORD_PARRY_STAGGER = 1.2         # seconds he staggers after his sword hits your shield
FLIP_TIME = 0.8                   # crouch + flip over you
FLIP_HEIGHT = 0.65                # apex of the flip, in his body heights
SHIELD_RADIUS = 0.367             # torso units (2/3 of the original 0.55)
SHIELD_COLOR = (150, 105, 40)
SHIELD_RIM_COLOR = (225, 185, 95)
BLADE_COLOR = (185, 160, 105)

# --- Mage spells -------------------------------------------------------------------
# Every spell captures your position when the cast starts and its danger zones never move.
# Distances are in torso units (your torso length) unless noted.
MAGE_CAST_WINDUP = 0.4            # hands raised before the spell appears
MAGE_CAST_CHANNEL = 0.9           # holds the cast while the warnings count down
MAGE_RECOVERY = 1.3               # hunched and drained after each spell: your opening to attack

METEOR_RADIUS = 1.0               # half-width of the danger zone on the floor
METEOR_WARNING = 1.8              # seconds from the circle appearing to impact
METEOR_FALL_TIME = 0.5            # last part of the warning, when the meteor is visibly falling
METEOR_DAMAGE = 30

ICICLE_COUNT = (5, 8)             # min, max zones (fewer if the screen edge leaves no room)
ICICLE_RADIUS = 0.22
ICICLE_SPACING = 0.32             # centre-to-centre spacing of the zones in the storm
ICICLE_AREA = 2.2                 # the storm spans this far either side of you (rest of the floor is clear)
ICICLE_MAX_FLOOR = 0.7            # ...but never more than this fraction of the playable floor
ICICLE_SAFE_GAP = 1.3             # one guaranteed clear gap this wide inside the storm...
ICICLE_GAP_OFFSET = (0.9, 1.4)    # ...centred this far to one side of where you stood
ICICLE_WARNING = 1.3              # seconds until the first icicle lands
ICICLE_STAGGER = 0.14             # seconds between consecutive impacts
ICICLE_FALL_TIME = 0.35
ICICLE_DAMAGE = 10

PLAYER_HALF_WIDTH = 0.3           # your body's half-width when checking ground zones

FIRE_WALL_TELEGRAPH = 1.1         # seconds the path glows before the wall launches
FIRE_WALL_SPEED = 1.0             # body heights / second
FIRE_WALL_WIDTH = 0.22            # thickness of the wall, in the mage's body heights
FIRE_WALL_LOW_HEIGHT = 0.28       # low wall: floor up to this fraction of body height (jump it)
FIRE_WALL_HIGH_CLEARANCE = 0.2    # high wall: its bottom edge sits this far below your standing head
FIRE_WALL_HIGH_THICKNESS = 1.2    # how far the high wall extends up from its bottom edge
FIRE_WALL_DAMAGE = 16
FIRE_WALL_MIN_LEAD = 0.5          # the wall always starts at least this many body heights from you

MAGE_TELEPORT_TRIGGER_DIST = 0.8  # "close" = within this many body heights of you
MAGE_TELEPORT_PRESSURE = 3.5      # seconds of being close (accumulated) before he repositions
MAGE_TELEPORT_DECAY = 0.5         # pressure drains at this rate per second while you're away
MAGE_TELEPORT_COOLDOWN = 10.0
MAGE_TELEPORT_TELL = 0.5          # shimmer before vanishing
MAGE_TELEPORT_GONE = 0.25         # time invisible
MAGE_TELEPORT_DISTANCE = 0.4      # reappears this fraction of the screen width from you

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
SAVE_FILE = "save_data.json"      # remembers the furthest boss you've reached (next to ar_fighter.py)

# --- Colours -----------------------------------------------------------------
ENEMY_BODY = (8, 6, 12)
ENEMY_BACK = (30, 26, 40)
ENEMY_FLASH = (235, 235, 245)
ENEMY_EYES = (230, 235, 255)
ENEMY_EYES_ANGRY = (255, 70, 40)
PLAYER_BAR = (70, 200, 255)
ENEMY_BAR = (200, 60, 230)
CHARACTER_FILE = "assets/character/fighter.npz"  # baked by tools/build_character.py (see ASSETS.md)
