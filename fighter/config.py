"""Tunable constants.

Player distances are in "torso units" (shoulder-midpoint to hip-midpoint length),
so colliders and speed thresholds scale automatically with distance from the camera.
Enemy distances are fractions of the enemy's height H.
"""

# --- Display -----------------------------------------------------------------
SCREEN_W = 1280
SCREEN_H = 720
FPS = 60
WINDOW_TITLE = "ARena: Fit Fighter"

# --- Music -------------------------------------------------------------------
MUSIC_DIR = "music"               # relative to ar_fighter.py; first .mp3/.ogg/.wav is the fight music
MENU_MUSIC_DIR = "music/menu"     # first audio file here plays on the main menu
                                  # (each boss can also have its own theme: see "music" in BOSSES)
MUSIC_VOLUME = 0.6                # 0.0 - 1.0
MUSIC_FADE_IN_MS = 500
MUSIC_FADE_OUT_MS = 1200

# --- Sound effects -----------------------------------------------------------
SFX_DIR = "sound_effects"         # relative to ar_fighter.py
SFX_VOLUME = 0.9                  # 0.0 - 1.0
SFX_KEYWORDS = {                  # a file is used for a role if its name contains any keyword
    "bell": ("bell",),               # (roles are matched in this order and a file only fills one role)
    "kick": ("kick",),
    "punch": ("punch", "impact", "hit"),
    "groan": ("disappoint", "groan", "boo", "aww"),  # defeat; before "cheer" so a "crowd ..." groan isn't the cheer
    "cheer": ("cheer", "crowd", "applause"),
    "throw": ("throw", "whoosh", "swish"),
}
SFX_POOLED_ROLES = ("punch", "kick", "throw")  # these use every matching file, picking one at random

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
THROW_RELEASE_FACTOR = 0.6        # stats: a punch/kick ends once the limb slows below this x its threshold
KICK_THROW_MIN_LIFT = 0.5         # stats: a missed kick only counts as thrown if the foot rose this many
                                  # torso units above the other foot (so fast steps aren't kicks)

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
        "name": "NIGHTFIST",
        "title": "THE NAMELESS BRAWLER",          # shown in the boss intro
        "music": "music/bosses/shadow",  # optional theme; falls back to the fight music
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
        "name": "KAID",
        "title": "THE SILENT BLADE",          # shown in the boss intro
        "music": "music/bosses/shadow_ninja",  # optional theme; falls back to the fight music
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
        "name": "IRON LOTUS",
        "title": "MASTER OF THE IRON STAFF",          # shown in the boss intro
        "music": "music/bosses/ninja_monk",  # optional theme; falls back to the fight music
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
    {
        "name": "ANUBIS",
        "title": "GUARDIAN OF THE SANDS",          # shown in the boss intro
        "music": "music/bosses/egyptian_soldier",  # optional theme; falls back to the fight music
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
        "name": "MALAKAR",
        "title": "THE ARCANE WARDEN",          # shown in the boss intro
        "music": "music/bosses/mage",  # optional theme; falls back to the fight music
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
SUMMARY_DELAY = 3.2               # seconds of K.O. / result banner before the Fight Summary appears
SUMMARY_TALLY_TIME = 0.9          # numbers on the Fight Summary count up over this long

# --- Boss intro (before the 3-2-1 countdown) ------------------------------------
INTRO_WARNING_TIME = 1.4          # flashing WARNING / "<BOSS> APPROACHING"
INTRO_TITLE_TIME = 1.6            # boss name and title (he walks in during both)

# --- Combos --------------------------------------------------------------------
COMBO_WINDOW = 1.5                # seconds allowed between landed hits to keep a combo going
COMBO_WORDS = (                   # (minimum hits, shout, colour), best match first
    (5, "UNSTOPPABLE!", (255, 120, 60)),
    (4, "AMAZING!", (255, 220, 120)),
    (3, "GREAT!", (120, 255, 160)),
    (2, "NICE!", (255, 255, 255)),
)

# --- Dramatic K.O. ----------------------------------------------------------------
KO_HITSTOP = 0.12                 # the world freezes for this long on the finishing blow...
KO_SLOWMO_TIME = 1.1              # ...then runs in slow motion until this many seconds after it
KO_SLOWMO_SCALE = 0.3
ENEMY_DEATH_FADE_START = 1.2      # the K.O.'d boss lies still this long (game seconds)...
ENEMY_DEATH_FADE_TIME = 0.8       # ...then fades away over this long

# --- Fitness score (a game score built from your fight stats) -------------------
SCORE_WEIGHTS = {"accuracy": 0.30, "dodging": 0.30, "activity": 0.25, "combos": 0.15}
SCORE_ACTIVITY_TARGET = 30.0      # punches + kicks + dodges per minute that earns 100 for Activity
SCORE_COMBO_TARGET = 6            # best combo that earns 100 for Combos
SCORE_RANKS = ((90, "S"), (80, "A"), (65, "B"), (50, "C"), (0, "D"))
SCORE_POINTS = {                  # Fitness Score = performance x 40 + these per stat
    "hit": 50, "dodge": 100, "block": 40, "best_combo": 80,
    "win": 1500, "health": 1000,  # win bonus, plus up to 1000 for health left (wins only)
}

# --- Fight Summary calorie estimate (rough: no body weight is measured) -------------
CALORIE_MET = 6.0                 # metabolic equivalent for active fitness boxing
CALORIE_WEIGHT_KG = 70.0          # assumed body weight
LEADERBOARD_FILE = "leaderboard.json"  # Fitness Score top 10 (next to ar_fighter.py)
LEADERBOARD_SIZE = 10
LEADERBOARD_NAME_LEN = 3          # arcade-style initials
SAVE_FILE = "save_data.json"      # remembers the furthest boss you've reached (next to ar_fighter.py)

# --- Colours -----------------------------------------------------------------
ENEMY_BODY = (8, 6, 12)
ENEMY_BACK = (30, 26, 40)
ENEMY_FLASH = (235, 235, 245)
ENEMY_EYES = (230, 235, 255)
ENEMY_EYES_ANGRY = (255, 70, 40)
PLAYER_BAR = (70, 200, 255)
ENEMY_BAR = (200, 60, 230)
