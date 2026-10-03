import sys
import threading
import time

import cv2
import mediapipe as mp
import pygame

# MediaPipe Hands setup (model_complexity=0 is the faster, lighter hand model)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, model_complexity=0,
                       min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Camera 0 is the iPhone (Continuity Camera) when it's nearby; 1 is the built-in FaceTime camera.
# Pick another with: python pingpong.py <camera number>
CAMERA_INDEX = int(sys.argv[1]) if len(sys.argv) > 1 else 1
cap = cv2.VideoCapture(CAMERA_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
if not cap.isOpened():
    raise SystemExit("Could not open the camera. On macOS, allow camera access for your terminal app "
                     "(System Settings > Privacy & Security > Camera), then fully quit and reopen it.")

# Screen & Paddle settings
WIDTH, HEIGHT = 1280, 720
PADDLE_W, PADDLE_H = 20, 140
PADDLE_SMOOTHING = 20  # Higher = paddles follow hands more tightly
CENTER_GAP = 60  # Paddles can't come closer than this to the center line
SWING_POWER = 0.8  # How much of the paddle's swing speed is passed on to the ball

# Colors (RGB)
BLUE, ORANGE, YELLOW = (0, 100, 255), (255, 100, 0), (255, 255, 0)
WHITE, GRAY, GREEN = (255, 255, 255), (100, 100, 100), (0, 255, 120)


def make_paddle(home_x, min_x, max_x, direction, color):
    # direction: +1 if this paddle hits the ball to the right, -1 to the left
    return {"x": home_x, "y": HEIGHT / 2, "target_x": home_x, "target_y": HEIGHT / 2,
            "vx": 0.0, "vy": 0.0, "min_x": min_x, "max_x": max_x, "dir": direction, "color": color}


p1 = make_paddle(60, PADDLE_W, WIDTH // 2 - CENTER_GAP, 1, BLUE)
p2 = make_paddle(WIDTH - 60, WIDTH // 2 + CENTER_GAP, WIDTH - PADDLE_W, -1, ORANGE)

# Ball settings (velocities are pixels per 1/60th of a second)
TARGET_FPS = 60
ball_x, ball_y = WIDTH // 2, HEIGHT // 2
ball_vx, ball_vy = 9, 6
BALL_R = 10
MIN_BALL_SPEED, MAX_BALL_SPEED = 7, 28  # Max keeps the ball from skipping through paddles
score_p1, score_p2 = 0, 0

# Latest camera frame and detected hands, shared with the tracking thread
latest = {"frame": None, "frame_id": 0, "hands": []}
lock = threading.Lock()
running = True


def track_hands():
    """Read the camera and run hand tracking in the background, so the game loop never waits on them."""
    failed_reads = 0
    while running:
        ret, frame = cap.read()
        if not ret:
            # The camera can return empty frames while it warms up
            failed_reads += 1
            if failed_reads > 100:
                print("Camera stopped returning frames.")
                break
            time.sleep(0.01)
            continue
        failed_reads = 0

        frame = cv2.flip(frame, 1)  # Mirror frame
        # Center-crop to 16:9 before resizing; stretching the image stops MediaPipe detecting hands
        h, w = frame.shape[:2]
        if w * HEIGHT > h * WIDTH:
            crop_w = h * WIDTH // HEIGHT
            frame = frame[:, (w - crop_w) // 2:(w + crop_w) // 2]
        else:
            crop_h = w * HEIGHT // WIDTH
            frame = frame[(h - crop_h) // 2:(h + crop_h) // 2, :]
        frame = cv2.cvtColor(cv2.resize(frame, (WIDTH, HEIGHT)), cv2.COLOR_BGR2RGB)

        # Landmarks are normalized 0-1, so detecting on a smaller copy works for the full-size frame
        results = hands.process(cv2.resize(frame, (WIDTH // 2, HEIGHT // 2)))

        with lock:
            latest["frame"] = frame
            latest["frame_id"] += 1
            latest["hands"] = results.multi_hand_landmarks or []


def draw_hand(surface, hand_landmarks):
    points = [(int(lm.x * WIDTH), int(lm.y * HEIGHT)) for lm in hand_landmarks.landmark]
    for a, b in mp_hands.HAND_CONNECTIONS:
        pygame.draw.line(surface, WHITE, points[a], points[b], 2)
    for point in points:
        pygame.draw.circle(surface, GREEN, point, 4)


pygame.init()
# SCALED lets the window go fullscreen (press F) while the game keeps drawing at 1280x720
screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.SCALED)
pygame.display.set_caption("Hand-Tracked Split-Screen Ping Pong")
score_font = pygame.font.Font(None, 90)
info_font = pygame.font.Font(None, 32)

tracker = threading.Thread(target=track_hands, daemon=True)
tracker.start()

camera_surface = None
shown_frame_id = 0
fps = TARGET_FPS
prev_time = next_frame = time.perf_counter()
while tracker.is_alive():
    # Wait until the next frame's deadline to hold a steady 60fps. Sleeping oversleeps on macOS
    # (pygame's clock.tick only reaches ~53fps), and a full busy-wait starves the tracking thread,
    # so sleep until just before the deadline and spin for the last couple of milliseconds.
    next_frame = max(next_frame + 1 / TARGET_FPS, time.perf_counter() - 1 / TARGET_FPS)
    if next_frame - time.perf_counter() > 0.003:
        time.sleep(next_frame - time.perf_counter() - 0.003)
    while time.perf_counter() < next_frame:
        pass

    now = time.perf_counter()
    dt = min(now - prev_time, 0.05)  # Cap the step so a hiccup can't teleport the ball
    prev_time = now
    step = dt * TARGET_FPS
    fps = fps * 0.9 + (1 / max(dt, 1e-6)) * 0.1

    for event in pygame.event.get():
        if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key in (pygame.K_q, pygame.K_ESCAPE)):
            running = False
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_f:
            pygame.display.toggle_fullscreen()
    if not running:
        break

    with lock:
        camera_frame, frame_id, hand_list = latest["frame"], latest["frame_id"], latest["hands"]
    if camera_frame is None:
        screen.fill((0, 0, 0))
        screen.blit(info_font.render("Starting camera...", True, WHITE), (20, HEIGHT - 40))
        pygame.display.flip()
        continue
    # The camera updates ~30 times a second, so only rebuild its image when there's a new frame
    if frame_id != shown_frame_id:
        camera_surface = pygame.image.frombuffer(camera_frame.tobytes(), (WIDTH, HEIGHT), "RGB")
        shown_frame_id = frame_id

    # Track hands for left (P1) and right (P2)
    for hand_landmarks in hand_list:
        palm = hand_landmarks.landmark[9]  # Landmark 9 = palm center
        hx, hy = int(palm.x * WIDTH), int(palm.y * HEIGHT)

        paddle = p1 if hx < WIDTH // 2 else p2
        paddle["target_x"], paddle["target_y"] = hx, hy

    # Glide paddles toward the hands; the camera updates slower than the game draws.
    # Track each paddle's velocity so a swing can be passed on to the ball.
    follow = min(1, dt * PADDLE_SMOOTHING)
    for paddle in (p1, p2):
        paddle["prev_x"] = paddle["x"]
        target_x = min(max(paddle["target_x"], paddle["min_x"]), paddle["max_x"])
        move_x = (target_x - paddle["x"]) * follow
        move_y = (paddle["target_y"] - paddle["y"]) * follow
        paddle["x"] += move_x
        paddle["y"] += move_y
        # Lightly smoothed so camera jitter doesn't register as a swing
        paddle["vx"] = paddle["vx"] * 0.5 + (move_x / step) * 0.5
        paddle["vy"] = paddle["vy"] * 0.5 + (move_y / step) * 0.5

    # Ball movement
    ball_x += ball_vx * step
    ball_y += ball_vy * step

    # Top/Bottom boundary bounce
    if ball_y - BALL_R <= 0:
        ball_vy = abs(ball_vy)
    elif ball_y + BALL_R >= HEIGHT:
        ball_vy = -abs(ball_vy)

    # Paddle collisions. The hit zone covers everywhere the paddle passed this frame,
    # so a fast swing can't jump over the ball.
    for paddle in (p1, p2):
        left = min(paddle["x"], paddle["prev_x"]) - PADDLE_W / 2
        right = max(paddle["x"], paddle["prev_x"]) + PADDLE_W / 2
        touching = (left - BALL_R <= ball_x <= right + BALL_R) and abs(ball_y - paddle["y"]) < PADDLE_H / 2 + BALL_R
        # Only hit if the ball is coming at the paddle's face (relative to the paddle's own movement)
        approaching = (ball_vx - paddle["vx"]) * paddle["dir"] < 0
        if touching and approaching:
            d = paddle["dir"]
            # Bounce back slightly faster, plus the paddle's forward swing speed
            swing = max(paddle["vx"] * d, 0)
            speed = abs(ball_vx) * 1.03 + swing * SWING_POWER
            ball_vx = d * min(max(speed, MIN_BALL_SPEED), MAX_BALL_SPEED)
            offset = (ball_y - paddle["y"]) / (PADDLE_H / 2)
            ball_vy = offset * 8 + paddle["vy"] * 0.3
            # Move the ball to the paddle's front face so it doesn't get hit twice
            ball_x = (right + BALL_R) if d > 0 else (left - BALL_R)

    # Score & Reset
    if ball_x < 0:
        score_p2 += 1
        ball_x, ball_y, ball_vx, ball_vy = WIDTH // 2, HEIGHT // 2, 9, 6
    elif ball_x > WIDTH:
        score_p1 += 1
        ball_x, ball_y, ball_vx, ball_vy = WIDTH // 2, HEIGHT // 2, -9, 6

    # Draw Camera, Hands, Center Divider, Paddles, Ball, and Scores
    screen.blit(camera_surface, (0, 0))
    for hand_landmarks in hand_list:
        draw_hand(screen, hand_landmarks)
    pygame.draw.line(screen, GRAY, (WIDTH // 2, 0), (WIDTH // 2, HEIGHT), 2)
    for paddle in (p1, p2):
        pygame.draw.rect(screen, paddle["color"],
                         (int(paddle["x"] - PADDLE_W / 2), int(paddle["y"] - PADDLE_H / 2), PADDLE_W, PADDLE_H))
    pygame.draw.circle(screen, YELLOW, (int(ball_x), int(ball_y)), BALL_R)
    score = score_font.render(f"{score_p1}   {score_p2}", True, WHITE)
    screen.blit(score, score.get_rect(midtop=(WIDTH // 2, 25)))
    info = info_font.render(f"Hands: {len(hand_list)}   FPS: {fps:.0f}   F: fullscreen   Q: quit", True, WHITE)
    screen.blit(info, (20, HEIGHT - 40))

    pygame.display.flip()

running = False
tracker.join(timeout=1)
cap.release()
pygame.quit()
