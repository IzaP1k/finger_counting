import cv2
from hand_func import handDetector, get_points
import numpy as np
from joblib import load
import pandas as pd
import pygame
from random import randrange
from camera_prediction import capture_frames, process_frames, predict_frames

pygame.init()
pygame.font.init()
font = pygame.font.SysFont('Arial', 30)

WINDOW = 1000
TITLE_SIZE = 50
length = 1
time_step = 200

def snake_game(WINDOW, TITLE_SIZE, length, time_step, model_path="SVC()-model.joblib"):

    results = []

    RANGE = (TITLE_SIZE // 2, WINDOW - TITLE_SIZE)
    get_random_position_snake = lambda: [randrange(TITLE_SIZE, WINDOW - TITLE_SIZE), randrange(TITLE_SIZE, WINDOW - TITLE_SIZE)]
    get_random_position_food = lambda: [randrange(TITLE_SIZE, WINDOW - TITLE_SIZE), randrange(TITLE_SIZE, WINDOW - TITLE_SIZE)]

    snake = pygame.Rect([0, 0, TITLE_SIZE - 2, TITLE_SIZE - 2])
    snake.center = get_random_position_snake()
    snake_dir = (0, 0)
    segments = [snake.copy()]

    food = pygame.Rect([0, 0, TITLE_SIZE - 2, TITLE_SIZE - 2])
    food.center = get_random_position_food()

    screen_width = WINDOW + 640
    screen_height = WINDOW
    screen = pygame.display.set_mode([screen_width, screen_height])

    clock = pygame.time.Clock()
    time = 0

    dirs = {pygame.K_w: 1, pygame.K_s: 1, pygame.K_a: 1, pygame.K_d: 1}

    cap = cv2.VideoCapture(0)
    detector = handDetector()
    best_model = load(model_path)

    if not cap.isOpened():
        print("Nie można otworzyć kamery")
        exit()

    pygame.display.set_caption('Snake Game with Hand Gesture Control')

    def get_most_common_prediction(frames):
        df = pd.DataFrame({'image': frames})
        df['points'] = df['image'].apply(lambda img: get_points(detector, img))
        try:
            df = process_frames(df)
            predictions, most_common_prediction = predict_frames(df, best_model)
            return most_common_prediction
        except IndexError:
            return None

    def display_text(text, position, color=(255, 0, 0)):
        text_surface = font.render(text, True, color)
        screen.blit(text_surface, position)

    while True:
        frames = capture_frames(cap)
        most_common_prediction = get_most_common_prediction(frames)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                cap.release()
                cv2.destroyAllWindows()
                pygame.quit()

                return results

        if most_common_prediction is not None:
            if most_common_prediction == 1:
                snake_dir = (0, -TITLE_SIZE)
            elif most_common_prediction == 0:
                snake_dir = (0, TITLE_SIZE)
            elif most_common_prediction == 3:
                snake_dir = (-TITLE_SIZE, 0)
            elif most_common_prediction == 5:
                snake_dir = (TITLE_SIZE, 0)

        screen.fill('black')

        pygame.draw.line(screen, (255, 255, 255), (WINDOW, 0), (WINDOW, WINDOW), 2)

        self_eating = pygame.Rect.collidelist(snake, segments[:-1]) != -1
        if snake.left < 0 or snake.right > WINDOW or snake.top < 0 or snake.bottom > WINDOW or self_eating:
            results.append(length)
            display_text(f"Game Over\nScore: {length}", (WINDOW // 2 - 100, WINDOW // 2 - 50))
            pygame.display.flip()
            pygame.time.wait(3000)
            snake.center, food.center = get_random_position_snake(), get_random_position_food()
            length, snake_dir = 1, (0, 0)
            segments = [snake.copy()]

        if snake.colliderect(food):
            food.center = get_random_position_food()
            length += 1

        pygame.draw.rect(screen, 'red', food)
        [pygame.draw.rect(screen, 'green', segment) for segment in segments]

        time_now = pygame.time.get_ticks()
        if time_now - time > time_step:
            time = time_now
            snake.move_ip(snake_dir)
            segments.append(snake.copy())
            if len(segments) > length:
                segments.pop(0)

        display_text(f"Score: {length}", (10, 10))

        frame = frames[0]
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_surface = pygame.surfarray.make_surface(frame_rgb.transpose(1, 0, 2))
        screen.blit(frame_surface, (WINDOW + 10, 10))

        pygame.display.flip()
        clock.tick(60)


# snake_game(WINDOW, TITLE_SIZE, length, time_step)
