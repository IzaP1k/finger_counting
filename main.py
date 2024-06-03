from camera_prediction import get_camera_data
from camera_snake import snake_game

"""

You can test my models in-real time. Answer the question on terminal

"""

count_finger = input('Czy chcesz włączyć rozpoznawanie cyfr? Y or N')

if count_finger == 'Y':
    count_finger = True
elif count_finger == 'N':
    count_finger = False
else:
    raise 'Niepoprawna odpowiedź'


if count_finger:

    get_camera_data('SVC()-model.joblib', svm=True)

snake = input('Czy chcesz włączyć grę snake? Y or N')

if snake == 'Y':
    snake = True
elif snake == 'N':
    snake = False
else:
    raise 'Niepoprawna odpowiedź'

if snake:

    WINDOW = 1000
    TITLE_SIZE = 50
    length = 1
    time_step = 200

    snake_game(WINDOW, TITLE_SIZE, length, time_step)