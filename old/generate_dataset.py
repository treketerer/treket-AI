import numpy as np


# --- Шаг 1: Алгоритм "Эксперт" (Минимакс) ---

def check_winner(board):
    """Проверяет, есть ли победитель на доске."""
    lines = [
        # Горизонтали
        board[0:3], board[3:6], board[6:9],
        # Вертикали
        board[0:9:3], board[1:9:3], board[2:9:3],
        # Диагонали
        board[0:9:4], board[2:7:2]
    ]
    for line in lines:
        if line[0] == line[1] == line[2] != 0:
            return line[0]  # Возвращает 1 или -1
    if 0 not in board:
        return 0  # Ничья
    return None  # Игра продолжается


def minimax(board, player):
    """
    Рекурсивный алгоритм Минимакс для поиска лучшего хода.
    player: 1 (мы, максимизируем), -1 (оппонент, минимизируем)
    """
    winner = check_winner(board)
    if winner is not None:
        return {'score': winner}  # Возвращаем оценку для конечной позиции

    moves = []
    # Находим все возможные ходы
    for i, cell in enumerate(board):
        if cell == 0:
            # Делаем ход
            new_board = board[:]
            new_board[i] = player

            # Рекурсивно вызываем минимакс для оппонента
            result = minimax(new_board, -player)

            # Сохраняем ход и его оценку
            moves.append({'index': i, 'score': result['score']})

    # Выбираем лучший ход для текущего игрока
    best_move = None
    if player == 1:  # Максимизирующий игрок (ИИ)
        best_score = -float('inf')
        for move in moves:
            if move['score'] > best_score:
                best_score = move['score']
                best_move = move
    else:  # Минимизирующий игрок (Оппонент)
        best_score = float('inf')
        for move in moves:
            if move['score'] < best_score:
                best_score = move['score']
                best_move = move

    return best_move


# --- Шаг 2: Генерация и аугментация данных ---

def augment_data(state, target_move):
    """
    Создает 8 симметричных позиций (оригинал, 3 поворота, 4 отражения).
    """
    augmented_data = []

    # Конвертируем в матрицу 3x3 для удобства
    board = np.array(state).reshape(3, 3)
    move_board = np.array(target_move).reshape(3, 3)

    for i in range(4):  # 4 поворота
        # Поворачиваем доску и целевой ход
        rotated_board = np.rot90(board, i)
        rotated_move = np.rot90(move_board, i)

        # Добавляем оригинал и отраженную версию
        augmented_data.append((rotated_board.flatten().tolist(), rotated_move.flatten().tolist()))
        augmented_data.append((np.fliplr(rotated_board).flatten().tolist(), np.fliplr(rotated_move).flatten().tolist()))

    return augmented_data


# --- Шаг 3: Основной цикл генерации ---

def generate_training_data():
    """
    Основная функция для генерации полного и чистого датасета.
    """
    all_positions = {}  # Используем словарь, чтобы избежать дубликатов
    initial_board = [0] * 9

    # Используем стек для обхода всех возможных игровых состояний
    stack = [initial_board]
    visited = {tuple(initial_board)}

    while stack:
        current_board = stack.pop()

        # Если игра не закончена
        if check_winner(current_board) is None:
            # Определяем, чей ход (ИИ ходит, если количество 1 и -1 равно)
            player_to_move = 1 if current_board.count(1) == current_board.count(-1) else -1

            # Если ход ИИ, находим лучший ход и записываем
            if player_to_move == 1:
                best_move_info = minimax(current_board, 1)
                if best_move_info:
                    move_index = best_move_info['index']

                    # Создаем one-hot вектор для хода
                    target = [0] * 9
                    target[move_index] = 1

                    # Добавляем позицию в наш словарь
                    all_positions[tuple(current_board)] = target

            # Генерируем следующие состояния
            for i, cell in enumerate(current_board):
                if cell == 0:
                    next_board = current_board[:]
                    next_board[i] = player_to_move
                    if tuple(next_board) not in visited:
                        visited.add(tuple(next_board))
                        stack.append(next_board)

    # Аугментируем и финализируем датасет
    final_data = {}
    for state, move in all_positions.items():
        augmented_pairs = augment_data(list(state), move)
        for aug_state, aug_move in augmented_pairs:
            # Проверяем, что в целевом векторе есть ход (не пустой)
            if sum(aug_move) > 0:
                final_data[tuple(aug_state)] = aug_move

    # Преобразуем в финальный список
    training_data = [[list(k), v] for k, v in final_data.items()]
    return training_data


# --- Запускаем генерацию и выводим результат ---
training_data = generate_training_data()
print(training_data)
# Выведем для примера количество уникальных позиций и первые 5
print(f"Сгенерировано {len(training_data)} уникальных обучающих примеров (с учетом симметрии).")
print("\nПервые 5 примеров из датасета:")
for i in range(5):
    print(f"Состояние: {training_data[i][0]}")
    print(f"Лучший ход: {training_data[i][1]}\n")

# Теперь переменная 'training_data' содержит полный и чистый датасет.
# Вы можете использовать ее напрямую в вашем коде для обучения ИИ.
# Например, так:
# data = training_data