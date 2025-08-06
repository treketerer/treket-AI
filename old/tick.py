import tkinter as tk
from tkinter import messagebox
import math

class TicTacToe:
    """
    Класс, отвечающий ИСКЛЮЧИТЕЛЬНО за логику игры в крестики-нолики.
    Он не знает о существовании кнопок или окон.
    Именно с ним будет взаимодействовать ваш будущий ИИ.
    """

    def __init__(self, forward_func):
        self.reset()
        self.forward_func = forward_func

    def reset(self):
        """Сбрасывает игру в начальное состояние."""
        self.board = [[None, None, None],
                      [None, None, None],
                      [None, None, None]]

        self.current_player = 'X'
        self.winner = None
        self.is_tie = False
        self.game_over = False

    def make_move(self, row, col):
        """
        Делает ход на доске, если это возможно.
        Возвращает True, если ход успешен, иначе False.
        """
        if not self.game_over and self.board[row][col] is None:
            self.board[row][col] = self.current_player
            self._check_for_winner()
            self._check_for_tie()
            if not self.game_over:
                self.current_player = 'O' if self.current_player == 'X' else 'X'
                if self.current_player == "O":
                    board = []

                    for row in self.board:
                        for item in row:
                            if item is None: board.append(0)
                            elif item == "X": board.append(-1)
                            elif item == "O": board.append(1)
                    print(board)
                    forward_ans = self.forward_func(board)
                    self.make_move(math.floor(forward_ans/3), forward_ans-math.floor(forward_ans/3)*3)
            return True
        print("Выбранна занятая клетка", 3*row + col)
        return False

    def _check_for_winner(self):
        """Проверяет, есть ли победитель."""
        # Проверка по горизонтали и вертикали
        for i in range(3):
            if self.board[i][0] == self.board[i][1] == self.board[i][2] and self.board[i][0] is not None:
                self.winner = self.board[i][0]
            if self.board[0][i] == self.board[1][i] == self.board[2][i] and self.board[0][i] is not None:
                self.winner = self.board[0][i]

        # Проверка по диагоналям
        if self.board[0][0] == self.board[1][1] == self.board[2][2] and self.board[0][0] is not None:
            self.winner = self.board[0][0]
        if self.board[0][2] == self.board[1][1] == self.board[2][0] and self.board[0][2] is not None:
            self.winner = self.board[0][2]

        if self.winner:
            self.game_over = True

    def _check_for_tie(self):
        """Проверяет, закончилась ли игра вничью."""
        # Ничья, если нет победителя и нет пустых клеток
        if self.winner is None and all(self.board[row][col] is not None for row in range(3) for col in range(3)):
            self.is_tie = True
            self.game_over = True


class GameUI:
    """
    Класс, отвечающий за графический интерфейс (UI).
    Он создает окно, кнопки и взаимодействует с классом логики TicTacToe.
    """

    def __init__(self, master, forward_func):
        self.master = master
        self.master.title("Крестики-Нолики")
        self.master.resizable(False, False)

        # Создаем экземпляр нашей игровой логики
        self.game = TicTacToe(forward_func)

        # Создаем виджеты
        self.buttons = [[None for _ in range(3)] for _ in range(3)]
        self.create_widgets()
        self.update_ui()

    def create_widgets(self):
        """Создает и размещает все элементы интерфейса."""
        # Рамка для игрового поля
        board_frame = tk.Frame(self.master)
        board_frame.pack()

        for i in range(3):
            for j in range(3):
                self.buttons[i][j] = tk.Button(
                    board_frame,
                    text="",
                    font=("Helvetica", 24, "bold"),
                    height=2,
                    width=5,
                    command=lambda r=i, c=j: self.on_button_click(r, c)
                )
                self.buttons[i][j].grid(row=i, column=j)

        # Рамка для управления
        control_frame = tk.Frame(self.master, pady=10)
        control_frame.pack()

        self.status_label = tk.Label(control_frame, text="", font=("Helvetica", 14))
        self.status_label.pack()

        restart_button = tk.Button(control_frame, text="Новая игра", command=self.reset_game, font=("Helvetica", 12))
        restart_button.pack()

    def on_button_click(self, row, col):
        """Обрабатывает нажатие на игровую кнопку."""
        if self.game.make_move(row, col):
            self.update_ui()
            self.check_game_over()

    def update_ui(self):
        """Обновляет состояние интерфейса в соответствии с логикой игры."""
        for i in range(3):
            for j in range(3):
                text = self.game.board[i][j] if self.game.board[i][j] is not None else ""
                self.buttons[i][j].config(text=text)

        if not self.game.game_over:
            self.status_label.config(text=f"Ход игрока: {self.game.current_player}")
        else:
            if self.game.winner:
                self.status_label.config(text=f"Победил игрок: {self.game.winner}!")
            elif self.game.is_tie:
                self.status_label.config(text="Ничья!")

    def check_game_over(self):
        """Проверяет, закончилась ли игра, и показывает сообщение."""
        if self.game.game_over:
            if self.game.winner:
                messagebox.showinfo("Конец игры", f"Победил игрок {self.game.winner}!")
            elif self.game.is_tie:
                messagebox.showinfo("Конец игры", "Ничья!")

    def reset_game(self):
        """Запускает игру заново."""
        self.game.reset()
        self.update_ui()


def start(forward_func):
    root = tk.Tk()
    app = GameUI(root, forward_func)
    root.mainloop()