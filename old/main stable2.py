import numpy as np
# from learndata import data
import random
from collections import Counter

import utils
# import kagglehub
# import csv

data_path = "C:/Users/Андрей/Desktop/андрей/Чаты/TIMA_BOT.txt"

data = []
with open(data_path, "r", encoding="utf-8") as file:
    for line in file:
        data.append(line.replace("\n", ""))

print(len(data), data[:20])

class AI_CLASS:
    def __init__(self):
        self.learning_speed = 0.01
        self.iterations = 80000
        self.batch_size = 48

        self.embedding_len = 64
        self.emb_input_len = 5

        self.hide_layers_count = 2
        self.hide_layer_height = 128
        self.MIN_FREQ = 3

        self.input_len = self.emb_input_len * self.embedding_len
        self.output_len = -1
        self.learning_data = []

        self.alphabet_emb = {}
        self.alphabet_indexes = []

        self.w_arr = []
        self.b_arr = []

        self.unk_token_index = -1

        self.init_random_weights()

        self.input = np.random.randn(1, self.input_len)
        self.target_output = np.random.randn(1, self.output_len)

        self.learn_input_words = {}

        self.init_learning_data(data)

        self.learn_cycle()
        self.use_cycle()

    def learn_cycle(self):
        for i in range(self.iterations):
            self.set_random_learn_couple()
            f = self.forward()

            if i % 100 == 0:
                target_indices = [self.get_word_index(item[1]) for item in self.learn_input_words]
                loss = np.mean(utils.sparse_cross_entropy_batch(f.get('answer'), target_indices)).item(),
                targets = np.argmax(self.target_output, axis=1)
                answers = np.argmax(f.get('answer'), axis=1)

                print(f'{i} Loss: {loss} Точность: {np.mean(answers == targets) * 100}%')
            self.backward( f.get('answer'), self.target_output, f.get('h_arr'), f.get('t_arr'))
        print("")

    def use_cycle(self):
        for i in range(100):
            try:
                words_inp = input("Введите слова для ИИ: ").lower()

                for word in words_inp.split():
                    if self.get_word_index(word) == self.unk_token_index:
                        print(f"Слова {word} нет в словаре!")

                embeddings = self.parse_embeddings(words_inp.split())
                self.input = self.set_input(embeddings)
                f = self.forward()
                logits = f.get('t_arr')[-1] # Получаем логиты до softmax
                argmax = self.sample_with_temperature(logits, temperature=0.8)
                # argmax = np.argmax(f.get('answer')[0])
                print("Ответ:", self.alphabet_indexes[argmax])
            except Exception as ex:
                print(ex)

    def get_word_index(self, word):
        try:
            index = self.alphabet_indexes.index(word)
            return index
        except:
            index = self.alphabet_indexes.index("<UNK>")
            return index

    def get_word_embedding(self, word):
        try:
            index = self.alphabet_emb[word]
            return index
        except:
            index = self.alphabet_emb["<UNK>"]
            return index

    def sample_with_temperature(self, logits, temperature=1.0):
        logits = logits / temperature
        # Предотвращаем переполнение
        probs = utils.softmax(logits)
        # Выбираем индекс на основе вероятностей
        next_word_idx = np.random.choice(len(probs[0]), p=probs[0])
        return next_word_idx


    def init_learning_data(self, learn_data):
        known_words = set(self.alphabet_indexes)

        for learn_couple in learn_data:
            couple_arr = learn_couple.lower().split()
            if len(couple_arr) < 2:
                continue

            processed_couple = [word if word in known_words else '<UNK>' for word in couple_arr]

            for i in range(len(processed_couple) - 1):
                learn_couple = processed_couple[:i+1]
                target_word = processed_couple[i+1]
                if target_word != '<UNK>':
                    self.learning_data.append([learn_couple, target_word])
        print("Число обучающих примеров -", len(self.learning_data), "Длина словаря -", len(self.alphabet_indexes), self.learning_data[:20])

    def init_random_weights(self):
        self.init_random_embeddings(self.get_unique_words(data))

        layers_count_arr = [self.input_len, *[self.hide_layer_height] * self.hide_layers_count, self.output_len,]
        print(layers_count_arr, "Массив длины скрытых слоев")
        for i in range(len(layers_count_arr) - 1):
            self.w_arr.append(np.random.randn(layers_count_arr[i], layers_count_arr[i+1]) * 0.01)
            self.b_arr.append(np.random.randn(1, layers_count_arr[i+1]))

    def init_random_embeddings(self, tokens_arr):
        self.output_len = len(tokens_arr)

        for i, token in enumerate(tokens_arr):
            self.alphabet_emb[token] = np.random.randn(1, self.embedding_len)
            self.alphabet_indexes.append(token)
        self.unk_token_index = self.get_word_index("<UNK>")

    def get_unique_words(self, couples_arr):
        joined_couples = (" ".join(couples_arr).lower()).split()

        word_counts = Counter(joined_couples)

        unique_words = [word for word, count in word_counts.items() if count > self.MIN_FREQ and len(word) < 25]

        unique_words.append('<UNK>')

        print(f"Изначальный размер словаря: {len(word_counts)}")
        print(f"Новый размер словаря (слова встречаются > {self.MIN_FREQ} раз): {len(unique_words)}")

        return unique_words

    def parse_embeddings(self, couple):
        embeddings_arr = []
        for word in couple:
            emb = self.get_word_embedding(word)
            if emb is None:
                emb = self.get_word_embedding("<UNK>")
            embeddings_arr.append( emb )
        return embeddings_arr

    def get_one_hot_output(self, index):
        zeros_arr = np.zeros((1, self.output_len))
        zeros_arr[0][index] = 1
        return zeros_arr

    def set_input(self, input_emb):
        emb_arr = [*input_emb, *[np.zeros((1, self.embedding_len))] * (self.emb_input_len - len(input_emb))]
        return np.concatenate( emb_arr, axis=1 )

    def set_random_learn_couple(self):
        input_emb_batch = []
        output_emb_batch = []
        words_butch = []

        for i in range(self.batch_size):
            couple = random.choice(self.learning_data)
            input_emb_batch.append(self.set_input( self.parse_embeddings(couple[0]) ))
            output_emb_batch.append(self.get_one_hot_output(self.get_word_index(couple[1])))
            words_butch.append(couple)
        # print(input_emb_batch, words_butch)
        self.input = np.concatenate(input_emb_batch, axis=0)
        self.target_output = np.concatenate(output_emb_batch, axis=0)
        self.learn_input_words = words_butch

    def forward(self):
        h_arr = []
        t_arr = []

        last_h = self.input

        for i in range(0, self.hide_layers_count + 1):
            # print("Перемногожение скрытого слоя", i, h_arr)
            t = (last_h @ self.w_arr[i]) + self.b_arr[i]
            t_arr.append(t)

            # print("Прямая проходка", t)
            if i < self.hide_layers_count:
                last_h = utils.relu(t)
                h_arr.append(last_h)
            else:
                last_h = t

        answer = utils.softmax(last_h)
        return {'answer': answer, 'h_arr': h_arr, 't_arr': t_arr}

    def backward(self, ai_answer, target, h_arr, t_arr):
        e = ai_answer - target

        layer_activations = [self.input] + h_arr
        dt = e

        for i in range(len(self.w_arr) - 1, -1, -1):
            dw = layer_activations[i].T @ dt
            db = np.sum(dt, axis=0, keepdims=True)

            dh = dt @ self.w_arr[i].T
            if i > 0:
                dt = dh * utils.relu_deriv(t_arr[i-1])
            else:
                for l, line in enumerate(self.learn_input_words):
                    emb_deriv = np.split(dh[l], self.emb_input_len)#, axis=0)
                    for j, word in enumerate(self.learn_input_words[l][0]):
                        if word == "<UNK>": continue
                        self.alphabet_emb[word] -= emb_deriv[j] * self.learning_speed

            self.w_arr[i] -= dw * self.learning_speed
            self.b_arr[i] -= db * self.learning_speed






AI_CLASS()







# def prepare_data():
#     path = kagglehub.dataset_download("fabdelja/tictactoe")
#     with open(path + "/Tic tac initial results.csv") as csv_file:
#         csv_reader = csv.reader(csv_file, delimiter=',')
#         for row in csv_reader:
#             if row[0] == "MOVE1" or row[-1] != "win": continue
#             now_board = [0, 0, 0, 0, 0, 0, 0, 0, 0]
#             # print(row)
#             for i, item in enumerate(row):
#                 answer_board = [0, 0, 0, 0, 0, 0, 0, 0, 0]
#
#                 if item == "?" or i == 7: break
#                 if i % 2 != 0:
#                     answer_board[int(item)] = 1
#                     data.append([now_board.copy(), answer_board])
#                     print([now_board, answer_board])
#                     now_board[int(item)] = 1
#                 else:
#                     now_board[int(item)] = -1
# prepare_data()


# def running_ai(self, now_input):
#     print(now_input)
#     self.input = now_input
#     answer = np.argmax( self.forward().get('answer') )
#     print("Ответ ИИ", answer)
#     return answer