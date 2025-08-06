import numpy as np
from learndata import data
import random


import kagglehub
import csv


# data = [ 'Я люблю кошек', "Я люблю собак" ]

class AI_CLASS:
    def __init__(self):
        self.learning_speed = 0.01
        self.iterations = 30000
        self.batch_size = 10

        self.embedding_len = 10
        self.emb_input_len = 5

        self.input_len = self.emb_input_len * self.embedding_len
        self.output_len = -1
        self.learning_data = []

        self.alphabet_emb = {}
        self.alphabet_indexes = []

        self.w_arr = []
        self.b_arr = []

        self.hide_layers_count = 1
        self.hide_layer_height = 15

        self.init_random_weights()

        self.input = np.random.randn(1, self.input_len)
        self.target_output = np.random.randn(1, self.output_len)

        self.learn_input_words = {}

        self.init_learning_data(data)

        for i in range(self.iterations):
            self.set_random_learn_couple()
            f = self.forward()
            target_indices = [self.alphabet_indexes.index(item[1]) for item in self.learn_input_words]
            # print(f.get('h_arr'))
            print(i,
                  np.sum(self.sparse_cross_entropy_batch(f.get('answer'), target_indices)),
                  "Цели:", np.argmax(self.target_output, axis=1).tolist(),
                  "Ответы:", np.argmax(f.get('answer'), axis=1).tolist(),
                  )
            self.backward( f.get('answer'), self.target_output, f.get('h_arr'), f.get('t_arr'))
        print("")

        for i in range(100):
            words_inp = input("Введите слова для ИИ: ").lower()
            print(words_inp.lower().split())
            embeddings = self.parse_embeddings(words_inp.split())
            self.input = self.set_input(embeddings)
            f = self.forward()
            argmax = np.argmax(f.get('answer')[0])
            print("Ответ:", self.alphabet_indexes[argmax])


    def running_ai(self, now_input):
        print(now_input)
        self.input = now_input
        answer = np.argmax( self.forward().get('answer') )
        print("Ответ ИИ", answer)
        return answer

    def sparse_cross_entropy(self, z, y):
        return -np.log(z[0, y])

    def sparse_cross_entropy_batch(self, z, y):
        return -np.log(z[np.arange(len(y)), y] + 1e-9)
        # return -np.log(np.array([z[j, y[j]] for j in range(len(y))]))

    def softmax(self, x):
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / e_x.sum(axis=1, keepdims=True)

    def init_learning_data(self, learn_data):
        for learn_couple in learn_data:
            couple_arr = learn_couple.lower().split()
            if len(couple_arr) < 2:
                continue
            for i in range(len(couple_arr) - 1):
                self.learning_data.append([couple_arr[:i+1], couple_arr[i+1]])

    def init_random_weights(self):
        self.init_random_embeddings(self.get_unique_words(data))

        layers_count_arr = [self.input_len, *[self.hide_layer_height] * self.hide_layers_count, self.output_len,]
        print(layers_count_arr, "Массив длины скрытых слоев")
        for i in range(len(layers_count_arr) - 1):
            print('W hide layer num', i, layers_count_arr[i], layers_count_arr[i+1])
            self.w_arr.append(np.random.randn(layers_count_arr[i], layers_count_arr[i+1]) * 0.01)
            self.b_arr.append(np.random.randn(1, layers_count_arr[i+1]))

    def init_random_embeddings(self, tokens_arr):
        tokens_arr.append("null")
        self.output_len = len(tokens_arr)

        for i, token in enumerate(tokens_arr):
            self.alphabet_emb[token] = np.random.randn(1, self.embedding_len)
            self.alphabet_indexes.append(token)

    def get_unique_words(self, couples_arr):
        joined_couples = (" ".join(couples_arr).lower()).split()

        unique_words = []
        for i in joined_couples:
            if not i in unique_words:
                unique_words.append(i)

        return unique_words

    def parse_embeddings(self, couple):
        embeddings_arr = []
        for word in couple:
            emb = self.alphabet_emb.get(word)
            if emb is None:
                emb = self.alphabet_emb.get('null')
            embeddings_arr.append( emb )
        return embeddings_arr

    def activate(self, x): # Это ReLU
        return np.maximum(0, x)

    def deriv_activate(self, x): # Производная ReLU
        return np.where(x > 0, 1, 0)

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
            output_emb_batch.append(self.get_one_hot_output(self.alphabet_indexes.index(couple[1])))
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
                last_h = self.activate(t)
                h_arr.append(last_h)
            else:
                last_h = t

        answer = self.softmax(last_h)
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
                dt = dh * self.deriv_activate(t_arr[i-1])
            else:
                for l, line in enumerate(self.learn_input_words):
                    emb_deriv = np.split(dh[l], self.emb_input_len)#, axis=0)
                    for j, word in enumerate(self.learn_input_words[l][0]):
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
