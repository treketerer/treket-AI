import numpy as np
import random

import utils

class AI_CLASS:
    def __init__(self, config, data_manager):
        self.data_manager = data_manager

        self.learning_speed = config['learning_speed']
        self.iterations = config['iterations']
        self.batch_size = config['batch_size']

        self.embedding_len = config['embedding_len']
        self.emb_input_len = config['emb_input_len']

        self.hide_layers_count = config['hide_layers_count']
        self.hide_layer_height = config['hide_layer_height']

        self.alphabet_emb = {}
        self.alphabet_indexes = []
        self.unk_token_index = self.data_manager.get_word_index("<unk>")

        self.input_len = self.emb_input_len * self.embedding_len
        self.output_len = -1

        self.w_arr = []
        self.b_arr = []

        if 'w_arr' in config and 'b_arr' in config and 'alphabet_emb' in config:
            self.w_arr = config['w_arr']
            self.b_arr = config['b_arr']
            self.alphabet_emb = config['alphabet_emb']
            self.output_len = len(config['alphabet_indexes'])
        else:
            self.init_random_weights()
            self.init_embeddings(self.data_manager.alphabet_indexes)

        self.input = np.random.randn(1, self.input_len)
        self.target_output = np.random.randn(1, self.output_len)
        self.learn_input_words = {}

    def parse_input(self, input_emb):
        emb_arr = [*input_emb, *[np.zeros((1, self.embedding_len))] * (self.emb_input_len - len(input_emb))]
        return np.concatenate( emb_arr, axis=1 )

    def init_embeddings(self, tokens_arr):
        for i, token in enumerate(tokens_arr):
            self.alphabet_emb[token] = np.random.randn(1, self.embedding_len)
            # self.data_manager.alphabet_indexes.append(token)


    def get_word_embedding(self, word):
        try:
            index = self.alphabet_emb[word]
            return index
        except:
            index = self.alphabet_emb["<unk>"]
            return index

    def set_random_learn_couple(self):
        input_emb_batch = []
        output_emb_batch = []
        words_butch = []

        for i in range(self.batch_size):
            couple = random.choice(self.data_manager.learning_data)
            input_emb_batch.append(self.parse_input( self.parse_embeddings(couple[0]) ))
            output_emb_batch.append(self.data_manager.get_one_hot_vector(self.output_len, self.data_manager.get_word_index(couple[1])))
            words_butch.append(couple)

        self.input = np.concatenate(input_emb_batch, axis=0)
        self.target_output = np.concatenate(output_emb_batch, axis=0)
        self.learn_input_words = words_butch


    def parse_embeddings(self, couple):
        embeddings_arr = []
        for word in couple:
            emb = self.get_word_embedding(word)
            if emb is None:
                emb = self.get_word_embedding("<unk>")
            embeddings_arr.append( emb )
        return embeddings_arr

    def sample_with_temperature(self, logits, temperature=1.0):
        logits = logits / temperature
        # Предотвращаем переполнение
        probs = utils.softmax(logits)
        # Выбираем индекс на основе вероятностей
        next_word_idx = np.random.choice(len(probs[0]), p=probs[0])
        return next_word_idx

    def init_random_weights(self):
        self.output_len = len(self.data_manager.alphabet_indexes)

        layers_count_arr = [self.input_len, *[self.hide_layer_height] * self.hide_layers_count, self.output_len,]
        print(layers_count_arr, "Массив длины скрытых слоев")
        for i in range(len(layers_count_arr) - 1):
            self.w_arr.append(np.random.randn(layers_count_arr[i], layers_count_arr[i+1]) * np.sqrt(2.0 / layers_count_arr[i]))
            self.b_arr.append(np.random.randn(1, layers_count_arr[i+1]))


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
                        if word == "<unk>": continue
                        self.alphabet_emb[word] -= emb_deriv[j] * self.learning_speed

            self.w_arr[i] -= dw * self.learning_speed
            self.b_arr[i] -= db * self.learning_speed






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