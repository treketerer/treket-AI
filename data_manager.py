import numpy as np
import random
import utils

class DataManagerClass:
    def __init__(self, config):
        self.data_path = config['data_path']
        self.MIN_FREQ = config['MIN_FREQ']
        self.emb_input_len = config['emb_input_len']
        self.parse_type = config['parse_data_by']

        self.raw_data = self.get_raw_data_from_file(self.data_path)
        self.learning_data = []

        print(len(self.raw_data), self.raw_data[:20])

        if 'alphabet_indexes' in config:
            self.alphabet_indexes = config['alphabet_indexes']
        else:
            self.alphabet_indexes = utils.get_unique_words(self.MIN_FREQ, self.raw_data)

        self.init_learning_data(self.raw_data)

    def get_raw_data_from_file(self, path):
        if self.parse_type == '.':
            raw_text = utils.parse_file_data_by_dots(path)
        else:
            raw_text = utils.parse_file_data_by_lines(path)
        learn_couples = self.parse_couples_in_text(raw_text)
        return learn_couples

    def get_word_index(self, word):
        try:
            index = self.alphabet_indexes.index(word)
            return index
        except:
            index = self.alphabet_indexes.index("<unk>")
            return index

    def parse_couples_in_text(self, text_couples:list):
        answer_couples = []
        for couple in text_couples:
            answer_couples.append(couple)
        return answer_couples

    def init_learning_data(self, raw_learn_data):
        known_words = set(self.alphabet_indexes)
        print("<eos>" in known_words)
        print(raw_learn_data[:20])
        for learn_couple in raw_learn_data:
            learn_couple += " "
            learn_couple += "<eos>"
            couple_arr = learn_couple.lower().split()
            if len(couple_arr) < 2:
                continue

            processed_couple = [word if word in known_words else '<unk>' for word in couple_arr]

            for i in range(1, len(processed_couple)):
                target_word = processed_couple[i]
                if target_word == '<unk>':
                    continue

                min_context_index = max(0, i - self.emb_input_len)
                full_context = processed_couple[min_context_index:i]

                for j in range(len(full_context)):
                    now_couple = full_context[j:]
                    if random.randint(0 + (j - 1) * round(50 / self.emb_input_len), 100) < 50: continue
                    if now_couple:
                        self.learning_data.append([now_couple, target_word])

        print("Число обучающих примеров -", len(self.learning_data), "Длина словаря -", len(self.alphabet_indexes), self.learning_data[:30])


    def get_one_hot_vector(self, vector_len, index):
        zeros_arr = np.zeros((1, vector_len))
        zeros_arr[0][index] = 1
        return zeros_arr

