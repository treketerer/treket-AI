import numpy as np
import random
from tqdm import tqdm

from data_manager import DataManagerClass
import utils
from ai_model import AI_CLASS

import pickle

def main(learning, ai, data_man, config):
    if learning: learn_cycle(ai, data_man, config)
    use_cycle(ai, data_man, config)

def learn_cycle(ai_object, data_man, config):
    print("Модель будет сохранена как", config['ai_model_file_name'])
    default_iter = 0
    if 'default_iter' in config:
        default_iter = config['default_iter']
    config['default_iter'] = default_iter + config['iterations']

    for i in tqdm(range(config['iterations'])):#tqdm():
        if i == round(config['iterations'] / 2) or i == round(config['iterations'] / 1.2):
            ai_object.learning_speed /= 10
            print("\nСкорость обучение уменьшена в 10 раз. Сейчас скорость равна", ai_object.learning_speed)

        ai_object.set_random_learn_couple()
        f = ai_object.forward()

        if i % config['stats_input_steps'] == 0:
            target_indices = [data_man.get_word_index(item[1]) for item in ai_object.learn_input_words]
            loss = np.mean(utils.sparse_cross_entropy_batch(f.get('answer'), target_indices)).item(),
            targets = np.argmax(ai_object.target_output, axis=1)
            answers = np.argmax(f.get('answer'), axis=1)
            accuracy = np.mean(answers == targets) * 100

            config['loss_arr'].append(loss[0])
            config['accuracy_arr'].append(accuracy)

            print(f'\n{i + default_iter} Loss: {loss[0]} Точность: {accuracy}%')


        ai_object.backward( f.get('answer'), ai_object.target_output, f.get('h_arr'), f.get('t_arr'))

    config['w_arr'] = ai_object.w_arr
    config['b_arr'] = ai_object.b_arr
    config['alphabet_emb'] = ai_object.alphabet_emb
    config['alphabet_indexes'] = data_man.alphabet_indexes

    with open(config['ai_model_file_name'], 'wb') as f:
        pickle.dump(config, f)
    print("")

def use_cycle(ai_object, data_man, config):
    for i in range(1000):
        try:
            words_inp = input("Введите слова для ИИ: ").lower()

            for word in words_inp.split():
                if data_man.get_word_index(word) == ai_object.unk_token_index:
                    print(f"Слова {word} нет в словаре!")

            for i in range(2):
                now_couple = words_inp
                for j in range(config['emb_input_len'] - len(now_couple.split())):
                    embeddings = ai_object.parse_embeddings(now_couple.split())
                    ai_object.input = ai_object.parse_input(embeddings)
                    forward = ai_object.forward()

                    logits = forward.get('t_arr')[-1] / config['temperature'] # Получаем логиты до softmax
                    probs = utils.softmax(logits) # Предотвращаем переполнение
                    next_word_idx = np.random.choice(len(probs[0]), p=probs[0]) # Выбираем индекс с temp
                    answer_word = data_man.alphabet_indexes[next_word_idx]
                    if answer_word == "<eos>" or answer_word == now_couple.split()[-1]:
                        now_couple += f"."
                        break

                    now_couple += f" {answer_word}"
                print(i, now_couple)

        except Exception as ex:
            print(ex)



if __name__ == "__main__":
    learning = True
    further_education = True
    config = {
        'ai_model_file_name': "models/pushkin_130k_3x512.pkl",
        'further_education_coef': 'punctuation',
        'data_path': "./data/pushk.txt",  # "C:/Users/Андрей/Desktop/андрей/Чаты/ANYA_BOT.txt",
        'parse_data_by': '.',

        'learning_speed': 0.0005,
        'iterations': 30000,
        'batch_size': 512,
        'temperature': 1.0,
        'MIN_FREQ': 5,

        'embedding_len': 256,
        'emb_input_len': 10,
        'hide_layers_count': 3,
        'hide_layer_height': 512,

        'loss_arr': [],
        'accuracy_arr': [],
        'stats_input_steps': 150
    }

    # config = {
    #     'ai_model_file_name': "models/robin.pkl",
    #     'further_education_coef': '2',
    #     'data_path': "./data/robin.txt", # "C:/Users/Андрей/Desktop/андрей/Чаты/ANYA_BOT.txt",
    #     'parse_data_by': '.',
    #
    #     'learning_speed': 0.001,
    #     'iterations': 44000,
    #     'batch_size': 256,
    #     'temperature': 0.8,
    #     'MIN_FREQ': 2,
    #
    #     'embedding_len': 128,
    #     'emb_input_len': 7,
    #     'hide_layers_count': 2,
    #     'hide_layer_height': 256,
    # }

    temperature = config['temperature']
    if learning is False:
        with open(config['ai_model_file_name'], 'rb') as f:
            loaded_config = pickle.load(f)
            config = loaded_config
            config['temperature'] = temperature
            print(config['learning_speed'])
    elif further_education is True:
        with open(config['ai_model_file_name'], 'rb') as f:
            now_config = config.copy()

            del now_config['embedding_len']
            del now_config['emb_input_len']
            del now_config['hide_layers_count']
            del now_config['hide_layer_height']

            further_education_coef = config['further_education_coef']

            loaded_config = pickle.load(f)
            config = loaded_config

            ai_model_file_name = config['ai_model_file_name'].split('.')
            ai_model_file_name[0] += f"_{further_education_coef}"
            ai_model_file_name = ".".join(ai_model_file_name)

            iters = config['iterations']
            config.update(now_config)
            # config['default_iter'] = iters
            print(config['learning_speed'])

            config['ai_model_file_name'] = ai_model_file_name

            old_size = len(config['alphabet_indexes'])

            if config['parse_data_by'] == '.':
                raw_text = utils.parse_file_data_by_dots(config['data_path'])
            else:
                raw_text = utils.parse_file_data_by_lines(config['data_path'])

            for word in utils.get_unique_words(config['MIN_FREQ'], raw_text):
                if word not in config['alphabet_indexes']:
                    config['alphabet_emb'][word] = np.random.randn(1, config['embedding_len'])
                    config['alphabet_indexes'].append(word)
                    print("В словарь добавлено слово", word)
            print(len(config['alphabet_indexes']))
            new_size = len(config['alphabet_indexes'])

            if new_size > old_size:
                differ = new_size - old_size

                last_hidden_size = config['w_arr'][-1].shape[0]

                # 3. Создаем "добавку" для матрицы весов и вектора смещения
                # Используем ту же инициализацию Хе, что и при создании модели
                new_weights = np.random.randn(last_hidden_size, differ) * np.sqrt(2.0 / last_hidden_size)
                new_biases = np.random.randn(1, differ)

                # 4. "Приклеиваем" новые веса к старым
                config['w_arr'][-1] = np.hstack((config['w_arr'][-1], new_weights))
                config['b_arr'][-1] = np.hstack((config['b_arr'][-1], new_biases))

                print(f"Форма выходной матрицы весов изменена на: {config['w_arr'][-1].shape}")
                print(f"Форма выходного вектора смещения изменена на: {config['b_arr'][-1].shape}")

                x = config.copy()
                del x
                print()
    data_man = DataManagerClass(config)
    ai_object = AI_CLASS(config, data_man)

    main(learning, ai_object, data_man, config)




#
# config = {
#         'ai_model_file_name': "models/akmal_25k_82b.pkl",
#         'further_education_coef': 'further',
#         'data_path': "C:/Users/Андрей/Desktop/андрей/Чаты/AKMAL_BOT.txt",
#
#         'learning_speed': 0.001,
#         'iterations': 20000,
#         'batch_size': 82,
#         'temperature': 1.0,
#         'MIN_FREQ': 2,
#
#         'embedding_len': 82,
#         'emb_input_len': 6,
#         'hide_layers_count': 2,
#         'hide_layer_height': 256,
#     }