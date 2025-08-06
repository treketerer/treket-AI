import numpy as np
from collections import Counter
import re
import string

def sparse_cross_entropy(z, y):
    return -np.log(z[0, y])

def sparse_cross_entropy_batch(z, y):
    return -np.log(z[np.arange(len(y)), y] + 1e-9)
    # return -np.log(np.array([z[j, y[j]] for j in range(len(y))]))

def clean_text(text: str) -> str:
    """
    Комплексная функция для очистки текста от мусора.
    """
    # Шаг 1: Переводим в нижний регистр
    text = text.lower()

    # Шаг 2: Удаляем URL-адреса
    text = re.sub(r'http\S+|www\.\S+', '', text)

    # Шаг 3: Удаляем эмодзи (улучшенное и более широкое регулярное выражение)
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F700-\U0001F77F"  # alchemical symbols
        "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
        "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
        "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs (включая 🤝)
        "\U0001FA00-\U0001FA6F"  # Chess Symbols
        "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
        "\U00002702-\U000027B0"  # Dingbats
        "\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)

    # Шаг 4: Удаляем цифры
    text = re.sub(r'[0-9]', '', text)

    # Шаг 5: Удаляем специфические символы и расширенную пунктуацию, включая дефисы
    text = re.sub(r'[«»„“"”"‘’`‛—–-]', ' ', text)  # Заменяем на пробел, чтобы не склеивать слова

    # Шаг 6: Удаляем стандартную пунктуацию
    # Создаем таблицу для перевода, исключая дефис, т.к. уже обработали его
    # Это также удалит символы вроде `_`, `+`, `=`, и т.д.
    punct_to_remove = string.punctuation.replace('-', '')
    table = str.maketrans('', '', punct_to_remove)
    text = text.translate(table)

    # Шаг 7: Убираем лишние пробелы
    text = " ".join(text.split())

    return text

def softmax(x):
    # print(x)
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)


def relu(x):
    return np.maximum(0, x)


def relu_deriv(x):
    return np.where(x > 0, 1, 0)


def get_unique_words(min_freq, couples_arr):
    joined_couples = (" ".join(couples_arr).lower()).split()

    word_counts = Counter(joined_couples)

    unique_words = [word for word, count in word_counts.items() if count > min_freq and len(word) < 25]

    unique_words.append('<unk>')
    unique_words.append('<eos>')

    print(f"Изначальный размер словаря: {len(word_counts)}")
    print(f"Новый размер словаря (слова встречаются > {min_freq} раз): {len(unique_words)}")

    return unique_words

def parse_file_data_by_lines(data_path):
    raw_data = []
    with open(data_path, "r", encoding="utf-8") as file:
        for line in file:
            raw_data.append(line.replace("\n", ""))
    return raw_data

def parse_file_data_by_dots(data_path):
    raw_data = []
    with open(data_path, "r", encoding="utf-8") as file:
        text = file.read()  # Читаем весь текст сразу
        sentences = re.split(r'[-.?!–\n:;]', text)

        # print(sentences)
        for couple in sentences:
            # print(couple)
            if len(couple.split()) < 2: continue
            raw_data.append(couple.strip().lower().replace("\n", ""))
    return raw_data