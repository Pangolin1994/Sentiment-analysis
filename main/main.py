import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import naive_bayes
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import googlemaps
from googlemaps.geocoding import geocode, reverse_geocode

pd.set_option('display.width', 500)
pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 8)
pd.set_option('display.max_colwidth', 160)


def fillout_nans(frame: pd.DataFrame, column: str, value):
    df = frame.query('{0}.isnull().values'.format(column))
    frame.loc[df.index.values, column] = value


def extract_address(response):
    return response[0]['formatted_address']


def extract_location(response):
    return response[0]['geometry']['location']


def find_locations(df: pd.DataFrame, client):
    no_loc = 'tweet_location.isnull().values & tweet_coord.notnull().values'
    df_noloc = df.query(no_loc).loc[:, ['tweet_coord', 'tweet_location']]
    addresses = []
    for row in range(len(df_noloc)):
        coords = df_noloc.iloc[row, 0].strip('[]')
        response = reverse_geocode(client, coords)
        if response:
            address = extract_address(response)
            addresses.append(address)
        else:
            addresses.append('Unknown location')
    df.loc[df_noloc.index.values, 'tweet_location'] = addresses


def find_coordinates(df: pd.DataFrame, client):
    no_coord = 'tweet_coord.isnull().values & tweet_location.notnull().values'
    df_nocoord = df.query(no_coord).loc[:, ['tweet_coord', 'tweet_location']]
    locations = []
    for row in range(len(df_nocoord)):
        loc = df_nocoord.iloc[row, 1]
        response = geocode(client, loc)
        if response:
            coords = extract_location(response)
            lat, long = coords['lat'], coords['lng']
            locations.append((lat, long))
        else:
            locations.append('Unknown coordinates')
    df.loc[df_nocoord.index.values, 'tweet_coord'] = locations


def main():
    df = pd.read_csv('../data/Tweets.csv')

    # Информация о фрейме
    print(df.info())
    print(df.shape)

    # Удаление столбца 'airline_sentiment_gold', содержащего помимо значений NaN
    # информацию идентичную информации в столбце airline_sentiment
    df_identical_cols = df.query('airline_sentiment_gold.notnull().values') \
                          .loc[:, ['airline_sentiment', 'airline_sentiment_gold']]
    print(df_identical_cols.head(10))
    df.drop(columns='airline_sentiment_gold', inplace=True)

    # В случае не негативного твита отсутствует причина негатива (значение NaN)
    # Установка отсутствия причины в столбце 'negativereason'
    fillout_nans(df, 'negativereason', 'No reason')

    # Большинство позитивных и нейтральных твитов отмечено нулевой уверенностью в негативе твита
    # Если уверенность в негативности твита отсутствует (т.е. NaN),
    # тогда заполняю недостающие значения в столбце 'negativereason_confidence' значением 0.0
    fillout_nans(df, 'negativereason_confidence', 0.0)

    # Столбцы с длинными именами переименовываю для удобства отображения
    # Преобразование делаем без потери читабельности и смысла
    df.rename(
        columns={'airline_sentiment': 'sentiment',
                 'airline_sentiment_confidence': 'sentiment_confidence'},
        inplace=True
    )

    # Удаление дублирующихся строк и индексация фрейма значениями в естественном порядке
    unique_ser = df.duplicated()
    if len(unique_ser[unique_ser]) < len(df):
        df = df.drop_duplicates()
    df = df.reindex(np.arange(len(df)))

    # Проверяю, что сообщения твитов и столбец настроения не пусты
    # Если такие твиты есть - удаляю их по причине отсутствия информативности в данных
    # Дополнительно мною обнаружено, что в обоих фреймах и все остальные столбцы состоят из значений NaN
    df_nosent = df.query('sentiment.isnull().values')
    df_notext = df.query('text.isnull().values')

    # Проверка на наличие в каждой клетке фрейма значения NaN
    print(df_notext != df_nosent)

    # Фреймы абсолютно идентичны друг другу
    print(df_notext.index.values == df_nosent.index.values)

    # Удаление строк
    df.drop(df_notext.index.values, inplace=True)

    # Работа с геоданными фрейма: локациями и геокоординатами с помощью GoogleMaps
    client = googlemaps.Client('AIzaSyBBJyrkuTdzQGG30_Dc4kboECFjP6bM43I')
    # Поиск места расположения твитнувшего по координатам (широта, долгота)
    find_locations(df, client)
    # Поиск координат твитнувшего по геолокации
    # Вызов закомментирован из-за слишком долгой работы при обращении к геосервису
    # Причина - в подавляющем ко-ве отсутствующих данных о локациях в фрейме (около 2/3)
    # find_coordinates(df, client)

    sentiments = df['sentiment']
    texts = df['text']

    # Сохранение меток классов для будущей классификации
    labels = sentiments.unique().tolist()

    # Конвертация категорий твитов в числовые значения
    le = LabelEncoder()
    le.fit(labels)
    enc_labels = le.transform(sentiments)

    # Составление словаря из слов тренировочных твитов с учётом 2000 наиболее частых слов в кач-ве признаков
    # Исключение из признаков малоинформативные слова английского словаря (he, the, and and etc.)
    vectorizer = CountVectorizer(max_features=3000,
                                 stop_words=stopwords.words('english'))
    vectorizer.fit(texts)
    # Получаю разреженную матрицу размера [кол-во твитов, кол-во слов в словаре]
    # Значение [i,j] - кол-во повторений j в i твите
    matrix = vectorizer.transform(texts)

    # Разделяю фрейм на 2: тренировочный и тестовый в долях 0.85 и 0.15 соответственно
    train_text, test_text, train_sents, test_sents = train_test_split(matrix, enc_labels, test_size=0.15)

    # Использую Multinomial Naive Bayes classifier в кач-ве алгоритма классификации на данных
    nb = naive_bayes.MultinomialNB()
    nb.fit(train_text, train_sents)
    predicted = nb.predict(test_text)

    # Вывод точности предсказаний - соотношение верных к общему кол-ву предсказаний
    accuracy = accuracy_score(test_sents, predicted)
    print(accuracy)


if __name__ == '__main__':
    main()
