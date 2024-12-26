import re

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

class Cleaner:
    def __init__(self):
        self._drop_list = [
            'Unnamed: 0', 'id', 'imgUrl', 'allPrice', 'sale', 'annotation', 'isbn',
            'bookName', 'datePublisher', 'da', 'db', 'dc', 'weight', 'age', 'bookGenres', 
            'decoration', 'typeObject', 'illustrations', 'groupOfType', 'underGroup',
            'genres', 'authors', 'publisher', 'series', 'sound_module'
        ]

        self._patterns = {
            'embossing_gold': r'тиснение золотом',
            'embossing_silver': r'тиснение серебром',
            'embossing_colored': r'тиснение цветное',
            'embossing_volume': r'тиснение объемное',
            'partial_lacquer': r'частичная лакировка',
            'puffy_cover': r'пухлая обложка',
            'bookmark_ribbon': r'ляссе',
            'super_cover': r'супер',
            'edge_trim_gold': r'обрез золотой',
            'edge_trim_silver': r'обрез серебряный',
            'edge_trim_colored': r'обрез цветной',
            'slipcase_close': r'футляр закрытый',
            'slipcase_open': r'футляр открытый',
            'stickers': r'с наклейками',
            'puzzles': r'с пазлами',
            'movable_elements': r'с подвижными элементами',
            'volume_panorama': r'с объемной панорамой',
            'sound_module': r'со звуковым модулем',
            'toy': r'с игрушкой',
            'magnet': r'с магнитами',
            'glitter': r'глиттер',
            'flocking': r'флокинг',
            'soft_touch': r'покрытие софттач',
            'cutouts': r'вырубка',
            'textile_inserts': r'текстильные и пластиковые вставки'
        }

        self._patterns_ill = {
            'black_white': r'черно-белые',
            'color': r'цветные'
        }

        self._quality = {
            'Газетная': 0, 'Офсет': 1, 'Крафт': 2, 'Типографская': 3, 'Мелованная': 4, 'Картон': 5,
            'Ламинированные': 6, 'Рисовая': 7, 'Дизайнерская бумага': 8, 'Синтетическая': 9, 'ПВХ': 10,
            'Рафлаглосс': 11, 'Ткань': 12
        }

        self._cover = {
            'обл': 0, 'Лист': 1, 'Пакет': 2, 'Blister': 3, 'Jewel-box': 4, 
            'Amarey': 5, 'Blu-Ray': 6, 'карт': 7, 'Обл.': 8, '7Б': 9, 
            '7А': 10, '7Бц': 11, 'Инт': 12, 'Box': 13
        }

        self._select_featers = [
            'pages', 'volume', 'covers', 'pageType', 'rateSize', 'foreign_language', 
            'rate', 'black_white', 'color', 'slipcase_open', 'partial_lacquer', 'bookmark_ribbon',
            'super_cover', 'embossing_volume', 'embossing_colored', 'embossing_gold', 'edge_trim_colored',
            'edge_trim_gold'
        ]

    def _add_binary_features(df_init, patterns):
        for feature, pattern in patterns.items():
            df_init[feature] = df_init['decoration'].apply(lambda x: 1 if re.search(pattern, x) else 0)
        return df_init

    def _add_binary_features_ill(df_init, patterns_ill):
        for feature, pattern in patterns_ill.items():
            df_init[feature] = df_init['illustrations'].apply(lambda x: 1 if re.search(pattern, str(x).lower()) else 0)
        return df_init

    def __call__(self, table):
        df = table.dropna(subset=['myPrice'])

        df['rate'] = df['rate'].fillna(0.0)
        df['pages'] = df['pages'].fillna(0.0)
        df['rate'] = df['rate'].round()

        df['da'] = df['da'].fillna(df['da'].median())
        df['db'] = df['db'].fillna(df['db'].median())
        df['dc'] = df['dc'].fillna(df['dc'].median())
        df['volume'] = df['da'] * df['db'] * df['dc'] * 10**(-3)

        df['volume'] = df['volume'].round(1)
        df['decoration'] = df['decoration'].fillna('Без декораций').str.lower()

        df = add_binary_features(df, patterns)
        df['typeObject'] = df['typeObject'].fillna('Книги')
        df['foreign_language'] = (df['typeObject'] == 'Книги на иностранном языке').astype(int)

        df['illustrations'] = df['illustrations'].fillna('черно-белые')

        # Применяем функцию
        df = add_binary_features_ill(df, patterns_ill)

        df['pageType'] = df['pageType'].fillna('Газетная')
        df['pageType'] = df['pageType'].map(self._quality)
        df['covers'] = df['covers'].fillna('обл - мягкий переплет')
        df['covers'] = df['covers'].apply(lambda x: x.strip().split(' ')[0])
        df['covers'] = df['covers'].map(cover)
        df['covers'] = df['covers'].fillna(0)

        df = df.drop(columns=self._drop_list)
        return df[self._select_featers]

    def transform(self, dataset):
        continuous_cols = ["pages", "rateSize", "volume"]
        dataset[continuous_cols] = StandardScaler().fit_transform(dataset[continuous_cols])
        dataset = PolynomialFeatures(
            degree=2, interaction_only=True, include_bias=False
        ).fit_transform(dataset)
        return dataset
