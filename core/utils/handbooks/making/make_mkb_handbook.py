# scripts/convert_icd10_xlsx_to_csv.py
"""
Конвертирует XLSX справочник МКБ-10 в CSV с 4-уровневым кодированием.
Структура выходного CSV:
- mkb_code: оригинальный код (E11.9)
- letter_idx: индекс буквы (E → 3)
- hierarchy_idx: иерархический код из колонки RN
- global_idx: сквозной уникальный индекс
- description: описание диагноза
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import json
import os
from pathlib import Path

class ICD10XLSXConverter:
    """Конвертер XLSX → CSV для МКБ-10"""
    
    def __init__(self, xlsx_path: str):
        self.xlsx_path = xlsx_path
        self.df = None
        self.vocabs = {}
        
    def load_xlsx(self) -> pd.DataFrame:
        """Загружает XLSX файл"""
        print(f"📂 Загрузка файла: {self.xlsx_path}")
        
        # Загружаем XLSX
        self.df = pd.read_excel(self.xlsx_path)
        
        # Проверяем наличие нужных колонок
        required_columns = ['RN', 'MKB_CODE', 'MKB_NAME']
        missing_cols = [col for col in required_columns if col not in self.df.columns]
        
        if missing_cols:
            raise ValueError(f"Отсутствуют колонки: {missing_cols}. "
                           f"Доступные колонки: {list(self.df.columns)}")
        
        print(f"✅ Загружено {len(self.df)} строк")
        print(f"📊 Колонки: {list(self.df.columns)}")
        print(f"🔍 Первые 5 строк:")
        print(self.df[['RN', 'MKB_CODE', 'MKB_NAME']].head())
        
        return self.df
    
    def clean_and_prepare(self) -> pd.DataFrame:
        """Очищает и подготавливает данные"""
        print("\n🧹 Очистка данных...")
        
        # 1. Удаляем строки с пустыми кодами
        initial_count = len(self.df)
        self.df = self.df.dropna(subset=['MKB_CODE'])
        print(f"   Удалено {initial_count - len(self.df)} строк с пустыми кодами")
        
        # 2. Приводим коды к строковому типу и очищаем
        self.df['MKB_CODE'] = self.df['MKB_CODE'].astype(str).str.strip().str.upper()
        
        # 3. Приводим описания к строковому типу
        self.df['MKB_NAME'] = self.df['MKB_NAME'].astype(str).str.strip()
        
        # 4. Обрабатываем RN (иерархический код)
        self.df['RN'] = pd.to_numeric(self.df['RN'], errors='coerce')
        
        # 5. Удаляем дубликаты по коду МКБ
        self.df = self.df.drop_duplicates(subset=['MKB_CODE'])
        print(f"   Удалено дубликатов, осталось {len(self.df)} уникальных кодов")
        
        return self.df
    
    def extract_letter_from_code(self, mkb_code: str) -> str:
        """Извлекает букву из кода МКБ"""
        if not mkb_code or pd.isna(mkb_code):
            return ''
        
        # Берем первый символ, если он буква
        first_char = str(mkb_code)[0]
        return first_char if first_char.isalpha() else ''
    
    def build_vocabularies(self) -> Dict[str, Dict]:
        """Строит словари для кодирования"""
        print("\n📝 Построение словарей...")
        
        # 1. Словарь для букв
        letters = []
        for code in self.df['MKB_CODE']:
            letter = self.extract_letter_from_code(code)
            if letter and letter not in letters:
                letters.append(letter)
        
        letters = sorted(letters)
        letter_to_idx = {'<PAD>': 0, '<UNK>': 1}
        for idx, letter in enumerate(letters, start=2):
            letter_to_idx[letter] = idx
        
        print(f"   Букв МКБ-10 найдено: {len(letters)} ({', '.join(letters)})")
        
        # 2. Словарь для иерархических кодов (RN)
        # Берем уникальные RN, убираем NaN
        rn_values = self.df['RN'].dropna().unique()
        rn_values = sorted([int(r) for r in rn_values if not pd.isna(r)])
        
        rn_to_idx = {'<PAD>': 0, '<UNK>': 1}
        for idx, rn in enumerate(rn_values, start=2):
            rn_to_idx[rn] = idx
        
        print(f"   Уникальных иерархических кодов (RN): {len(rn_values)}")
        
        # 3. Словарь для полных кодов МКБ (сквозная нумерация)
        codes = sorted(self.df['MKB_CODE'].unique())
        code_to_idx = {'<PAD>': 0, '<UNK>': 1}
        for idx, code in enumerate(codes, start=2):
            code_to_idx[code] = idx
        
        print(f"   Уникальных кодов МКБ: {len(codes)}")
        
        self.vocabs = {
            'letter': letter_to_idx,
            'hierarchy': rn_to_idx,
            'code': code_to_idx
        }
        
        return self.vocabs
    
    def create_output_dataframe(self) -> pd.DataFrame:
        """Создает финальный DataFrame с 5 колонками"""
        print("\n🛠️ Создание выходного DataFrame...")
        
        output_data = []
        
        for _, row in self.df.iterrows():
            mkb_code = row['MKB_CODE']
            rn_value = row['RN']
            description = row['MKB_NAME']
            
            # Извлекаем букву
            letter = self.extract_letter_from_code(mkb_code)
            
            # Получаем индексы из словарей
            letter_idx = self.vocabs['letter'].get(letter, 1)  # 1 = UNK
            hierarchy_idx = self.vocabs['hierarchy'].get(rn_value, 0) if pd.notna(rn_value) else 0  # 0 = PAD
            global_idx = self.vocabs['code'].get(mkb_code, 1)  # 1 = UNK
            
            output_data.append({
                'mkb_code': mkb_code,
                'letter_idx': letter_idx,
                'hierarchy_idx': hierarchy_idx,
                'global_idx': global_idx,
                'description': description
            })
        
        # Создаем DataFrame
        output_df = pd.DataFrame(output_data)
        
        # Сортируем по global_idx для удобства
        output_df = output_df.sort_values('global_idx')
        
        print(f"✅ Создан DataFrame с {len(output_df)} записями")
        print("\n📋 Структура выходных данных:")
        print(output_df[['mkb_code', 'letter_idx', 'hierarchy_idx', 
                        'global_idx', 'description']].head(10))
        
        return output_df
    
    def save_output_csv(self, output_df: pd.DataFrame, output_path: str):
        """Сохраняет DataFrame в CSV"""
        print(f"\n💾 Сохранение CSV в {output_path}")
        
        # Создаем директорию если нужно
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Сохраняем только нужные 5 колонок
        output_df[['mkb_code', 'letter_idx', 'hierarchy_idx', 
                  'global_idx', 'description']].to_csv(
            output_path, 
            index=False,
            encoding='utf-8'
        )
        
        print(f"✅ CSV сохранен: {len(output_df)} строк")
        
        # Также сохраняем полную версию для отладки
        debug_path = output_path.replace('.csv', '_debug.csv')
        output_df.to_csv(debug_path, index=False, encoding='utf-8')
        print(f"📄 Отладочная версия сохранена: {debug_path}")
    
    def save_vocabularies(self, output_dir: str):
        """Сохраняет словари в JSON файлы"""
        print(f"\n📚 Сохранение словарей в {output_dir}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        for vocab_name, vocab_dict in self.vocabs.items():
            file_path = os.path.join(output_dir, f'icd10_{vocab_name}_vocab.json')
            
            # Для JSON сохраняем как есть (ключи-строки)
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(
                    {str(k): v for k, v in vocab_dict.items()},  # Все ключи в строки
                    f, 
                    indent=2, 
                    ensure_ascii=False
                )
            
            print(f"   ✅ {vocab_name}: {len(vocab_dict)} записей → {file_path}")
        
        # Также сохраняем обратные словари (idx → value) для декодирования
        reverse_vocabs = {}
        for vocab_name, vocab_dict in self.vocabs.items():
            reverse_dict = {v: k for k, v in vocab_dict.items()}
            reverse_vocabs[vocab_name] = reverse_dict
            
            file_path = os.path.join(output_dir, f'icd10_{vocab_name}_reverse.json')
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(
                    {str(k): str(v) for k, v in reverse_dict.items()},
                    f,
                    indent=2,
                    ensure_ascii=False
                )
        
        print("✅ Обратные словари сохранены")
        
        return reverse_vocabs
    
    def analyze_data(self):
        """Анализирует данные и выводит статистику"""
        print("\n📊 АНАЛИЗ ДАННЫХ МКБ-10:")
        print("=" * 50)
        
        # 1. Количество кодов по буквам
        letter_counts = {}
        for code in self.df['MKB_CODE']:
            letter = self.extract_letter_from_code(code)
            letter_counts[letter] = letter_counts.get(letter, 0) + 1
        
        print("📈 Количество кодов по буквам:")
        for letter in sorted(letter_counts.keys()):
            if letter:  # Пропускаем пустые
                print(f"   {letter}: {letter_counts[letter]:4d} кодов")
        
        # 2. Распределение по RN (иерархии)
        rn_counts = self.df['RN'].value_counts().head(10)
        print(f"\n🏆 Топ-10 самых больших иерархических групп (RN):")
        for rn, count in rn_counts.items():
            if pd.notna(rn):
                print(f"   RN={rn}: {count} кодов")
        
        # 3. Примеры кодов
        print(f"\n🔍 Примеры кодов МКБ:")
        sample_codes = self.df['MKB_CODE'].head(5).tolist()
        for code in sample_codes:
            letter = self.extract_letter_from_code(code)
            print(f"   {code} → буква '{letter}'")
        
        print("=" * 50)

# Главная функция
def main():
    
    current_path = Path(__file__).parent  # Директория file.py
    xlsx_path = current_path / '..' / '..' / '..' / 'res' / 'datasets' / 'mkb.xlsx'
    xlsx_path = xlsx_path.resolve()

    print(f"Путь к XLSX: {xlsx_path}")
    input()

    # Создаем конвертер
    converter = ICD10XLSXConverter(xlsx_path)
    
    try:
        # 1. Загрузка
        converter.load_xlsx()
        
        # 2. Очистка
        converter.clean_and_prepare()
        
        # 3. Анализ
        converter.analyze_data()
        
        # 4. Построение словарей
        vocabs = converter.build_vocabularies()
        
        # 5. Создание выходного DataFrame
        output_df = converter.create_output_dataframe()
        
        # 6. Сохранение CSV
        save_file_path = current_path / '..' / '..' / '..' / 'res' / 'datasets' / 'mkb_handbook.csv'
        converter.save_output_csv(output_df, save_file_path)
        
        print(f"\n🎉 Конвертация завершена успешно!")
        print(f"📁 Выходные файлы:")
        print(f"   • CSV с кодами: {args.output_csv}")
        print(f"   • Словари: {args.vocab_dir}/*.json")
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        raise



main()