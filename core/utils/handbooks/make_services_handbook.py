"""
Конвертирует справочник услуг в CSV с 4-уровневым кодированием.
Структура выходного CSV:
- service_code: оригинальный код (st1.2.3, ds4.5.6, 123, 1.2.3.4)
- prefix_type: признак ds, st, simple, dotted (2, 3, 4, 5)
- hierarchy_idx: иерархический код из колонки A
- global_idx: сквозной уникальный индекс
- description: описание услуги (если есть)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import json
import os
from pathlib import Path

class ServiceXLSXConverter:
    """Конвертер XLSX → CSV для справочника услуг"""
    
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
        required_columns = ['A', 'TEXTCODE']
        missing_cols = [col for col in required_columns if col not in self.df.columns]
        
        if missing_cols:
            raise ValueError(f"Отсутствуют колонки: {missing_cols}. "
                           f"Доступные колонки: {list(self.df.columns)}")
        
        print(f"✅ Загружено {len(self.df)} строк")
        print(f"📊 Колонки: {list(self.df.columns)}")
        print(f"🔍 Первые 5 строк:")
        print(self.df[['A', 'TEXTCODE']].head())
        
        return self.df
    
    def clean_and_prepare(self) -> pd.DataFrame:
        """Очищает и подготавливает данные"""
        print("\n🧹 Очистка данных...")
        
        # 1. Удаляем строки с пустыми кодами
        initial_count = len(self.df)
        self.df = self.df.dropna(subset=['TEXTCODE'])
        print(f"   Удалено {initial_count - len(self.df)} строк с пустыми кодами")
        
        # 2. Приводим коды к строковому типу и очищаем
        self.df['TEXTCODE'] = self.df['TEXTCODE'].astype(str).str.strip()
        
        # 3. Обрабатываем колонку A (иерархический код)
        self.df['A'] = pd.to_numeric(self.df['A'], errors='coerce')
        
        # 4. Удаляем дубликаты по коду услуги
        self.df = self.df.drop_duplicates(subset=['TEXTCODE'])
        print(f"   Удалено дубликатов, осталось {len(self.df)} уникальных кодов")
        
        return self.df
    
    def determine_prefix_type(self, service_code: str) -> str:
        """Определяет тип префикса кода услуги"""
        if not service_code or pd.isna(service_code):
            return 'simple'
        
        code_str = str(service_code)
        
        # Проверяем начинается ли с ds или st
        if code_str.lower().startswith('ds'):
            return 'ds'
        elif code_str.lower().startswith('st'):
            return 'st'
        # Проверяем есть ли точки в коде (кроме префиксов)
        elif '.' in code_str:
            # Если есть точки и код состоит только из цифр и точек
            code_without_prefix = code_str.lower().replace('ds', '').replace('st', '')
            if all(c.isdigit() or c == '.' for c in code_without_prefix):
                return 'dotted'
        
        # Все остальные случаи
        return 'simple'
    
    def prefix_type_to_idx(self, prefix_type: str) -> int:
        """Преобразует тип префикса в числовой индекс"""
        mapping = {
            'ds': 2,      # ds коды
            'st': 3,      # st коды
            'simple': 4,  # простые коды без точек
            'dotted': 5   # коды с точками (1.2.3.4)
        }
        return mapping.get(prefix_type, 4)  # по умолчанию simple
    
    def build_vocabularies(self) -> Dict[str, Dict]:
        """Строит словари для кодирования"""
        print("\n📝 Построение словарей...")
        
        # 1. Словарь для типов префиксов
        prefix_types = []
        for code in self.df['TEXTCODE']:
            prefix_type = self.determine_prefix_type(code)
            if prefix_type and prefix_type not in prefix_types:
                prefix_types.append(prefix_type)
        
        prefix_types = sorted(prefix_types)
        prefix_to_idx = {'<PAD>': 0, '<UNK>': 1}
        for prefix_type in prefix_types:
            prefix_to_idx[prefix_type] = self.prefix_type_to_idx(prefix_type)
        
        print(f"   Типов префиксов найдено: {len(prefix_types)} ({', '.join(prefix_types)})")
        
        # 2. Словарь для иерархических кодов (A)
        # Берем уникальные A, убираем NaN
        a_values = self.df['A'].dropna().unique()
        a_values = sorted([int(a) for a in a_values if not pd.isna(a)])
        
        a_to_idx = {'<PAD>': 0, '<UNK>': 1}
        for idx, a in enumerate(a_values, start=2):
            a_to_idx[a] = idx
        
        print(f"   Уникальных иерархических кодов (A): {len(a_values)}")
        
        # 3. Словарь для полных кодов услуг (сквозная нумерация)
        codes = sorted(self.df['TEXTCODE'].unique())
        code_to_idx = {'<PAD>': 0, '<UNK>': 1}
        for idx, code in enumerate(codes, start=2):
            code_to_idx[code] = idx
        
        print(f"   Уникальных кодов услуг: {len(codes)}")
        
        self.vocabs = {
            'prefix': prefix_to_idx,
            'hierarchy': a_to_idx,
            'code': code_to_idx
        }
        
        return self.vocabs
    
    def create_output_dataframe(self) -> pd.DataFrame:
        """Создает финальный DataFrame с 5 колонками"""
        print("\n🛠️ Создание выходного DataFrame...")
        
        output_data = []
        
        for _, row in self.df.iterrows():
            service_code = row['TEXTCODE']
            a_value = row['A']
            
            # Получаем описание, если есть соответствующая колонка
            description = row['NAME']
            
            # Определяем тип префикса
            prefix_type = self.determine_prefix_type(service_code)
            
            # Получаем индексы из словарей
            prefix_idx = self.vocabs['prefix'].get(prefix_type, 1)  # 1 = UNK
            hierarchy_idx = self.vocabs['hierarchy'].get(a_value, 0) if pd.notna(a_value) else 0  # 0 = PAD
            global_idx = self.vocabs['code'].get(service_code, 1)  # 1 = UNK
            
            output_data.append({
                'service_code': service_code,
                'prefix_type': prefix_idx,
                'hierarchy_idx': hierarchy_idx,
                'global_idx': global_idx,
                'description': description
            })
        
        # Создаем DataFrame
        output_df = pd.DataFrame(output_data)
        print(f"DDDD: {output_data[['hierarchy_idx']]}")
        
        # Сортируем по global_idx для удобства
        output_df = output_df.sort_values('global_idx')
        
        print(f"✅ Создан DataFrame с {len(output_df)} записями")
        print("\n📋 Структура выходных данных:")
        print(output_df[['service_code', 'prefix_type', 'hierarchy_idx', 
                        'global_idx', 'description']].head(10))
        
        return output_df
    
    def save_output_csv(self, output_df: pd.DataFrame, output_path: str):
        """Сохраняет DataFrame в CSV"""
        print(f"\n💾 Сохранение CSV в {output_path}")
        
        # Создаем директорию если нужно
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Сохраняем только нужные 5 колонок
        output_df[['service_code', 'prefix_type', 'hierarchy_idx', 
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
            file_path = os.path.join(output_dir, f'service_{vocab_name}_vocab.json')
            
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
            
            file_path = os.path.join(output_dir, f'service_{vocab_name}_reverse.json')
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
        print("\n📊 АНАЛИЗ ДАННЫХ УСЛУГ:")
        print("=" * 50)
        
        # 1. Количество кодов по типам префиксов
        type_counts = {'ds': 0, 'st': 0, 'simple': 0, 'dotted': 0}
        
        for code in self.df['TEXTCODE']:
            prefix_type = self.determine_prefix_type(code)
            if prefix_type in type_counts:
                type_counts[prefix_type] += 1
        
        print("📈 Распределение кодов по типам:")
        for prefix_type, count in type_counts.items():
            if count > 0:
                print(f"   {prefix_type}: {count:4d} кодов")
        
        # 2. Примеры кодов каждого типа
        print(f"\n🔍 Примеры кодов каждого типа:")
        
        # Собираем примеры
        examples = {'ds': [], 'st': [], 'simple': [], 'dotted': []}
        
        for code in self.df['TEXTCODE']:
            prefix_type = self.determine_prefix_type(code)
            if len(examples[prefix_type]) < 3:
                examples[prefix_type].append(code)
        
        for prefix_type, codes_list in examples.items():
            if codes_list:
                print(f"   {prefix_type}: {', '.join(codes_list[:3])}")
        
        # 3. Распределение по иерархическим кодам (A)
        a_counts = self.df['A'].value_counts().head(10)
        print(f"\n🏆 Топ-10 самых больших иерархических групп (A):")
        for a, count in a_counts.items():
            if pd.notna(a):
                print(f"   A={a}: {count} кодов")
        
        print("=" * 50)
    
    def decode_prefix_type(self, prefix_idx: int) -> str:
        """Декодирует числовой индекс обратно в тип префикса"""
        reverse_mapping = {
            2: 'ds',
            3: 'st',
            4: 'simple',
            5: 'dotted',
            0: '<PAD>',
            1: '<UNK>'
        }
        return reverse_mapping.get(prefix_idx, '<UNK>')

# Главная функция
def main():
    # Определяем путь к файлу
    current_path = Path(__file__).parent  # Директория file.py
    xlsx_path = current_path / '..' / '..' / '..' / 'res' / 'datasets' / 'codeUsl.xlsx'  # предполагаемое имя файла
    xlsx_path = xlsx_path.resolve()
    
    print(f"Путь к XLSX: {xlsx_path}")
    
    if not xlsx_path.exists():
        print(f"❌ Файл не найден: {xlsx_path}")
        print("Пожалуйста, поместите файл справочника услуг в указанную директорию")
        return
    
    # Создаем конвертер
    converter = ServiceXLSXConverter(xlsx_path)
    
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
        save_file_path = current_path / '..' / '..' / '..' / 'res' / 'datasets' / 'services_handbook.csv'
        #converter.save_output_csv(output_df, save_file_path)
        
        print(f"\n🎉 Конвертация завершена успешно!")
        print(f"📁 Выходные файлы:")
        print(f"   • CSV с кодами: {save_file_path}")
        print(f"   • Словари: {vocab_dir}/*.json")
        
        # Дополнительная проверка
        print(f"\n🔍 Проверка первых 10 записей:")
        for i, row in output_df.head(10).iterrows():
            prefix_type = converter.decode_prefix_type(row['prefix_type'])
            print(f"   {row['service_code']} -> тип: {prefix_type}({row['prefix_type']}), "
                  f"иерархия: {row['hierarchy_idx']}, global: {row['global_idx']}")
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()



main()