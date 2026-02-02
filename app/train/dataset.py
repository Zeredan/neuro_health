import pandas as pd
import numpy as np
from torch.utils.data import Dataset, IterableDataset
import torch
from typing import List, Tuple, Dict, Any, Iterator, Optional
from pathlib import Path
from datetime import datetime


class PatientSequenceDataset(IterableDataset):
    """
    Dataset для работы с последовательностями случаев пациентов.
    Читает данные из TSV потоково, не загружая все в память.
    Генерирует окна вида [start:start+window_size] -> [start+window_size+1]
    """
    
    def __init__(
        self,
        tsv_path: str,
        min_sequence_length: int = 5,
        max_sequence_length: Optional[int] = None,
        window_stride: int = 1,
        chunk_size: int = 10000,  # Размер чанка для чтения TSV
    ):
        """
        Args:
            tsv_path: Путь к TSV файлу с данными
            min_sequence_length: Минимальная длина последовательности для обучения
            max_sequence_length: Максимальная длина последовательности (если None - без ограничений)
            window_stride: Шаг скользящего окна (обычно 1)
            chunk_size: Размер чанка для потокового чтения TSV
        """
        self.tsv_path = tsv_path
        self.min_sequence_length = min_sequence_length
        self.max_sequence_length = max_sequence_length
        self.window_stride = window_stride
        self.chunk_size = chunk_size
        
        # Проверим существование файла
        try:
            # Читаем только первую строку для проверки
            with open(tsv_path, 'r', encoding='utf-8') as f:
                self.columns = f.readline().strip().split('\t')
            print(f"✓ Файл найден: {tsv_path}")
            print(f"  Колонок в файле: {len(self.columns)}")
            print(f"  Первые 5 колонок: {self.columns[:5]}")
        except FileNotFoundError:
            print(f"✗ ОШИБКА: Файл не найден: {tsv_path}")
            raise
        
        # Предполагаемые колонки (на основе описания)
        self.expected_columns = [
            'ENP', 'SEX', 'CASE_START_DATE', 'AGE', 'DIAGNOSIS', 
            'SERVICE', 'GROUP', 'PROFILE', 'RESULT', 'TYPE', 'FORM', 'IS_DEAD'
        ]
        
        # Проверяем наличие всех колонок
        missing_cols = []
        for col in self.expected_columns:
            if col not in self.columns:
                missing_cols.append(col)
        
        if missing_cols:
            print(f"  ⚠ Предупреждение: Отсутствуют колонки: {missing_cols}")
        else:
            print("  ✓ Все ожидаемые колонки присутствуют")
        
        print()
    
    def _read_patient_data(self) -> Iterator[Tuple[str, pd.DataFrame]]:
        """
        Генератор, который читает TSV потоково и группирует по пациентам.
        Возвращает (enp, dataframe_для_пациента) для каждого пациента.
        """
        print("Начинаем потоковое чтение TSV файла...")
        
        # Считываем TSV чанками (используем sep='\t' для табуляции)
        chunk_reader = pd.read_csv(
            self.tsv_path,
            sep='\t',
            chunksize=self.chunk_size,
            dtype={
                'ENP': str,
                'SEX': 'category',
                'AGE': float,
                'DIAGNOSIS': str,
                'SERVICE': str,
                'GROUP': 'category',
                'PROFILE': 'category',
                'RESULT': 'category',
                'TYPE': 'category',
                'FORM': 'category',
                'IS_DEAD': 'category'
            },
            parse_dates=['CASE_START_DATE'],
            encoding='utf-8'
        )
        
        current_patient = None
        patient_records = []
        chunk_counter = 0
        
        for chunk in chunk_reader:
            chunk_counter += 1
            
            if chunk_counter % 10 == 0:
                print(f"  Обработан чанк #{chunk_counter}")
            
            # Сортируем внутри чанка по ENP и дате (на всякий случай)
            chunk = chunk.sort_values(['ENP', 'CASE_START_DATE'])
            
            for _, row in chunk.iterrows():
                enp = str(row['ENP'])
                
                if current_patient is None:
                    current_patient = enp
                    patient_records.append(row)
                elif enp == current_patient:
                    patient_records.append(row)
                else:
                    # Новый пациент - отдаем накопленные данные
                    if patient_records:
                        df_patient = pd.DataFrame(patient_records)
                        yield current_patient, df_patient
                    
                    # Начинаем собирать для нового пациента
                    current_patient = enp
                    patient_records = [row]
        
        # Отдаем последнего пациента
        if patient_records:
            df_patient = pd.DataFrame(patient_records)
            yield current_patient, df_patient
        
        print(f"✓ Потоковое чтение завершено. Обработано чанков: {chunk_counter}")
        print()
    
    def _process_diagnosis_string(self, diagnosis_str: str) -> List[str]:
        """
        Обрабатывает строку диагнозов.
        DIAGNOSIS может быть строкой с разделителями - ПРОБЕЛАМИ.
        Пример: "I10 I11 E11.9"
        """
        if pd.isna(diagnosis_str) or diagnosis_str == '':
            return []
        
        # Преобразуем в строку и разделяем по пробелам
        diagnosis_str = str(diagnosis_str).strip()
        
        # Разделяем по одному или нескольким пробелам
        diagnoses = diagnosis_str.split()
        
        # Очищаем от пустых строк и лишних пробелов
        diagnoses = [d.strip() for d in diagnoses if d.strip()]
        
        return diagnoses
    
    def _get_season_from_date(self, date) -> int:
        """
        Определяет сезон по дате.
        Возвращает индекс сезона:
        0: Pad (служебное)
        1: Unknown (Unk)
        2: Зима (декабрь, январь, февраль)
        3: Весна (март, апрель, май)
        4: Лето (июнь, июль, август)
        5: Осень (сентябрь, октябрь, ноябрь)
        """
        if pd.isna(date):
            return 1  # Unknown, если дата не определена
        
        # Если дата - строка, преобразуем в datetime
        if isinstance(date, str):
            try:
                # Формат: 01.01.2019
                date = datetime.strptime(date, '%d.%m.%Y')
            except ValueError:
                try:
                    # Пробуем другие форматы
                    date = pd.to_datetime(date, dayfirst=True, errors='coerce')
                    if pd.isna(date):
                        return 1  # Unknown при ошибке преобразования
                except:
                    return 1  # Unknown при ошибке преобразования
        
        month = date.month
        if month in [12, 1, 2]:
            return 2  # Зима
        elif month in [3, 4, 5]:
            return 3  # Весна
        elif month in [6, 7, 8]:
            return 4  # Лето
        else:  # 9, 10, 11
            return 5  # Осень
    
    def _generate_windows_for_patient(self, patient_df: pd.DataFrame) -> List[Tuple[pd.DataFrame, pd.Series]]:
        """
        Генерирует окна для одного пациента.
        Возвращает список кортежей (окно_признаков, целевой_случай)
        """
        windows = []
        n_cases = len(patient_df)
        
        if n_cases <= self.min_sequence_length:
            # Слишком короткая последовательность
            return windows
        
        # Определяем максимальную длину окна
        max_len = self.max_sequence_length if self.max_sequence_length else n_cases - 1
        
        # Генерируем окна
        for start_idx in range(0, 1):#n_cases - self.min_sequence_length, self.window_stride):
            # Длина окна увеличивается от min_sequence_length до max_len
            for window_size in range(self.min_sequence_length, min(max_len, n_cases - start_idx)):
                end_idx = start_idx + window_size
                target_idx = end_idx  # Предсказываем следующий после окна
                
                if target_idx >= n_cases:
                    break
                
                window_df = patient_df.iloc[start_idx:end_idx].copy()
                target_row = patient_df.iloc[target_idx].copy()
                
                windows.append((window_df, target_row))
                
                # Если достигли максимальной длины окна, выходим из внутреннего цикла
                if window_size >= max_len:
                    break
        
        return windows
    
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """
        Итератор по датасету.
        Возвращает словарь с окном данных и целевым случаем.
        """
        patient_generator = self._read_patient_data()
        total_patients = 0
        total_windows = 0
        
        for patient_idx, (enp, patient_df) in enumerate(patient_generator):
            total_patients += 1
            
            if patient_idx % 100 == 0:
                print(f"Обработан пациент #{patient_idx + 1}: ENP={enp}, случаев={len(patient_df)}")
            
            # Генерируем окна для этого пациента
            windows = self._generate_windows_for_patient(patient_df)
            window_count = len(windows)
            total_windows += window_count
            
            for window_idx, (window_df, target_row) in enumerate(windows):
                # Добавляем сезоны для каждого случая в окне
                window_seasons = [self._get_season_from_date(date) for date in window_df['CASE_START_DATE']]
                target_season = self._get_season_from_date(target_row['CASE_START_DATE'])
                
                # Подготавливаем данные для окна
                window_data = {
                    'enp': enp,
                    'patient_idx': patient_idx,
                    'window_idx': window_idx,
                    
                    # Окно признаков (история)
                    'window_sex': window_df['SEX'].values.tolist(),
                    'window_age': window_df['AGE'].values.tolist(),
                    'window_diagnosis': [self._process_diagnosis_string(d) for d in window_df['DIAGNOSIS'].values],
                    'window_service': window_df['SERVICE'].values.tolist(),
                    'window_group': window_df['GROUP'].values.tolist(),
                    'window_profile': window_df['PROFILE'].values.tolist(),
                    'window_result': window_df['RESULT'].values.tolist(),
                    'window_type': window_df['TYPE'].values.tolist(),
                    'window_form': window_df['FORM'].values.tolist(),
                    'window_is_dead': window_df['IS_DEAD'].values.tolist(),
                    'window_dates': window_df['CASE_START_DATE'].values.tolist(),
                    'window_season': window_seasons,  # НОВОЕ: сезоны
                    
                    # Целевой случай (то, что предсказываем)
                    'target_sex': target_row['SEX'],
                    'target_age': float(target_row['AGE']),
                    'target_diagnosis': self._process_diagnosis_string(target_row['DIAGNOSIS']),
                    'target_service': target_row['SERVICE'],
                    'target_group': target_row['GROUP'],
                    'target_profile': target_row['PROFILE'],
                    'target_result': target_row['RESULT'],
                    'target_type': target_row['TYPE'],
                    'target_form': target_row['FORM'],
                    'target_is_dead': target_row['IS_DEAD'],
                    'target_date': target_row['CASE_START_DATE'],
                    'target_season': target_season,  # НОВОЕ: сезон целевого случая
                }
                
                yield window_data
            
            # Очищаем память
            del patient_df, windows
        
        print()
        print("=" * 60)
        print(f"ИТОГО:")
        print(f"  Обработано пациентов: {total_patients}")
        print(f"  Сгенерировано окон: {total_windows}")
        print("=" * 60)
        print()
    
    def get_patient_count(self) -> int:
        """
        Быстрая оценка количества пациентов (читает только ENP колонку).
        Может быть неточной при потоковом чтении, но дает представление.
        """
        try:
            # Читаем только колонку ENP для подсчета уникальных значений
            enp_series = pd.read_csv(
                self.tsv_path, 
                sep='\t', 
                usecols=['ENP'],
                encoding='utf-8'
            )
            unique_patients = enp_series['ENP'].nunique()
            print(f"≈ Количество уникальных пациентов: {unique_patients}")
            return unique_patients
        except Exception as e:
            print(f"✗ Не удалось подсчитать пациентов: {e}")
            return 0
    
    def get_window_count_estimate(self) -> int:
        """
        Более точная оценка количества окон (требует полного прохода по данных).
        Использовать осторожно для больших файлов!
        """
        print("Начинаем точный подсчет окон...")
        window_count = 0
        patient_count = 0
        
        for patient_idx, (enp, patient_df) in enumerate(self._read_patient_data()):
            patient_count += 1
            n_cases = len(patient_df)
            
            if n_cases <= self.min_sequence_length:
                continue
            
            max_len = self.max_sequence_length if self.max_sequence_length else n_cases - 1
            
            # Аналитическая формула для подсчета окон
            for start_idx in range(0, n_cases - self.min_sequence_length, self.window_stride):
                possible_windows = min(max_len, n_cases - start_idx) - self.min_sequence_length + 1
                window_count += max(0, possible_windows)
            
            # Показываем прогресс
            if patient_count % 500 == 0:
                print(f"  Обработано пациентов: {patient_count}, окон: {window_count}")
        
        print()
        print(f"✓ Точная оценка количества окон: {window_count}")
        print(f"  Общее количество пациентов: {patient_count}")
        print()
        
        return window_count
    
    def __len__(self) -> int:
        """
        Приблизительная длина датасета.
        Внимание: точный подсчет требует полного прохода по данных!
        """
        # Используем быструю оценку
        patient_count = self.get_patient_count()
        
        # Предположим в среднем по 10 окон на пациента
        estimated_windows = patient_count * 10 if patient_count > 0 else 0
        
        print(f"≈ Предполагаемое количество окон: {estimated_windows}")
        print()
        
        return estimated_windows


# Пример использования
if __name__ == "__main__":
    
    current_file = Path(__file__).parent
    project_root = current_file.parent.parent
    resource_path = project_root / "res" / "datasets" / "test_dataset.tsv"
    
    print("=" * 60)
    print("ТЕСТ DATASET С СЕЗОНАМИ")
    print("=" * 60)
    print()

    # Создаем датасет
    dataset = PatientSequenceDataset(
        tsv_path=resource_path,
        min_sequence_length=5,
        window_stride=1,
        chunk_size=5000
    )
    
    print()
    print("-" * 60)
    print("ПРОВЕРКА ОБРАБОТКИ ДИАГНОЗОВ")
    print("-" * 60)
    
    # Проверка обработки диагнозов
    test_strings = ["I10 I11 E11.9", "C34.1 D12.6", "A01.1", "", "  I20  I21  "]
    
    for test_str in test_strings:
        result = dataset._process_diagnosis_string(test_str)
        print(f"  '{test_str}'")
        print(f"    → {result}")
        print()
    
    print("-" * 60)
    print("ПРОВЕРКА СЕЗОНОВ")
    print("-" * 60)
    
    # Проверка определения сезонов
    test_dates = [
        pd.Timestamp("2023-01-15"),  # Зима
        pd.Timestamp("2023-03-20"),  # Весна
        pd.Timestamp("2023-07-10"),  # Лето
        pd.Timestamp("2023-10-05"),  # Осень
        pd.Timestamp("2023-12-25"),  # Зима
    ]
    
    
    print()
    print("-" * 60)
    print("ПРОСМОТР ПЕРВЫХ ОКОН")
    print("-" * 60)
    print()
    
    # Пример: итерация по первым нескольким окнам
    max_samples = 30
    samples_collected = 0
    
    print(f"Собираем первые {max_samples} окна:")
    print()
    
    for i, data in enumerate(dataset):
        if samples_collected >= max_samples:
            break
        
        samples_collected += 1
        
        print(f"ОКНО #{samples_collected}")
        print(f"  Пациент: {data['enp']}")
        print(f"  Длина окна: {len(data['window_age'])} случаев")
        print()
        
        print(f"  ПЕРВЫЙ СЛУЧАЙ В ОКНЕ:")
        print(f"    • Возраст: {data['window_age'][0]:.2f}")
        print(f"    • Пол: {data['window_sex'][0]}")
        print(f"    • Сезон: {data['window_season'][0]}")
        
        if data['window_diagnosis'][0]:
            print(f"    • Диагнозы: {data['window_diagnosis'][0]}...")
        else:
            print(f"    • Диагнозы: []")
        
        print(f"    • Услуга: {data['window_service'][0]}")
        print()
        
        print(f"  ЦЕЛЕВОЙ СЛУЧАЙ:")
        print(f"    • Возраст: {data['target_age']:.2f}")
        print(f"    • Пол: {data['target_sex']}")
        print(f"    • Сезон: {data['target_season']}")
        
        if data['target_diagnosis']:
            print(f"    • Диагнозы: {data['target_diagnosis'][:5]}")
        else:
            print(f"    • Диагнозы: []")
        
        print(f"    • Услуга: {data['target_service']}")
        print()
        
        print("-" * 40)
        print()
    
    print("=" * 60)
    print("ТЕСТ ЗАВЕРШЕН")
    print("=" * 60)