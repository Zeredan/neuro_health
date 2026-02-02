import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


def get_age_stats(
    tsv_path: str,
    sample_size: Optional[int] = None,
    chunk_size: int = 10000
) -> Tuple[float, float, dict]:
    """
    Вычисляет статистики возраста из TSV файла.
    
    Args:
        tsv_path: Путь к TSV файлу
        sample_size: Сколько строк проанализировать (если None - все)
        chunk_size: Размер чанка для потокового чтения
        
    Returns:
        Tuple[mean, std, stats_dict]
        stats_dict содержит полную статистику
    """
    logger.info(f"Вычисление статистик возраста из файла: {tsv_path}")
    
    if not Path(tsv_path).exists():
        logger.error(f"Файл не найден: {tsv_path}")
        raise FileNotFoundError(f"Файл не найден: {tsv_path}")
    
    total_rows = 0
    age_sum = 0.0
    age_squared_sum = 0.0
    ages = [] if sample_size else None  # Собираем все возраста если sample_size небольшой
    
    # Читаем файл чанками
    try:
        chunk_reader = pd.read_csv(
            tsv_path,
            sep='\t',
            chunksize=chunk_size,
            usecols=['AGE'],  # Читаем только колонку AGE
            dtype={'AGE': float},
            encoding='utf-8'
        )
        
        for chunk_idx, chunk in enumerate(chunk_reader):
            if sample_size and total_rows >= sample_size:
                break
            
            # Убираем NaN значения
            chunk_ages = chunk['AGE'].dropna().values
            
            if sample_size:
                # Если ограничиваем sample_size, берем только нужное количество
                remaining = sample_size - total_rows
                if remaining < len(chunk_ages):
                    chunk_ages = chunk_ages[:remaining]
            
            chunk_size_actual = len(chunk_ages)
            
            if chunk_size_actual == 0:
                continue
            
            total_rows += chunk_size_actual
            
            # Вычисляем суммы для mean и std
            age_sum += np.sum(chunk_ages)
            age_squared_sum += np.sum(chunk_ages ** 2)
            
            # Сохраняем возраста если нужно для дополнительной статистики
            if ages is not None and sample_size is None:
                ages.extend(chunk_ages.tolist())
            elif ages is not None:
                ages.extend(chunk_ages.tolist())
                if len(ages) > sample_size:
                    ages = ages[:sample_size]
            
            # Логируем прогресс
            if chunk_idx % 10 == 0:
                logger.info(f"Обработано {total_rows} строк...")
            
            if sample_size and total_rows >= sample_size:
                break
        
        # Вычисляем статистики
        if total_rows == 0:
            logger.warning("Не найдено ни одного значения возраста")
            return 0.0, 1.0, {}
        
        mean_age = age_sum / total_rows
        variance = (age_squared_sum / total_rows) - (mean_age ** 2)
        std_age = np.sqrt(max(variance, 0))  # Защита от отрицательной дисперсии
        
        logger.info(f"Найдено {total_rows} значений возраста")
        logger.info(f"Средний возраст: {mean_age:.2f} лет")
        logger.info(f"Стандартное отклонение: {std_age:.2f} лет")
        
        # Дополнительная статистика если собрали все возраста
        stats = {
            'mean': float(mean_age),
            'std': float(std_age),
            'n_samples': total_rows,
            'sum': float(age_sum),
        }
        
        if ages is not None:
            ages_array = np.array(ages)
            stats.update({
                'min': float(np.min(ages_array)),
                'max': float(np.max(ages_array)),
                'median': float(np.median(ages_array)),
                'percentile_25': float(np.percentile(ages_array, 25)),
                'percentile_75': float(np.percentile(ages_array, 75)),
                'percentile_5': float(np.percentile(ages_array, 5)),
                'percentile_95': float(np.percentile(ages_array, 95)),
            })
            
            logger.info(f"Минимальный возраст: {stats['min']:.2f} лет")
            logger.info(f"Максимальный возраст: {stats['max']:.2f} лет")
            logger.info(f"Медиана: {stats['median']:.2f} лет")
            logger.info(f"25-й перцентиль: {stats['percentile_25']:.2f} лет")
            logger.info(f"75-й перцентиль: {stats['percentile_75']:.2f} лет")
        
        return mean_age, std_age, stats
        
    except Exception as e:
        logger.error(f"Ошибка при вычислении статистик возраста: {e}")
        raise





if __name__ == "__main__":
    import sys
    
    # Настройка логирования
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("=" * 60)
    print("🧪 ТЕСТ ФУНКЦИЙ ДЛЯ ВЫЧИСЛЕНИЯ СТАТИСТИК ВОЗРАСТА")
    print("=" * 60)
    
    # Определяем пути
    current_dir = Path(__file__).parent
    project_root = current_dir.parent.parent.parent
    test_data_path = project_root / "res" / "datasets" / "test_dataset.tsv"
    
    print(f"Тестовый файл: {test_data_path}")
    print()
    
    # Тест 1: Прямое вычисление из TSV
    print("\n1. 📊 ВЫЧИСЛЕНИЕ ИЗ TSV ФАЙЛА:")
    print("-" * 40)
        
    mean_age, std_age, stats = get_age_stats(
        str(test_data_path)
    )
        
    print(f"   Средний возраст: {mean_age:.2f} лет")
    print(f"   Стандартное отклонение: {std_age:.2f} лет")
    print(f"   Минимум: {stats.get('min', 'N/A')}")
    print(f"   Максимум: {stats.get('max', 'N/A')}")
    print(f"   Количество образцов: {stats['n_samples']}")