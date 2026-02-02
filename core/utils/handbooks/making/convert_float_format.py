import pandas as pd

from pathlib import Path
current_file = Path(__file__).parent
project_root = current_file.parent.parent.parent
resource_path = project_root / "res" / "datasets" / "test_dataset.tsv"

df = pd.read_csv(resource_path, sep='\t', encoding='utf-8')

print("Исходные данные (первые 5 строк):")
print(df.head())
print(f"\nТип столбца AGE: {df['AGE'].dtype}")

# Преобразуем столбец AGE
df['AGE'] = pd.to_numeric(
    df['AGE'].astype(str).str.replace(',', '.', regex=False), 
    errors='coerce'
)

print("\nПосле преобразования (первые 5 строк):")
print(df.head())
print(f"\nНовый тип столбца AGE: {df['AGE'].dtype}")

# Сохраняем обратно в TSV
df.to_csv(resource_path, sep='\t', index=False, encoding='utf-8')
print("\n✓ Файл сохранен: tmp.tsv")