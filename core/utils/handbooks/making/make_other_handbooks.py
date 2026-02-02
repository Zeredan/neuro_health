from pathlib import Path

def fix_csv_simple_cut(input_path, output_path):
    """
    Самая простая обработка:
    1. Убирает кавычку в начале если есть
    2. Берет только первые 5 значений через split(',')
    3. Игнорирует все остальное
    """
    # Читаем с правильной кодировкой
    try:
        with open(input_path, 'r', encoding='utf-8-sig') as f:
            content = f.read()
    except:
        try:
            with open(input_path, 'r', encoding='cp1251') as f:
                content = f.read()
        except:
            print("Ошибка чтения файла")
            return
    
    lines = content.split('\n')
    results = []
    
    for line in lines:
        line = line.strip()
        if not line:
            results.append("")
            continue
        
        # Убираем начальную кавычку
        if line.startswith('"'):
            line = line[1:]
        
        # Разбиваем по запятым и берем первые 5
        parts = line.split(',')
        
        # Берем максимум 5 частей
        if len(parts) > 5:
            parts = parts[:5]
        
        # Убираем лишние символы в последнем поле
        if len(parts) >= 5:
            parts[4] = parts[4].rstrip('";')
        
        # Дополняем до 5 полей если нужно
        while len(parts) < 5:
            parts.append("")
        
        # Собираем обратно
        result_line = ','.join(parts)
        results.append(result_line)
    
    # Сохраняем
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(results))
    
    print(f"Готово! Обработано строк: {len(results)}")
    
    # Показываем пример
    print("\nПример исправления:")
    for i in range(min(3, len(lines))):
        if lines[i].strip():
            print(f"Было: {lines[i].strip()[:60]}...")
            print(f"Стало: {results[i]}")
            print()

# Использование
root = Path(__file__).parent.parent.parent.parent.parent / "res" / "handbooks"
input_file = root / "services_handbook.csv"
output_file = root / "services_handbook_fixed.csv"

fix_csv_simple_cut(input_file, output_file)