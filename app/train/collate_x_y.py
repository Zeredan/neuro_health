def collate_train(batch: List[Dict[str, Any]], 
                  vocabs: Dict[str, Any],
                  normalization_stats: Optional[Dict[str, Dict[str, float]]] = None) -> Dict[str, Any]:
    """
    Collate функция с глобальным паддингом диагнозов и нормализацией числовых признаков.
    
    Args:
        batch: Список примеров от Dataset
        vocabs: Агрегированные справочники
        normalization_stats: Статистики для нормализации числовых признаков.
            Формат: {
                'age': {'mean': float, 'std': float},
            }
            Если None, используется нормализация по умолчанию (Z-score с mean=0, std=1)
    
    Returns:
        Словарь с нормализованными тензорами
    """
    # Константы
    MAX_DIAGS_ALLOWED = 15  # Максимум диагнозов на случай (можно менять)
    
    # Значения нормализации по умолчанию
    default_stats = {
        'age': {'mean': 40.2, 'std': 21.0},
        'sex': {'min': 0.0, 'max': 1.0},     # для 0/1 значений
        'is_dead': {'min': 0.0, 'max': 1.0},     # для 0/1 значений
    }
    
    if normalization_stats is None:
        normalization_stats = default_stats
    else:
        # Объединяем с дефолтными значениями
        for key in default_stats:
            if key not in normalization_stats:
                normalization_stats[key] = default_stats[key]
    
    # 1. Сортируем по убыванию длины последовательности
    batch.sort(key=lambda x: len(x['window_age']), reverse=True)
    seq_lengths = [len(x['window_age']) for x in batch]
    batch_size = len(batch)
    max_seq_len = max(seq_lengths)
    
    # 2. Находим максимальное количество диагнозов в батче
    max_diags_in_batch = 0
    for example in batch:
        for case_diagnoses in example['window_diagnosis']:
            max_diags_in_batch = max(max_diags_in_batch, len(case_diagnoses))
    
    # 3. Ограничиваем если слишком много
    if max_diags_in_batch > MAX_DIAGS_ALLOWED:
        print(f"⚠ В батче найдены случаи с {max_diags_in_batch} диагнозами. Обрезаем до {MAX_DIAGS_ALLOWED}")
        max_diags_in_batch = MAX_DIAGS_ALLOWED
    
    print(f"📊 В батче: batch_size={batch_size}, seq_len={max_seq_len}, max_diags={max_diags_in_batch}")
    
    # Инициализируем структуры
    window_data = {
        # Числовые признаки (будут нормализованы)
        'age': [],
        'sex': [],
        'is_dead': [],
        
        # Категориальные признаки (сезон теперь категориальный!)
        'season': [],
        
        # Диагнозы (будут тензоры [B, S, max_diags])
        'diagnosis_letter': [],
        'diagnosis_hierarchy': [],
        'diagnosis_full': [],
        'diagnosis_mask': [],
        
        # Услуги
        'service_letter': [],
        'service_hierarchy': [],
        'service_full': [],
        
        # Остальные категориальные
        'group': [],
        'profile': [],
        'result': [],
        'type': [],
        'form': [],
        
        # Метаданные
        'lengths': torch.tensor(seq_lengths, dtype=torch.long),
    }
    
    target_data = {
        # Числовые цели (будут нормализованы)
        'age': [],
        'sex': [],
        'is_dead': [],
        
        # Категориальные цели
        'season': [],
        
        # Диагнозы цели (главный диагноз)
        'diagnosis_letter': [],
        'diagnosis_hierarchy': [],
        'diagnosis_full': [],
        
        # Услуги цели
        'service_letter': [],
        'service_hierarchy': [],
        'service_full': [],
        
        # Остальные категориальные цели
        'group': [], 'profile': [], 'result': [], 'type': [], 'form': [],
    }
    
    # 4. Обрабатываем каждый пример
    for example in batch:
        seq_len = len(example['window_age'])
        
        # === ОКНО ===
        
        # НОРМАЛИЗАЦИЯ числовых признаков окна
        age_stats = normalization_stats['age']
        sex_stats = normalization_stats['sex']
        is_dead_stats = normalization_stats['is_dead']
        
        # Возраст: Z-score нормализация
        window_age_norm = [(a - age_stats['mean']) / age_stats['std'] for a in example['window_age']]
        window_data['age'].append(torch.tensor(window_age_norm, dtype=torch.float32))
        
        window_sex_float = [float(s) for s in example['window_sex']]
        window_sex_norm = [(s - sex_stats['min']) / (sex_stats['max'] - sex_stats['min']) for s in window_sex_float]
        window_data['sex'].append(torch.tensor(window_sex_norm, dtype=torch.float32))
        
        window_is_dead_float = [float(d) for d in example['window_is_dead']]
        window_is_dead_norm = [(d - is_dead_stats['min']) / (is_dead_stats['max'] - is_dead_stats['min'])
                               for d in window_is_dead_float]
        window_data['is_dead'].append(torch.tensor(window_is_dead_norm, dtype=torch.float32))
        
        # Сезон: категориальный признак (0-3)
        window_data['season'].append(torch.tensor(example['window_season'], dtype=torch.long))
        
        # Диагнозы: создаем тензоры [seq_len, max_diags]
        diag_letter_seq = []
        diag_hierarchy_seq = []
        diag_full_seq = []
        diag_mask_seq = []
        
        for case_diagnoses in example['window_diagnosis']:
            num_diags = len(case_diagnoses)
            
            # Кодируем реальные диагнозы
            case_letter = []
            case_hierarchy = []
            case_full = []
            
            for diag in case_diagnoses[:max_diags_in_batch]:  # обрезаем если нужно
                case_letter.append(vocabs['diagnosis_letter'].get(diag, 1))
                case_hierarchy.append(vocabs['diagnosis_hierarchy'].get(diag, 1))
                case_full.append(vocabs['diagnosis'].get(diag, 1))
            
            # Дополняем PAD если нужно
            if num_diags < max_diags_in_batch:
                pad_count = max_diags_in_batch - num_diags
                case_letter.extend([0] * pad_count)      # PAD = 0
                case_hierarchy.extend([0] * pad_count)
                case_full.extend([0] * pad_count)
            
            # Маска: 1 для реальных диагнозов, 0 для PAD
            case_mask = [1] * min(num_diags, max_diags_in_batch) + \
                       [0] * max(0, max_diags_in_batch - num_diags)
            
            diag_letter_seq.append(case_letter)
            diag_hierarchy_seq.append(case_hierarchy)
            diag_full_seq.append(case_full)
            diag_mask_seq.append(case_mask)
        
        # Преобразуем в тензоры
        window_data['diagnosis_letter'].append(torch.tensor(diag_letter_seq, dtype=torch.long))
        window_data['diagnosis_hierarchy'].append(torch.tensor(diag_hierarchy_seq, dtype=torch.long))
        window_data['diagnosis_full'].append(torch.tensor(diag_full_seq, dtype=torch.long))
        window_data['diagnosis_mask'].append(torch.tensor(diag_mask_seq, dtype=torch.float32))
        
        # Услуги (проще - одна услуга на случай)
        service_letter_seq = []
        service_hierarchy_seq = []
        service_full_seq = []
        
        for service in example['window_service']:
            service_letter_seq.append(vocabs['service_letter'].get(service, 1))
            service_hierarchy_seq.append(vocabs['service_hierarchy'].get(service, 1))
            service_full_seq.append(vocabs['service'].get(service, 1))
        
        window_data['service_letter'].append(torch.tensor(service_letter_seq, dtype=torch.long))
        window_data['service_hierarchy'].append(torch.tensor(service_hierarchy_seq, dtype=torch.long))
        window_data['service_full'].append(torch.tensor(service_full_seq, dtype=torch.long))
        
        # Категориальные признаки
        for cat_name in ['group', 'profile', 'result', 'type', 'form']:
            key = f'window_{cat_name}'
            coded = [vocabs[cat_name].get(str(val), 1) for val in example[key]]
            window_data[cat_name].append(torch.tensor(coded, dtype=torch.long))
        
        # === ЦЕЛИ ===
        
        # НОРМАЛИЗАЦИЯ числовых целей
        # Возраст цели
        target_age_norm = (example['target_age'] - age_stats['mean']) / age_stats['std']
        target_data['age'].append(target_age_norm)
        
        # Пол цели
        target_sex_float = float(example['target_sex'])
        target_sex_norm = (target_sex_float - sex_stats['min']) / (sex_stats['max'] - sex_stats['min'])
        target_data['sex'].append(target_sex_norm)
        
        # is_dead цели
        target_is_dead_float = float(example['target_is_dead'])
        target_is_dead_norm = (target_is_dead_float - is_dead_stats['min']) / (is_dead_stats['max'] - is_dead_stats['min'])
        target_data['is_dead'].append(target_is_dead_norm)
        
        # Сезон цели (категориальный)
        target_data['season'].append(example['target_season'])
        
        # Диагнозы цели (главный диагноз)
        target_diagnoses = example['target_diagnosis']
        if target_diagnoses:
            main_diagnosis = target_diagnoses[0]
            target_data['diagnosis_letter'].append(vocabs['diagnosis_letter'].get(main_diagnosis, 1))
            target_data['diagnosis_hierarchy'].append(vocabs['diagnosis_hierarchy'].get(main_diagnosis, 1))
            target_data['diagnosis_full'].append(vocabs['diagnosis'].get(main_diagnosis, 1))
        else:
            target_data['diagnosis_letter'].append(1)  # UNK
            target_data['diagnosis_hierarchy'].append(1)
            target_data['diagnosis_full'].append(1)
        
        # Услуги цели
        target_service = example['target_service']
        target_data['service_letter'].append(vocabs['service_letter'].get(target_service, 1))
        target_data['service_hierarchy'].append(vocabs['service_hierarchy'].get(target_service, 1))
        target_data['service_full'].append(vocabs['service'].get(target_service, 1))
        
        # Категориальные цели
        for cat_name in ['group', 'profile', 'result', 'type', 'form']:
            key = f'target_{cat_name}'
            val = example[key]
            target_data[cat_name].append(vocabs[cat_name].get(str(val), 1))
    
    # 5. Делаем паддинг последовательностей (по оси S)
    
    def pad_batch(sequences, padding_value=0):
        return pad_sequence(sequences, batch_first=True, padding_value=padding_value)
    
    # Обрабатываем окно
    processed_window = {}
    
    # Числовые признаки (уже нормализованы)
    for key in ['age', 'sex', 'is_dead']:
        padded = pad_batch(window_data[key], padding_value=0.0)
        processed_window[key] = padded.unsqueeze(-1) if padded.dim() == 2 else padded
    
    # Категориальные признаки
    processed_window['season'] = pad_batch(window_data['season'], padding_value=0)
    
    # Диагнозы (уже имеют размер [seq_len, max_diags], нужно только по оси S)
    for key in ['diagnosis_letter', 'diagnosis_hierarchy', 'diagnosis_full', 'diagnosis_mask']:
        padded = pad_batch(window_data[key], padding_value=0)
        processed_window[key] = padded
    
    # Услуги
    for key in ['service_letter', 'service_hierarchy', 'service_full']:
        processed_window[key] = pad_batch(window_data[key], padding_value=0)
    
    # Остальные категориальные
    for cat_name in ['group', 'profile', 'result', 'type', 'form']:
        processed_window[cat_name] = pad_batch(window_data[cat_name], padding_value=0)
    
    processed_window['lengths'] = window_data['lengths']
    
    # 6. Обрабатываем цели
    processed_target = {}
    
    # Числовые цели (уже нормализованы)
    processed_target['age'] = torch.tensor(target_data['age'], dtype=torch.float32).unsqueeze(-1)
    processed_target['sex'] = torch.tensor(target_data['sex'], dtype=torch.float32).unsqueeze(-1)
    processed_target['is_dead'] = torch.tensor(target_data['is_dead'], dtype=torch.float32).unsqueeze(-1)
    
    # Категориальные цели
    processed_target['season'] = torch.tensor(target_data['season'], dtype=torch.long)
    
    # Диагнозы цели
    processed_target['diagnosis_letter'] = torch.tensor(target_data['diagnosis_letter'], dtype=torch.long)
    processed_target['diagnosis_hierarchy'] = torch.tensor(target_data['diagnosis_hierarchy'], dtype=torch.long)
    processed_target['diagnosis_full'] = torch.tensor(target_data['diagnosis_full'], dtype=torch.long)
    
    # Услуги цели
    processed_target['service_letter'] = torch.tensor(target_data['service_letter'], dtype=torch.long)
    processed_target['service_hierarchy'] = torch.tensor(target_data['service_hierarchy'], dtype=torch.long)
    processed_target['service_full'] = torch.tensor(target_data['service_full'], dtype=torch.long)
    
    # Остальные категориальные цели
    for cat_name in ['group', 'profile', 'result', 'type', 'form']:
        processed_target[cat_name] = torch.tensor(target_data[cat_name], dtype=torch.long)
    
    return {
        'window': processed_window,
        'target': processed_target,
        'batch_size': batch_size,
        'max_seq_len': max_seq_len,
        'max_diags': max_diags_in_batch,
        'normalization_stats': normalization_stats,  # возвращаем использованные статистики
        'metadata': {
            'seq_lengths': seq_lengths,
            'max_diags': max_diags_in_batch,
        }
    }