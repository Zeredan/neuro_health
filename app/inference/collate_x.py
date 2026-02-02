def collate_inference(batch: List[Dict[str, Any]], vocabs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Collate функция для инференса (только окно истории, без целевых значений).
    Аналогична collate_train, но не обрабатывает target.
    
    Args:
        batch: Список примеров от PatientSequenceDataset (только window данные)
        vocabs: Агрегированные справочники
        
    Returns:
        Словарь только с window данными для инференса
    """
    # Константы
    MAX_DIAGS_ALLOWED = 15  # Максимум диагнозов на случай (можно менять)
    
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
    
    print(f"📊 [Inference] В батче: batch_size={batch_size}, seq_len={max_seq_len}, max_diags={max_diags_in_batch}")
    
    # Инициализируем структуры ТОЛЬКО для window
    window_data = {
        # Числовые признаки
        'age': [],
        'sex': [],
        'season': [],
        'is_dead': [],
        
        # Диагнозы (будут тензоры [B, S, max_diags])
        'diagnosis_letter': [],
        'diagnosis_hierarchy': [],
        'diagnosis_full': [],
        'diagnosis_mask': [],
        
        # Услуги
        'service_letter': [],
        'service_hierarchy': [],
        'service_full': [],
        
        # Категориальные
        'group': [],
        'profile': [],
        'result': [],
        'type': [],
        'form': [],
        
        # Метаданные
        'lengths': torch.tensor(seq_lengths, dtype=torch.long),
    }
    
    # 4. Обрабатываем каждый пример (ТОЛЬКО window)
    for example in batch:
        seq_len = len(example['window_age'])
        
        # === ТОЛЬКО ОКНО (история) ===
        
        # Числовые признаки
        window_data['age'].append(torch.tensor(example['window_age'], dtype=torch.float32))
        window_data['sex'].append(torch.tensor([int(s) for s in example['window_sex']], dtype=torch.float32))
        window_data['season'].append(torch.tensor(example['window_season'], dtype=torch.long))
        window_data['is_dead'].append(torch.tensor([int(d) for d in example['window_is_dead']], dtype=torch.float32))
        
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
    
    # 5. Делаем паддинг последовательностей (по оси S)
    
    def pad_batch(sequences, padding_value=0):
        return pad_sequence(sequences, batch_first=True, padding_value=padding_value)
    
    # Обрабатываем окно
    processed_window = {}
    
    # Числовые признаки
    for key in ['age', 'sex', 'is_dead']:
        padded = pad_batch(window_data[key], padding_value=0.0)
        processed_window[key] = padded.unsqueeze(-1) if padded.dim() == 2 else padded
    
    processed_window['season'] = pad_batch(window_data['season'], padding_value=0)
    
    # Диагнозы (уже имеют размер [seq_len, max_diags], нужно только по оси S)
    for key in ['diagnosis_letter', 'diagnosis_hierarchy', 'diagnosis_full', 'diagnosis_mask']:
        padded = pad_batch(window_data[key], padding_value=0)
        processed_window[key] = padded
    
    # Услуги
    for key in ['service_letter', 'service_hierarchy', 'service_full']:
        processed_window[key] = pad_batch(window_data[key], padding_value=0)
    
    # Категориальные
    for cat_name in ['group', 'profile', 'result', 'type', 'form']:
        processed_window[cat_name] = pad_batch(window_data[cat_name], padding_value=0)
    
    processed_window['lengths'] = window_data['lengths']
    
    return {
        'window': processed_window,
        'batch_size': batch_size,
        'max_seq_len': max_seq_len,
        'max_diags': max_diags_in_batch,
        'metadata': {
            'seq_lengths': seq_lengths,
            'max_diags': max_diags_in_batch,
        }
    }


def raw_to_result(predictions: Dict[str, torch.Tensor], vocabs: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Преобразует выходы модели (тензоры) обратно в читаемый формат.
    
    Args:
        predictions: Словарь с предсказаниями модели. Ожидаемые ключи:
            - Для диагнозов: 'diagnosis_letter', 'diagnosis_hierarchy', 'diagnosis_full'
            - Для услуг: 'service_letter', 'service_hierarchy', 'service_full'
            - Для числовых: 'age', 'sex', 'is_dead'
            - Для категориальных: 'season', 'group', 'profile', 'result', 'type', 'form'
            (не обязательно все, только то что модель предсказывает)
        
        vocabs: Агрегированные справочники
    
    Returns:
        Список словарей с читаемыми предсказаниями для каждого примера в батче
    """
    batch_size = predictions.get('diagnosis_full', 
                       predictions.get('diagnosis_letter',
                       predictions.get('age', torch.tensor([])))).shape[0]
    
    if batch_size == 0:
        return []
    
    # Создаем обратные справочники для декодирования
    reverse_vocabs = {}
    for name, vocab in vocabs.items():
        if isinstance(vocab, dict):
            reverse_vocabs[name] = {v: k for k, v in vocab.items()}
    
    results = []
    
    for i in range(batch_size):
        result = {}
        
        # Декодируем диагнозы (если есть в predictions)
        if 'diagnosis_letter' in predictions:
            diag_letter_idx = predictions['diagnosis_letter'][i].item()
            result['diagnosis_letter'] = reverse_vocabs.get('diagnosis_letter', {}).get(diag_letter_idx, '<UNK>')
        
        if 'diagnosis_hierarchy' in predictions:
            diag_hier_idx = predictions['diagnosis_hierarchy'][i].item()
            result['diagnosis_hierarchy'] = reverse_vocabs.get('diagnosis_hierarchy', {}).get(diag_hier_idx, '<UNK>')
        
        if 'diagnosis_full' in predictions:
            diag_full_idx = predictions['diagnosis_full'][i].item()
            result['diagnosis_full'] = reverse_vocabs.get('diagnosis', {}).get(diag_full_idx, '<UNK>')
        
        # Декодируем услуги (если есть в predictions)
        if 'service_letter' in predictions:
            serv_letter_idx = predictions['service_letter'][i].item()
            result['service_letter'] = reverse_vocabs.get('service_letter', {}).get(serv_letter_idx, '<UNK>')
        
        if 'service_hierarchy' in predictions:
            serv_hier_idx = predictions['service_hierarchy'][i].item()
            result['service_hierarchy'] = reverse_vocabs.get('service_hierarchy', {}).get(serv_hier_idx, '<UNK>')
        
        if 'service_full' in predictions:
            serv_full_idx = predictions['service_full'][i].item()
            result['service_full'] = reverse_vocabs.get('service', {}).get(serv_full_idx, '<UNK>')
        
        # Обрабатываем числовые признаки (денормализуем если нужно)
        if 'age' in predictions:
            age_val = predictions['age'][i].item()
            # Если age был нормализован, здесь может потребоваться денормализация
            # result['age'] = age_val * age_std + age_mean
            result['age'] = round(age_val, 2)
        
        if 'sex' in predictions:
            sex_val = predictions['sex'][i].item()
            # Если sex был 0/1, преобразуем в строку
            if sex_val > 0.5:
                result['sex'] = 'Ж'
            else:
                result['sex'] = 'М'
            # Или можно сохранить числом
            # result['sex'] = 1 if sex_val > 0.5 else 0
        
        if 'is_dead' in predictions:
            is_dead_val = predictions['is_dead'][i].item()
            result['is_dead'] = 1 if is_dead_val > 0.5 else 0
        
        # Обрабатываем категориальные признаки
        if 'season' in predictions:
            season_idx = predictions['season'][i].item()
            # Для сезона используем свой обратный справочник
            season_names = {2: 'Зима', 3: 'Весна', 4: 'Лето', 5: 'Осень'}
            result['season'] = season_names.get(season_idx, f'Сезон_{season_idx}')
        
        # Декодируем остальные категориальные признаки
        cat_names = ['group', 'profile', 'result', 'type', 'form']
        for cat_name in cat_names:
            if cat_name in predictions:
                cat_idx = predictions[cat_name][i].item()
                result[cat_name] = reverse_vocabs.get(cat_name, {}).get(cat_idx, f'<UNK_{cat_idx}>')
        
        # Добавляем индексы для отладки
        result['prediction_index'] = i
        
        results.append(result)
    
    return results

