def test_collate_function():
    """
    Комплексный тест collate функции.
    Проверяет размерности, значения, маски, логику паддинга.
    """
    print("=" * 70)
    print("🧪 ТЕСТИРОВАНИЕ COLLATE_FN")
    print("=" * 70)
    
    # 1. Создаем тестовые справочники
    test_vocabs = {
        'diagnosis_letter': {'<PAD>': 0, '<UNK>': 1, 'I': 2, 'E': 3, 'C': 4, 'D': 5, 'A': 6},
        'diagnosis_hierarchy': {'<PAD>': 0, '<UNK>': 1, 'I10': 2, 'I20': 3, 'E11.9': 4, 'C34.1': 5, 'D12.6': 6, 'I25': 7, 'A01': 8},
        'diagnosis': {'<PAD>': 0, '<UNK>': 1, 'I10': 2, 'I20': 3, 'E11.9': 4, 'C34.1': 5, 'D12.6': 6, 'I25': 7, 'A01': 8},
        
        'service_letter': {'<PAD>': 0, '<UNK>': 1, 'A': 2, 'B': 3},
        'service_hierarchy': {'<PAD>': 0, '<UNK>': 1, 'A01': 2, 'A02': 3, 'A03': 4, 'A04': 5, 'B01': 6, 'B02': 7},
        'service': {'<PAD>': 0, '<UNK>': 1, 'A01': 2, 'A02': 3, 'A03': 4, 'A04': 5, 'B01': 6, 'B02': 7},
        
        'group': {'<PAD>': 0, '<UNK>': 1, 'G1': 2, 'G2': 3, 'G3': 4},
        'profile': {'<PAD>': 0, '<UNK>': 1, 'P1': 2, 'P2': 3},
        'result': {'<PAD>': 0, '<UNK>': 1, 'R1': 2, 'R2': 3, 'R3': 4},
        'type': {'<PAD>': 0, '<UNK>': 1, 'T1': 2, 'T2': 3},
        'form': {'<PAD>': 0, '<UNK>': 1, 'F1': 2, 'F2': 3, 'F3': 4},
    }
    
    # 2. Создаем тестовый батч с РАЗНЫМИ длинами и количеством диагнозов
    test_batch = [
        {
            # Пациент 1: 3 случая, разное количество диагнозов
            'window_age': [30.5, 31.0, 31.5],
            'window_sex': ['0', '0', '0'],
            'window_season': [0, 1, 2],
            'window_is_dead': ['0', '0', '0'],
            'window_diagnosis': [
                ['I10', 'I20'],           # 2 диагноза
                ['E11.9'],                # 1 диагноз (минимально)
                ['C34.1', 'D12.6', 'I10'] # 3 диагноза
            ],
            'window_service': ['A01', 'A02', 'A03'],
            'window_group': ['G1', 'G1', 'G2'],
            'window_profile': ['P1', 'P1', 'P1'],
            'window_result': ['R1', 'R1', 'R2'],
            'window_type': ['T1', 'T1', 'T1'],
            'window_form': ['F1', 'F1', 'F2'],
            
            'target_age': 32.0,
            'target_sex': '0',
            'target_season': 3,
            'target_is_dead': '0',
            'target_diagnosis': ['I25', 'I10'],  # 2 диагноза, берем первый
            'target_service': 'A04',
            'target_group': 'G2',
            'target_profile': 'P1',
            'target_result': 'R1',
            'target_type': 'T1',
            'target_form': 'F1',
        },
        {
            # Пациент 2: 2 случая (короче), тоже разное количество диагнозов
            'window_age': [25.0, 26.0],
            'window_sex': ['1', '1'],
            'window_season': [2, 3],
            'window_is_dead': ['0', '0'],
            'window_diagnosis': [
                ['A01', 'I10', 'E11.9'],  # 3 диагноза
                ['C34.1']                 # 1 диагноз
            ],
            'window_service': ['B01', 'B02'],
            'window_group': ['G3', 'G3'],
            'window_profile': ['P2', 'P2'],
            'window_result': ['R3', 'R3'],
            'window_type': ['T2', 'T2'],
            'window_form': ['F3', 'F3'],
            
            'target_age': 27.0,
            'target_sex': '1',
            'target_season': 0,
            'target_is_dead': '0',
            'target_diagnosis': ['D12.6'],
            'target_service': 'B02',
            'target_group': 'G3',
            'target_profile': 'P2',
            'target_result': 'R3',
            'target_type': 'T2',
            'target_form': 'F3',
        },
        {
            # Пациент 3: 1 случай (самый короткий)
            'window_age': [40.0],
            'window_sex': ['0'],
            'window_season': [1],
            'window_is_dead': ['0'],
            'window_diagnosis': [
                ['I10', 'I20', 'E11.9', 'C34.1']  # 4 диагноза (максимум в батче)
            ],
            'window_service': ['A01'],
            'window_group': ['G1'],
            'window_profile': ['P1'],
            'window_result': ['R1'],
            'window_type': ['T1'],
            'window_form': ['F1'],
            
            'target_age': 41.0,
            'target_sex': '0',
            'target_season': 2,
            'target_is_dead': '0',
            'target_diagnosis': ['I25'],
            'target_service': 'A02',
            'target_group': 'G1',
            'target_profile': 'P1',
            'target_result': 'R1',
            'target_type': 'T1',
            'target_form': 'F1',
        }
    ]
    
    print("\n📋 ИСХОДНЫЕ ДАННЫЕ:")
    print("-" * 40)
    for i, example in enumerate(test_batch):
        print(f"\nПациент {i}:")
        print(f"  Длина последовательности: {len(example['window_age'])}")
        print(f"  Диагнозы по случаям: {[len(d) for d in example['window_diagnosis']]}")
        print(f"  Возраст: {example['window_age']}")
    
    # 3. Запускаем collate_fn
    print("\n" + "=" * 70)
    print("🚀 ЗАПУСК COLLATE_FN")
    print("=" * 70)
    
    batch_result = collate_train(test_batch, test_vocabs)
    
    # 4. Проверяем размерности
    print("\n📏 ПРОВЕРКА РАЗМЕРНОСТЕЙ:")
    print("-" * 40)
    
    window = batch_result['window']
    target = batch_result['target']
    
    expected_shapes = {
        # Окно
        'window/age': [3, 3, 1],           # B=3, S=3, 1
        'window/sex': [3, 3, 1],
        'window/season': [3, 3],
        'window/is_dead': [3, 3, 1],
        'window/diagnosis_letter': [3, 3, 4],  # D=4 (максимум диагнозов в батче)
        'window/diagnosis_hierarchy': [3, 3, 4],
        'window/diagnosis_full': [3, 3, 4],
        'window/diagnosis_mask': [3, 3, 4],
        'window/service_letter': [3, 3],
        'window/lengths': [3],
        
        # Цель
        'target/age': [3, 1],
        'target/sex': [3, 1],
        'target/season': [3],
        'target/diagnosis_letter': [3],
        'target/service_letter': [3],
    }
    
    for path, expected_shape in expected_shapes.items():
        if '/' in path:
            dict_name, key = path.split('/')
            tensor = batch_result[dict_name][key]
        else:
            tensor = batch_result[path]
        
        actual_shape = list(tensor.shape)
        status = "✅" if actual_shape == expected_shape else "❌"
        print(f"{status} {path:30} ожидалось: {expected_shape}, получено: {actual_shape}")
    
    # 5. Проверяем конкретные значения
    print("\n🔍 ПРОВЕРКА ЗНАЧЕНИЙ И МАСОК:")
    print("-" * 40)
    
    # Проверяем длины последовательностей
    print("\nДлины последовательностей (window/lengths):")
    print(f"  Ожидалось: [3, 2, 1] (пациенты отсортированы по убыванию)")
    print(f"  Получено:  {window['lengths'].tolist()}")
    
    # Проверяем маски диагнозов
    print("\nМаски диагнозов (window/diagnosis_mask):")
    print("Пациент 0 (3 случая, диагнозов: 2, 1, 3):")
    for i in range(3):
        mask = window['diagnosis_mask'][0, i].tolist()
        print(f"  Случай {i}: {mask} (реальных диагнозов: {sum(mask)})")
    
    print("\nПациент 1 (2 случая, диагнозов: 3, 1):")
    for i in range(2):
        mask = window['diagnosis_mask'][1, i].tolist()
        print(f"  Случай {i}: {mask} (реальных диагнозов: {sum(mask)})")
    
    print("\nПациент 2 (1 случай, диагнозов: 4):")
    mask = window['diagnosis_mask'][2, 0].tolist()
    print(f"  Случай 0: {mask} (реальных диагнозов: {sum(mask)})")
    
    # Проверяем паддинг диагнозов
    print("\n🔬 ДЕТАЛЬНАЯ ПРОВЕРКА ДИАГНОЗОВ:")
    print("-" * 40)
    
    # Смотрим первый случай первого пациента (должно быть 2 реальных диагноза + 2 PAD)
    print("\nПациент 0, Случай 0 (должно быть: I10, I20, PAD, PAD):")
    diag_letter = window['diagnosis_letter'][0, 0].tolist()
    diag_mask = window['diagnosis_mask'][0, 0].tolist()
    
    print(f"  Индексы букв: {diag_letter}")
    print(f"  Маска:        {diag_mask}")
    
    # Декодируем обратно
    reverse_letter = {v: k for k, v in test_vocabs['diagnosis_letter'].items()}
    decoded = [reverse_letter[idx] for idx in diag_letter]
    print(f"  Декодировано: {decoded}")
    
    # Проверяем что PAD соответствуют маске 0
    for i, (idx, mask_val) in enumerate(zip(diag_letter, diag_mask)):
        if mask_val == 0:
            assert idx == 0, f"PAD должен быть 0, но получил {idx} в позиции {i}"
        else:
            assert idx != 0, f"Реальный диагноз не должен быть 0 в позиции {i}"
    print("  ✅ PAD корректны (индекс 0 где маска 0)")
    
    # 6. Проверяем цели
    print("\n🎯 ПРОВЕРКА ЦЕЛЕЙ:")
    print("-" * 40)
    
    print("Целевые диагнозы (должны быть первые диагнозы из target_diagnosis):")
    expected_target_diag = ['I25', 'D12.6', 'I25']  # Первые диагнозы из каждого примера
    for i in range(3):
        diag_idx = target['diagnosis_full'][i].item()
        diag_name = test_vocabs['diagnosis'].get(diag_idx, '<UNK>')
        expected = expected_target_diag[i]
        status = "✅" if diag_name == expected else "❌"
        print(f"  Пациент {i}: индекс {diag_idx} → '{diag_name}' (ожидалось '{expected}') {status}")
    
    # 7. Проверяем числовые признаки
    print("\n📊 ПРОВЕРКА ЧИСЛОВЫХ ПРИЗНАКОВ:")
    print("-" * 40)
    
    print("Возраст в окне (должен сохранить значения):")
    for i in range(3):
        ages = window['age'][i, :, 0].tolist()
        # Берем только реальные значения (не паддинг)
        real_ages = ages[:batch_result['window']['lengths'][i]]
        print(f"  Пациент {i}: {real_ages}")
    
    print("\nЦелевой возраст:")
    for i in range(3):
        age = target['age'][i, 0].item()
        expected = test_batch[i]['target_age']
        diff = abs(age - expected)
        status = "✅" if diff < 0.001 else "❌"
        print(f"  Пациент {i}: {age:.1f} (ожидалось {expected:.1f}) {status}")
    
    # 8. Тест с инференсом
    print("\n" + "=" * 70)
    print("🧪 ТЕСТ COLLATE_INFERENCE")
    print("=" * 70)
    
    inference_batch = collate_inference(test_batch, test_vocabs)
    
    print("\nПроверяем что в инференсе нет target:")
    has_target = 'target' in inference_batch
    print(f"  Есть 'target'? {has_target} (должно быть False)")
    print(f"  Есть 'window'? {'window' in inference_batch}")
    print(f"  Ключи: {list(inference_batch.keys())}")
    
    # 9. Тест raw_to_result
    print("\n" + "=" * 70)
    print("🧪 ТЕСТ RAW_TO_RESULT")
    print("=" * 70)
    
    # Создаем тестовые предсказания
    test_predictions = {
        'diagnosis_letter': torch.tensor([2, 5, 2]),      # I, D, I
        'diagnosis_hierarchy': torch.tensor([7, 6, 7]),   # I25, D12.6, I25
        'diagnosis_full': torch.tensor([7, 6, 7]),        # I25, D12.6, I25
        'service_letter': torch.tensor([2, 3, 2]),        # A, B, A
        'service_hierarchy': torch.tensor([5, 7, 3]),     # A04, B02, A02
        'service_full': torch.tensor([5, 7, 3]),          # A04, B02, A02
        'age': torch.tensor([[32.0], [27.0], [41.0]]),
        'sex': torch.tensor([[0.0], [1.0], [0.0]]),
        'season': torch.tensor([3, 0, 2]),
        'group': torch.tensor([3, 4, 2]),    # G2, G3, G1
        'profile': torch.tensor([2, 3, 2]),  # P1, P2, P1
    }
    
    decoded_results = raw_to_result(test_predictions, test_vocabs)
    
    print("\nДекодированные предсказания:")
    for i, res in enumerate(decoded_results):
        print(f"\nПациент {i}:")
        print(f"  Диагноз: {res['diagnosis_full']} (буква: {res['diagnosis_letter']})")
        print(f"  Услуга: {res['service_full']} (буква: {res['service_letter']})")
        print(f"  Возраст: {res['age']:.1f}, Пол: {'М' if res['sex'] == 'M' else 'Ж' if res['sex'] == 'F' else '?'}")
    
    print("\n" + "=" * 70)
    print("🎉 ТЕСТИРОВАНИЕ ЗАВЕРШЕНО")
    print("=" * 70)
    
    # Возвращаем результат для ручной проверки
    return batch_result, inference_batch, decoded_results


# Запуск теста
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("🚀 ЗАПУСК ПОЛНОГО ТЕСТА COLLATE ФУНКЦИЙ")
    print("=" * 70)
    
    try:
        batch_result, inference_batch, decoded_results = test_collate_function()
        
        print("\n📋 ФИНАЛЬНАЯ СВОДКА:")
        print("-" * 40)
        print(f"✅ collate_train: успешно создан батч")
        print(f"   - batch_size: {batch_result['batch_size']}")
        print(f"   - max_seq_len: {batch_result['max_seq_len']}")
        print(f"   - max_diags: {batch_result['max_diags']}")
        
        print(f"\n✅ collate_inference: успешно создан батч без target")
        print(f"   - имеет window: {'window' in inference_batch}")
        print(f"   - не имеет target: {'target' not in inference_batch}")
        
        print(f"\n✅ raw_to_result: успешно декодировано {len(decoded_results)} предсказаний")
        
        # Дополнительно: можно сохранить тензоры для визуальной проверки
        print("\n💾 Для дополнительной проверки:")
        print("batch_result сохранен в переменной 'batch_result'")
        print("inference_batch сохранен в переменной 'inference_batch'")
        print("decoded_results сохранен в переменной 'decoded_results'")
        
        # Интерактивная проверка
        print("\n🔍 Для ручной проверки в интерактивном режиме:")
        print("   >>> batch_result['window']['diagnosis_mask'][0]  # посмотреть маски пациента 0")
        print("   >>> batch_result['target']['diagnosis_full']     # посмотреть целевые диагнозы")
        
    except Exception as e:
        print(f"\n❌ ОШИБКА В ТЕСТЕ: {e}")
        import traceback
        traceback.print_exc()