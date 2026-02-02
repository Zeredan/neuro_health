# app/train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import json
import os
from datetime import datetime

# Ваши модули
from dataset import MedicalDataset
from collate import collate_fn_train
from model import MedicalLSTM
from vocab import load_or_build_vocabs, save_vocabs

# ========== КОНСТАНТЫ ==========
DATA_PATH = "data/medical_cases.csv"
CHECKPOINT_DIR = "checkpoints"
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 100
PATIENCE = 10  # для early stopping
TRAIN_VAL_SPLIT = 0.8

# Признаки модели (настройте под свои данные)
MODEL_CONFIG = {
    'diagnosis_dim': 10,  # 10 диагнозов
    'diagnosis_embed_dims': [64, 32, 16],  # эмбеддинги для иерархии
    'service_vocab_size': 100,  # нужно получить из словаря
    'service_embed_dim': 32,
    'age_proj_dim': 16,
    'lstm_hidden': 128,
    'num_lstm_layers': 2,
    'dropout': 0.3
}
# ===============================

def save_checkpoint(state, filename, checkpoint_dir=CHECKPOINT_DIR):
    """Сохраняет чекпоинт"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)
    print(f"Checkpoint saved to {filepath}")

def load_checkpoint(filename, checkpoint_dir=CHECKPOINT_DIR):
    """Загружает чекпоинт"""
    filepath = os.path.join(checkpoint_dir, filename)
    if os.path.exists(filepath):
        checkpoint = torch.load(filepath, map_location='cpu')
        print(f"Checkpoint loaded from {filepath}")
        return checkpoint
    else:
        print(f"Checkpoint not found: {filepath}")
        return None

def save_history(history, filename="training_history.json"):
    """Сохраняет историю обучения в JSON"""
    with open(filename, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"History saved to {filename}")

def train_epoch(model, dataloader, optimizer, device):
    """Одна эпоха обучения"""
    model.train()
    total_loss = 0
    total_samples = 0
    
    # Loss функции для разных задач
    criterion_diagnosis = nn.CrossEntropyLoss()
    criterion_age = nn.MSELoss()
    
    for batch_idx, (x_tensors, y_tensors) in enumerate(dataloader):
        # Перемещаем на устройство
        x_tensors = {k: v.to(device) for k, v in x_tensors.items()}
        y_tensors = {k: v.to(device) for k, v in y_tensors.items()}
        
        # Forward
        optimizer.zero_grad()
        predictions = model(**x_tensors)
        
        # Вычисляем общий loss
        loss = 0
        
        # 1. Loss для диагнозов (10 диагнозов, каждый по 3 иерархических уровня)
        for i in range(10):  # для каждого из 10 диагнозов
            # predictions['diagnosis'][i] имеет shape [B, 3, vocab_size_i]
            # y_tensors['diagnosis'][:, i, :] имеет shape [B, 3]
            
            for level in range(3):  # 3 уровня иерархии
                loss += criterion_diagnosis(
                    predictions['diagnosis'][i][:, level, :],
                    y_tensors['diagnosis'][:, i, level]
                )
        
        # 2. Loss для возраста (регрессия)
        loss += criterion_age(predictions['age'], y_tensors['age'])
        
        # 3. Loss для услуг (если есть)
        if 'service' in predictions:
            loss += criterion_diagnosis(
                predictions['service'],
                y_tensors['service']
            )
        
        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Статистика
        batch_size = x_tensors['age'].shape[0]
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        
        if batch_idx % 20 == 0:
            print(f"  Batch {batch_idx:3d}, Loss: {loss.item():.4f}")
    
    return total_loss / total_samples if total_samples > 0 else 0

def validate(model, dataloader, device):
    """Валидация"""
    model.eval()
    total_loss = 0
    total_samples = 0
    
    criterion_diagnosis = nn.CrossEntropyLoss()
    criterion_age = nn.MSELoss()
    
    with torch.no_grad():
        for x_tensors, y_tensors in dataloader:
            x_tensors = {k: v.to(device) for k, v in x_tensors.items()}
            y_tensors = {k: v.to(device) for k, v in y_tensors.items()}
            
            predictions = model(**x_tensors)
            
            # Loss для валидации (только диагнозы для простоты)
            loss = 0
            for i in range(10):
                for level in range(3):
                    loss += criterion_diagnosis(
                        predictions['diagnosis'][i][:, level, :],
                        y_tensors['diagnosis'][:, i, level]
                    )
            
            batch_size = x_tensors['age'].shape[0]
            total_loss += loss.item() * batch_size
            total_samples += batch_size
    
    return total_loss / total_samples if total_samples > 0 else 0

def load_medical_data(csv_path):
    """Загружает данные из CSV"""
    import pandas as pd
    
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Преобразуем DataFrame в список пациентов
    patients = []
    
    # Предположим, что данные сгруппированы по patient_id
    for patient_id, group in df.groupby('patient_id'):
        cases = []
        # Сортируем по дате
        group = group.sort_values('date')
        
        for _, row in group.iterrows():
            # Создаем MedicalCase для каждой строки
            # Здесь нужно адаптировать под вашу структуру данных
            case = {
                'age': float(row['age']),
                'diagnosis_codes': [
                    row[f'diagnosis_{i}'] for i in range(10)  # 10 диагнозов
                ],
                'service_code': row['service_code'],
                # ... другие признаки
            }
            cases.append(case)
        
        if len(cases) >= 2:  # минимум 2 случая для создания окна
            patients.append(cases)
    
    print(f"Loaded {len(patients)} patients")
    return patients

def main():
    print("=" * 60)
    print("MEDICAL LSTM TRAINING")
    print("=" * 60)
    
    # 1. Выбор: загрузить или начать с нуля
    choice = input("Load from checkpoint? (y/n): ").strip().lower()
    
    if choice == 'y':
        checkpoint_name = input("Checkpoint filename [best_model.pth]: ").strip()
        if not checkpoint_name:
            checkpoint_name = "best_model.pth"
        
        checkpoint = load_checkpoint(checkpoint_name)
        
        if checkpoint:
            # Восстанавливаем состояние
            start_epoch = checkpoint['epoch'] + 1
            train_history = checkpoint.get('train_history', [])
            val_history = checkpoint.get('val_history', [])
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            vocabs = checkpoint['vocabs']
            model_config = checkpoint.get('model_config', MODEL_CONFIG)
            
            print(f"Resuming from epoch {start_epoch}")
            print(f"Previous best val loss: {best_val_loss:.4f}")
        else:
            print("Starting from scratch...")
            start_epoch = 0
            train_history = []
            val_history = []
            best_val_loss = float('inf')
            vocabs = None
            model_config = MODEL_CONFIG
    else:
        print("Starting from scratch...")
        start_epoch = 0
        train_history = []
        val_history = []
        best_val_loss = float('inf')
        vocabs = None
        model_config = MODEL_CONFIG
    
    # 2. Устройство
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # 3. Загрузка данных и создание словарей
    print("\n" + "=" * 40)
    print("PREPARING DATA")
    print("=" * 40)
    
    patients = load_medical_data(DATA_PATH)
    
    if not vocabs:
        print("Building vocabs from data...")
        # Здесь нужно построить словари из данных
        # Это зависит от вашей структуры данных
        vocabs = {
            'diagnosis': {},  # словарь для диагнозов
            'service': {},    # словарь для услуг
            # ... другие словари
        }
        save_vocabs(vocabs, "vocabs.json")
    else:
        print("Using loaded vocabs")
    
    # Обновляем размеры словарей в конфиге
    if vocabs:
        model_config['service_vocab_size'] = len(vocabs.get('service', {}))
        # Добавляем размеры словарей для диагнозов
        model_config['diagnosis_vocab_sizes'] = [
            len(vocabs.get('diagnosis_letter', {})),
            len(vocabs.get('diagnosis_group', {})),
            len(vocabs.get('diagnosis_code', {}))
        ]
    
    # 4. Создаем Dataset и DataLoader
    print("\nCreating datasets...")
    
    # Разделяем пациентов на train/val
    train_size = int(len(patients) * TRAIN_VAL_SPLIT)
    val_size = len(patients) - train_size
    
    train_patients, val_patients = random_split(
        patients, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_dataset = MedicalDataset(train_patients)
    val_dataset = MedicalDataset(val_patients)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda batch: collate_fn_train(batch, vocabs),
        num_workers=2 if device.type == 'cpu' else 0,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=lambda batch: collate_fn_train(batch, vocabs),
        num_workers=2 if device.type == 'cpu' else 0,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # 5. Создаем модель
    print("\n" + "=" * 40)
    print("CREATING MODEL")
    print("=" * 40)
    
    model = MedicalLSTM(config=model_config).to(device)
    print(f"Model created with config: {model_config}")
    
    # 6. Оптимизатор
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Если загружаем чекпоинт, восстанавливаем optimizer
    if choice == 'y' and checkpoint:
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        print("Model and optimizer states restored")
    
    # 7. Цикл обучения
    print("\n" + "=" * 40)
    print("STARTING TRAINING")
    print("=" * 40)
    
    epochs_without_improvement = 0
    
    for epoch in range(start_epoch, EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        print("-" * 30)
        
        # Обучение
        train_loss = train_epoch(model, train_loader, optimizer, device)
        train_history.append(train_loss)
        
        # Валидация
        val_loss = validate(model, val_loader, device)
        val_history.append(val_loss)
        
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Сохраняем историю
        save_history({
            'train_loss': train_history,
            'val_loss': val_history,
            'epochs': list(range(len(train_history))),
            'config': model_config,
            'timestamp': datetime.now().isoformat()
        })
        
        # Early stopping и сохранение
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            
            # Сохраняем лучшую модель
            save_checkpoint({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'train_history': train_history,
                'val_history': val_history,
                'best_val_loss': best_val_loss,
                'vocabs': vocabs,
                'model_config': model_config
            }, filename="best_model.pth")
            
            print(f"✓ New best model! Val loss: {val_loss:.4f}")
        else:
            epochs_without_improvement += 1
            print(f"✗ No improvement ({epochs_without_improvement}/{PATIENCE})")
        
        # Сохраняем последний чекпоинт
        save_checkpoint({
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'train_history': train_history,
            'val_history': val_history,
            'best_val_loss': best_val_loss,
            'vocabs': vocabs,
            'model_config': model_config
        }, filename="last_checkpoint.pth")
        
        # Проверка early stopping
        if epochs_without_improvement >= PATIENCE:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs")
            break
    
    # 8. Сохраняем финальную модель для инференса
    print("\n" + "=" * 40)
    print("SAVING FINAL MODEL")
    print("=" * 40)
    
    final_state = {
        'model_state': model.state_dict(),
        'vocabs': vocabs,
        'model_config': model_config,
        'train_history': train_history,
        'val_history': val_history,
        'best_val_loss': best_val_loss
    }
    
    save_checkpoint(final_state, filename="final_model.pth")
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETED!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Total epochs trained: {len(train_history)}")
    print("=" * 60)

if __name__ == "__main__":
    main()