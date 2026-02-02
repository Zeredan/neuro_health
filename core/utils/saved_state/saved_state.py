import torch
import pickle
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import json
import logging

logger = logging.getLogger(__name__)


def save_training_state(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    history: Dict[str, list],
    best_val_loss: float,
    model_name: str = "model"
):
    """
    Сохраняет полное состояние обучения.
    
    Args:
        model: Модель PyTorch
        optimizer: Оптимизатор
        epoch: Текущая эпоха
        history: История обучения {'train_loss': [], 'val_loss': [], ...}
        best_val_loss: Лучшая ошибка на валидации
        model_name: Имя модели (для имени файла)
    """
    try:
        # Определяем пути
        current_dir = Path(__file__).parent
        project_root = current_dir.parent.parent.parent  # core/utils/saved_state -> project/
        
        model_dir = project_root / "res" / "model"
        train_state_dir = project_root / "res" / "train_state"
        
        # Создаем директории если их нет
        model_dir.mkdir(parents=True, exist_ok=True)
        train_state_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Сохраняем модель
        model_path = model_dir / f"{model_name}.pth"
        torch.save(model.state_dict(), model_path)
        logger.info(f"Модель сохранена: {model_path}")
        
        # 2. Сохраняем состояние обучения
        checkpoint = {
            'epoch': epoch,
            'optimizer_state_dict': optimizer.state_dict(),
            'history': history,
            'best_val_loss': best_val_loss,
            'model_class': model.__class__.__name__,  # Для информации
        }
        
        # Сохраняем двумя способами для надежности
        # a) Pickle (бинарный)
        checkpoint_path = train_state_dir / f"{model_name}_checkpoint.pkl"
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint, f)
        
        # b) JSON (читаемый, но только для истории и метаданных)
        json_checkpoint = {
            'epoch': epoch,
            'best_val_loss': best_val_loss,
            'model_class': model.__class__.__name__,
            'history_lengths': {k: len(v) for k, v in history.items()},
        }
        
        json_path = train_state_dir / f"{model_name}_checkpoint.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_checkpoint, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Состояние обучения сохранено:")
        logger.info(f"  - Эпоха: {epoch}")
        logger.info(f"  - Лучшая val_loss: {best_val_loss:.6f}")
        logger.info(f"  - История: {json_checkpoint['history_lengths']}")
        
        return True
        
    except Exception as e:
        logger.error(f"Ошибка при сохранении состояния: {e}")
        return False


def load_training_state(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    model_name: str = "model"
) -> Tuple[bool, Optional[int], Optional[Dict[str, list]], Optional[float]]:
    """
    Загружает полное состояние обучения.
    
    Args:
        model: Модель PyTorch (должна быть той же архитектуры)
        optimizer: Оптимизатор (того же типа и с теми же параметрами)
        model_name: Имя модели (для поиска файлов)
    
    Returns:
        Tuple[success, epoch, history, best_val_loss]
        success: Успешно ли загружено
        epoch: Текущая эпоха или None
        history: История обучения или None
        best_val_loss: Лучшая ошибка или None
    """
    try:
        # Определяем пути
        current_dir = Path(__file__).parent
        project_root = current_dir.parent.parent.parent
        
        model_dir = project_root / "res" / "model"
        train_state_dir = project_root / "res" / "train_state"
        
        model_path = model_dir / f"{model_name}.pth"
        checkpoint_path = train_state_dir / f"{model_name}_checkpoint.pkl"
        
        # Проверяем существование файлов
        if not model_path.exists():
            logger.warning(f"Файл модели не найден: {model_path}")
            return False, None, None, None
        
        if not checkpoint_path.exists():
            logger.warning(f"Файл состояния не найден: {checkpoint_path}")
            return False, None, None, None
        
        # 1. Загружаем модель
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        logger.info(f"Модель загружена: {model_path}")
        
        # 2. Загружаем состояние обучения
        with open(checkpoint_path, 'rb') as f:
            checkpoint = pickle.load(f)
        
        # Восстанавливаем состояние
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        history = checkpoint['history']
        best_val_loss = checkpoint['best_val_loss']
        
        logger.info(f"Состояние обучения загружено:")
        logger.info(f"  - Эпоха: {epoch}")
        logger.info(f"  - Лучшая val_loss: {best_val_loss:.6f}")
        logger.info(f"  - История: { {k: len(v) for k, v in history.items()} }")
        
        return True, epoch, history, best_val_loss
        
    except Exception as e:
        logger.error(f"Ошибка при загрузке состояния: {e}")
        return False, None, None, None


def get_available_checkpoints() -> Dict[str, Dict[str, Any]]:
    """
    Возвращает информацию о доступных чекпоинтах.
    
    Returns:
        Словарь с информацией о чекпоинтах
        {
            'model_name': {
                'epoch': int,
                'best_val_loss': float,
                'history_lengths': Dict[str, int],
                'model_path': Path,
                'checkpoint_path': Path
            },
            ...
        }
    """
    try:
        current_dir = Path(__file__).parent
        project_root = current_dir.parent.parent.parent
        
        train_state_dir = project_root / "res" / "train_state"
        model_dir = project_root / "res" / "model"
        
        if not train_state_dir.exists():
            return {}
        
        checkpoints_info = {}
        
        # Ищем все JSON файлы с чекпоинтами
        for json_file in train_state_dir.glob("*_checkpoint.json"):
            model_name = json_file.name.replace("_checkpoint.json", "")
            
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    info = json.load(f)
                
                # Проверяем существование соответствующих файлов
                model_path = model_dir / f"{model_name}.pth"
                checkpoint_path = train_state_dir / f"{model_name}_checkpoint.pkl"
                
                if model_path.exists() and checkpoint_path.exists():
                    checkpoints_info[model_name] = {
                        'epoch': info['epoch'],
                        'best_val_loss': info['best_val_loss'],
                        'history_lengths': info['history_lengths'],
                        'model_class': info.get('model_class', 'Unknown'),
                        'model_path': model_path,
                        'checkpoint_path': checkpoint_path,
                        'json_path': json_file
                    }
                    
            except Exception as e:
                logger.warning(f"Не удалось прочитать {json_file}: {e}")
        
        logger.info(f"Найдено чекпоинтов: {len(checkpoints_info)}")
        for name, info in checkpoints_info.items():
            logger.info(f"  {name}: эпоха {info['epoch']}, val_loss {info['best_val_loss']:.6f}")
        
        return checkpoints_info
        
    except Exception as e:
        logger.error(f"Ошибка при поиске чекпоинтов: {e}")
        return {}


def clear_training_state(model_name: str = "model") -> bool:
    """
    Удаляет сохраненное состояние обучения для указанной модели.
    
    Args:
        model_name: Имя модели
    
    Returns:
        True если успешно удалено
    """
    try:
        current_dir = Path(__file__).parent
        project_root = current_dir.parent.parent.parent
        
        model_dir = project_root / "res" / "model"
        train_state_dir = project_root / "res" / "train_state"
        
        # Удаляем все файлы связанные с моделью
        files_to_remove = [
            model_dir / f"{model_name}.pth",
            train_state_dir / f"{model_name}_checkpoint.pkl",
            train_state_dir / f"{model_name}_checkpoint.json",
        ]
        
        removed = []
        for file_path in files_to_remove:
            if file_path.exists():
                file_path.unlink()
                removed.append(file_path.name)
        
        if removed:
            logger.info(f"Удалены файлы: {removed}")
            return True
        else:
            logger.warning(f"Файлы для модели '{model_name}' не найдены")
            return False
            
    except Exception as e:
        logger.error(f"Ошибка при удалении состояния: {e}")
        return False


# Пример использования
if __name__ == "__main__":
    # Настройка логирования
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("=" * 60)
    print("ТЕСТ ФУНКЦИЙ СОХРАНЕНИЯ/ЗАГРУЗКИ")
    print("=" * 60)
    
    # Создаем тестовую модель и оптимизатор
    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = torch.nn.Linear(10, 1)
        
        def forward(self, x):
            return self.layer(x)
    
    # Тест 1: Проверяем доступные чекпоинты
    print("\n1. ПОИСК ДОСТУПНЫХ ЧЕКПОИНТОВ:")
    checkpoints = get_available_checkpoints()
    
    if checkpoints:
        for name, info in checkpoints.items():
            print(f"  ✓ {name}:")
            print(f"     Эпоха: {info['epoch']}")
            print(f"     Val loss: {info['best_val_loss']:.6f}")
            print(f"     История: {info['history_lengths']}")
    else:
        print("  ℹ Чекпоинты не найдены")
    
    # Тест 2: Сохранение тестового состояния
    print("\n2. ТЕСТ СОХРАНЕНИЯ:")
    
    test_model = SimpleModel()
    test_optimizer = torch.optim.Adam(test_model.parameters(), lr=0.001)
    
    test_history = {
        'train_loss': [0.5, 0.4, 0.3],
        'val_loss': [0.6, 0.5, 0.4],
        'train_acc': [0.7, 0.75, 0.8],
        'val_acc': [0.65, 0.72, 0.78]
    }
    
    success = save_training_state(
        model=test_model,
        optimizer=test_optimizer,
        epoch=10,
        history=test_history,
        best_val_loss=0.35,
        model_name="test_model"
    )
    
    if success:
        print("  ✓ Тестовое состояние сохранено")
        
        # Тест 3: Загрузка состояния
        print("\n3. ТЕСТ ЗАГРУЗКИ:")
        
        new_model = SimpleModel()
        new_optimizer = torch.optim.Adam(new_model.parameters(), lr=0.001)
        
        load_success, epoch, history, best_loss = load_training_state(
            model=new_model,
            optimizer=new_optimizer,
            model_name="test_model"
        )
        
        if load_success:
            print(f"  ✓ Состояние загружено:")
            print(f"     Эпоха: {epoch}")
            print(f"     Лучший val loss: {best_loss}")
            print(f"     Длина истории: { {k: len(v) for k, v in history.items()} }")
        
        # Тест 4: Очистка
        #print("\n4. ТЕСТ ОЧИСТКИ:")
        #clear_success = clear_training_state("test_model")
        #if clear_success:
        #    print("  ✓ Тестовые файлы удалены")
    
    print("\n" + "=" * 60)
    print("ТЕСТ ЗАВЕРШЕН")
    print("=" * 60)