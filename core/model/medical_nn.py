import torch
import torch.nn as nn
import torch.nn.functional as F

class MedicalLSTM(nn.Module):
    def __init__(self, config):
        super(MedicalLSTM, self).__init__()
        
        self.config = config
        
        # 1. ВСЕ ПРОСТЫЕ КАТЕГОРИАЛЬНЫЕ ПРИЗНАКИ (6 штук)
        self.simple_features = ['group', 'profile', 'result', 'type', 'form', 'season']
        self.embeddings_simple = nn.ModuleDict()
        
        for feat in self.simple_features:
            vocab_size = config['vocab_sizes'][feat]
            embed_dim = config['embedding_dims'][feat]
            self.embeddings_simple[feat] = nn.Embedding(
                vocab_size, embed_dim, padding_idx=0
            )
        
        # 2. ЭМБЕДДИНГИ ДЛЯ ДИАГНОЗОВ (3 уровня)
        self.diag_levels = ['diagnosis_letter', 'diagnosis_hierarchy', 'diagnosis_full']
        self.embeddings_diagnosis = nn.ModuleDict()
        
        for level in self.diag_levels:
            vocab_size = config['vocab_sizes'][level]
            embed_dim = config['embedding_dims'][level]
            self.embeddings_diagnosis[level] = nn.Embedding(
                vocab_size, embed_dim, padding_idx=0
            )
        
        # Размерность эмбеддингов диагнозов (сумма всех уровней)
        self.diag_embed_total = sum(
            config['embedding_dims'][level] for level in self.diag_levels
        )
        
        # 3. ЭМБЕДДИНГИ ДЛЯ УСЛУГ (3 уровня)
        self.service_levels = ['service_letter', 'service_hierarchy', 'service_full']
        self.embeddings_service = nn.ModuleDict()
        
        for level in self.service_levels:
            vocab_size = config['vocab_sizes'][level]
            embed_dim = config['embedding_dims'][level]
            self.embeddings_service[level] = nn.Embedding(
                vocab_size, embed_dim, padding_idx=0
            )
        
        # 4. ATTENTION ДЛЯ ДИАГНОЗОВ
        # Вход: [B, S, max_diags, diag_embed_total + 3]
        # Выход: [B, S, max_diags, 1]
        num_features = 3  # age, sex, is_dead
        self.diag_attention = nn.Sequential(
            nn.Linear(self.diag_embed_total + num_features, 64),
            nn.ReLU(),
            nn.Dropout(config.get('dropout', 0.1)),
            nn.Linear(64, 1)
        )
        
        # 5. MLP ПЕРЕД LSTM
        # Вычисляем общую размерность входных признаков
        
        # 5.1 Все простые категории (включая сезон)
        simple_embed_total = sum(
            config['embedding_dims'][feat] for feat in self.simple_features
        )
        
        # 5.2 Размерность услуг
        service_embed_total = sum(
            config['embedding_dims'][level] for level in self.service_levels
        )
        
        # 5.3 ИТОГОВАЯ РАЗМЕРНОСТЬ ВХОДА
        total_input_dim = (
            simple_embed_total +      # все простые категории (6 признаков)
            self.diag_embed_total +   # диагнозы после attention (сумма эмбеддингов)
            service_embed_total +     # услуги
            num_features              # числовые признаки
        )
        
        # 5.4 MLP слои
        mlp_hidden = config['mlp_hidden']
        self.mlp = nn.Sequential(
            nn.Linear(total_input_dim, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(config.get('dropout', 0.1)),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(config.get('dropout', 0.1))
        )
        
        # 6. LSTM
        lstm_hidden = config['lstm_hidden']
        self.lstm = nn.LSTM(
            input_size=mlp_hidden,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )
        
        # 7. ВЫХОДНЫЕ ГОЛОВЫ
        
        # 7.1 Регрессия
        self.head_age = nn.Linear(lstm_hidden, 1)
        self.head_death = nn.Linear(lstm_hidden, 1)
        
        # 7.2 Диагнозы (3 уровня)
        self.head_diagnosis = nn.ModuleDict()
        for level in self.diag_levels:
            vocab_size = config['vocab_sizes'][level]
            self.head_diagnosis[level] = nn.Linear(lstm_hidden, vocab_size)
        
        # 7.3 Услуги (3 уровня)
        self.head_service = nn.ModuleDict()
        for level in self.service_levels:
            vocab_size = config['vocab_sizes'][level]
            self.head_service[level] = nn.Linear(lstm_hidden, vocab_size)
        
        # 7.4 Все простые категории (6 признаков)
        self.head_simple = nn.ModuleDict()
        for feat in self.simple_features:
            vocab_size = config['vocab_sizes'][feat]
            self.head_simple[feat] = nn.Linear(lstm_hidden, vocab_size)
        
        # 8. ВСПОМОГАТЕЛЬНЫЕ СЛОИ
        self.dropout = nn.Dropout(config.get('dropout', 0.1))
        
        # 9. НОРМАЛИЗАЦИЯ (опционально)
        self.layer_norm = nn.LayerNorm(total_input_dim) if config.get('use_layer_norm', False) else None
    
    def _process_diagnoses(self, diag_tensors, num_tensors, mask):
        """
        Обработка диагнозов с attention механизмом.
        
        Args:
            diag_tensors: dict с 3 тензорами иерархий [B, S, max_diags]
            num_tensors: [B, S, 3] (age, sex, is_dead)
            mask: [B, S, max_diags] (1 - реальный диагноз, 0 - pad)
        
        Returns:
            processed_diag: [B, S, diag_embed_total]
            attention_weights: [B, S, max_diags]
        """
        B, S, max_diags = mask.shape
        
        # 1. Применяем эмбеддинги к каждому уровню диагнозов
        diag_embeds = []
        for level in self.diag_levels:
            # Вход: [B, S, max_diags]
            # Эмбеддинг работает с последним измерением
            embedded = self.embeddings_diagnosis[level](diag_tensors[level])
            # embedded: [B, S, max_diags, embed_dim]
            diag_embeds.append(embedded)
        
        # 2. Конкатенируем по последней оси
        diag_concat = torch.cat(diag_embeds, dim=-1)  # [B, S, max_diags, diag_embed_total]
        
        # 3. Добавляем числовые признаки к каждому диагнозу
        num_expanded = num_tensors.unsqueeze(2).expand(-1, -1, max_diags, -1)  # [B, S, max_diags, 3]
        diag_with_features = torch.cat([diag_concat, num_expanded], dim=-1)  # [B, S, max_diags, diag_embed_total + 3]
        
        # 4. Вычисляем веса attention
        # Linear автоматически работает по последней оси
        attention_logits = self.diag_attention(diag_with_features)  # [B, S, max_diags, 1]
        attention_logits = attention_logits.squeeze(-1)  # [B, S, max_diags]
        
        # 5. Применяем маску
        mask_bool = (mask == 0)  # True для pad диагнозов
        attention_logits = attention_logits.masked_fill(mask_bool, float('-inf'))
        
        # 6. Softmax по оси диагнозов
        attention_weights = F.softmax(attention_logits, dim=-1)  # [B, S, max_diags]
        
        # 7. Взвешенная сумма эмбеддингов диагнозов
        # attention_weights: [B, S, max_diags, 1]
        # diag_concat: [B, S, max_diags, diag_embed_total]
        weights_expanded = attention_weights.unsqueeze(-1)
        weighted_diag = (diag_concat * weights_expanded).sum(dim=2)  # [B, S, diag_embed_total]
        
        # НЕТ ПРОЕКЦИИ - оставляем как есть, сумма эмбеддингов
        
        return weighted_diag, attention_weights
    
    def forward(self, batch, return_attention=False):
        """
        Прямой проход модели.
        
        Args:
            batch: dict с тензорами window_data
            return_attention: если True, возвращает attention weights
            
        Returns:
            predictions: dict с предсказаниями
            attention_weights: (опционально) [B, S, max_diags]
        """
        B, S = batch['age'].shape[:2]
        
        # 1. ЧИСЛОВЫЕ ПРИЗНАКИ
        numeric_features = torch.stack([
            batch['age'],
            batch['sex'],
            batch['is_dead']
        ], dim=-1)  # [B, S, 3]
        
        # 2. ВСЕ ПРОСТЫЕ КАТЕГОРИАЛЬНЫЕ ПРИЗНАКИ (включая сезон)
        simple_embeds = []
        for feat in self.simple_features:  # group, profile, result, type, form, season
            embedded = self.embeddings_simple[feat](batch[feat])
            simple_embeds.append(embedded)
        
        simple_concat = torch.cat(simple_embeds, dim=-1)  # [B, S, simple_embed_total]
        
        # 3. ДИАГНОЗЫ С ATTENTION
        diag_tensors = {
            'diagnosis_letter': batch['diagnosis_letter'],
            'diagnosis_hierarchy': batch['diagnosis_hierarchy'],
            'diagnosis_full': batch['diagnosis_full']
        }
        
        diag_processed, attention_weights = self._process_diagnoses(
            diag_tensors,
            numeric_features,
            batch['diagnosis_mask']
        )  # [B, S, diag_embed_total]
        
        # 4. УСЛУГИ
        service_embeds = []
        for level in self.service_levels:
            embedded = self.embeddings_service[level](batch[level])
            service_embeds.append(embedded)
        
        service_concat = torch.cat(service_embeds, dim=-1)  # [B, S, service_embed_total]
        
        # 5. ОБЪЕДИНЕНИЕ ВСЕХ ПРИЗНАКОВ
        combined = torch.cat([
            simple_concat,      # все простые категории (6 признаков)
            diag_processed,     # диагнозы (сумма эмбеддингов уровней)
            service_concat,     # услуги (сумма эмбеддингов уровней)
            numeric_features    # числовые признаки
        ], dim=-1)  # [B, S, total_input_dim]
        
        # 6. НОРМАЛИЗАЦИЯ (если используется)
        if self.layer_norm:
            combined = self.layer_norm(combined)
        
        # 7. MLP ПРЕОБРАЗОВАНИЕ
        mlp_out = self.mlp(combined)  # [B, S, mlp_hidden]
        mlp_out = mlp_out.view(B, S)
        
        # 8. LSTM ОБРАБОТКА
        lengths = batch['lengths'].cpu()
        packed_input = nn.utils.rnn.pack_padded_sequence(
            mlp_out,
            lengths,
            batch_first=True,
            enforce_sorted=False
        )
        
        packed_output, (hn, cn) = self.lstm(packed_input)
        last_hidden = hn[-1]  # [B, lstm_hidden]
        last_hidden = self.dropout(last_hidden)
        
        # 9. ПРЕДСКАЗАНИЯ
        predictions = {}
        
        # Регрессия
        predictions['age'] = self.head_age(last_hidden)  # [B, 1]
        predictions['death_logits'] = self.head_death(last_hidden)  # [B, 1]
        
        # Диагнозы
        for level in self.diag_levels:
            predictions[f'{level}_logits'] = self.head_diagnosis[level](last_hidden)
        
        # Услуги
        for level in self.service_levels:
            predictions[f'{level}_logits'] = self.head_service[level](last_hidden)
        
        # Все простые категории
        for feat in self.simple_features:
            predictions[f'{feat}_logits'] = self.head_simple[feat](last_hidden)
        
        if return_attention:
            return predictions, attention_weights
        
        return predictions
    
    def get_loss_weights(self):
        """Веса для разных loss функций."""
        weights = {
            'age': 1.0,
            'death': 1.5,
            'diagnosis_letter': 2.0,
            'diagnosis_hierarchy': 2.0,
            'diagnosis_full': 2.0,
            'service_letter': 1.5,
            'service_hierarchy': 1.5,
            'service_full': 1.5,
            'group': 0.7,
            'profile': 0.7,
            'result': 1.0,
            'type': 0.7,
            'form': 0.7,
            'season': 0.7
        }
        return weights
    
    def get_total_params(self):
        """Возвращает общее количество параметров модели."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable