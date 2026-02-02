"""
RuBERT-Tiny2 Embedding Service
FastAPI сервер для создания эмбеддингов русских текстов
"""

import logging
from typing import List, Optional
import numpy as np

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
import torch

# ==================== Конфигурация ====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==================== Модели данных ====================
class EmbeddingRequest(BaseModel):
    """Модель запроса для создания эмбеддингов"""
    texts: List[str] = Field(
        ..., 
        description="Список текстов для векторизации",
        example=["привет мир", "как дела?", "тестовый текст"]
    )
    normalize_embeddings: bool = Field(
        True, 
        description="Нормализовать ли эмбеддинги (рекомендуется)"
    )
    batch_size: int = Field(
        32, 
        description="Размер батча для обработки", 
        ge=1, 
        le=256
    )
    show_progress_bar: bool = Field(
        False, 
        description="Показывать прогресс-бар (для отладки)"
    )

class EmbeddingResponse(BaseModel):
    """Модель ответа с эмбеддингами"""
    embeddings: List[List[float]]
    model: str
    dimensions: int
    num_texts: int
    device: str

class HealthResponse(BaseModel):
    """Модель ответа для health check"""
    status: str
    model_loaded: bool
    device: str
    model_name: Optional[str] = None
    embedding_dimension: Optional[int] = None

# ==================== Инициализация ====================
app = FastAPI(
    title="RuBERT-Tiny2 Embedding Service",
    description="REST API для получения векторных представлений русских текстов",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Глобальные переменные для модели
_model = None
_device = None

def get_device() -> str:
    """Определяем доступное устройство"""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

def load_model():
    """Загружаем модель при старте приложения"""
    global _model, _device
    
    try:
        _device = get_device()
        logger.info(f"Определено устройство: {_device}")
        
        logger.info("Начинаю загрузку модели 'cointegrated/rubert-tiny2'...")
        
        # Загружаем модель с указанием кеш-директории
        _model = SentenceTransformer(
            model_name_or_path='cointegrated/rubert-tiny2',
            device=_device,
            cache_folder="/root/.cache/huggingface"
        )
        
        # Тестовый прогон
        test_texts = ["тестовый текст"]
        test_embeddings = _model.encode(
            test_texts, 
            normalize_embeddings=True,
            show_progress_bar=False
        )
        
        logger.info(f"✓ Модель успешно загружена!")
        logger.info(f"  Размерность эмбеддингов: {test_embeddings.shape[1]}")
        logger.info(f"  Устройство выполнения: {_model.device}")
        logger.info(f"  Макс. длина последовательности: {_model.max_seq_length}")
        
    except Exception as e:
        logger.error(f"✗ Ошибка загрузки модели: {str(e)}")
        _model = None
        raise

# Загружаем модель при старте приложения
@app.on_event("startup")
async def startup_event():
    """Событие запуска приложения"""
    load_model()

# ==================== Роуты API ====================
@app.get("/", include_in_schema=False)
async def root():
    """Перенаправление на документацию"""
    return {"message": "RuBERT-Tiny2 API. Перейдите на /docs для документации"}

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Проверка здоровья сервиса"""
    if _model is None:
        return HealthResponse(
            status="unhealthy",
            model_loaded=False,
            device=_device or "unknown"
        )
    
    return HealthResponse(
        status="healthy",
        model_loaded=True,
        device=str(_device),
        model_name="cointegrated/rubert-tiny2",
        embedding_dimension=_model.get_sentence_embedding_dimension()
    )

@app.post("/embed", response_model=EmbeddingResponse)
async def create_embeddings(request: EmbeddingRequest):
    """
    Создать эмбеддинги для списка текстов
    
    - **texts**: список текстов на русском языке
    - **normalize_embeddings**: нормализовать векторы (по умолчанию True)
    - **batch_size**: размер батча для обработки
    """
    if _model is None:
        raise HTTPException(
            status_code=503,
            detail="Модель не загружена. Попробуйте позже."
        )
    
    if not request.texts:
        raise HTTPException(
            status_code=400,
            detail="Список текстов не может быть пустым"
        )
    
    try:
        logger.info(f"Обработка {len(request.texts)} текстов...")
        
        # Создаем эмбеддинги
        embeddings = _model.encode(
            sentences=request.texts,
            normalize_embeddings=request.normalize_embeddings,
            batch_size=request.batch_size,
            show_progress_bar=request.show_progress_bar,
            convert_to_numpy=True
        )
        
        # Конвертируем в список для JSON
        embeddings_list = embeddings.tolist()
        
        logger.info(f"Успешно создано {len(embeddings_list)} эмбеддингов")
        
        return EmbeddingResponse(
            embeddings=embeddings_list,
            model="cointegrated/rubert-tiny2",
            dimensions=len(embeddings_list[0]) if embeddings_list else 0,
            num_texts=len(embeddings_list),
            device=str(_device)
        )
        
    except Exception as e:
        logger.error(f"Ошибка при создании эмбеддингов: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Внутренняя ошибка сервера: {str(e)}"
        )

@app.get("/model-info")
async def get_model_info():
    """Получить информацию о загруженной модели"""
    if _model is None:
        raise HTTPException(
            status_code=503,
            detail="Модель не загружена"
        )
    
    return {
        "model": "cointegrated/rubert-tiny2",
        "status": "loaded",
        "device": str(_device),
        "embedding_dimensions": _model.get_sentence_embedding_dimension(),
        "max_sequence_length": _model.max_seq_length,
        "normalize_embeddings_default": True,
        "supports_gpu": torch.cuda.is_available(),
        "model_size_mb": 29.4  # Из описания модели
    }

@app.post("/test-embedding")
async def test_embedding():
    """Тестовый запрос для проверки работы"""
    test_texts = [
        "Привет, мир!",
        "Как дела?",
        "Тестовое предложение для проверки эмбеддингов."
    ]
    
    request = EmbeddingRequest(
        texts=test_texts,
        normalize_embeddings=True,
        batch_size=2
    )
    
    return await create_embeddings(request)

# ==================== Обработка ошибок ====================
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Обработчик ошибок HTTP"""
    return {
        "error": True,
        "message": exc.detail,
        "status_code": exc.status_code
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)