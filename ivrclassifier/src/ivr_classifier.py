import os
import re
import json
import logging
import numpy as np
import pickle
from typing import Dict, List, Union, Optional
import pandas as pd
from tqdm import tqdm
from pandarallel import pandarallel
from src.bedrock_models_v2 import LLMClient, ModelConfig

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

class IvrClassifier(BaseEstimator, ClassifierMixin):
    """
    Clasificador basado en LLM que utiliza técnicas de extracción de patrones para clasificar textos.
    
    Parámetros
    ----------
    pep : str
        Prompt para la extracción de patrones.
    pep_llm : 
        Llm específica para generar la KB para cada categoría.
    general_prompt : str
        Plantilla del prompt final a enviar al LLM.
    model_config : dict, optional (default=None)
        Configuración del modelo LLM a utilizar en INFERENCIA. Si es None, se usará la configuración por defecto.
    temperature : float, optional (default=0.0)
        Temperatura para la generación de texto en el LLM DE INFERENCIA.
    max_tokens : int, optional (default=500)
        Número máximo de tokens a generar en la respuesta del LLM DE INFERENCIA.
    enable_reasoning : bool, optional (default=False)
        Si se debe habilitar el razonamiento explícito en el LLM DE INFERENCIA.
    reasoning_budget : int, optional (default=0)
        Presupuesto de tokens para el razonamiento en el LLM DE INFERENCIA, si está habilitado.
    n_workers : int, optional (default=4)
        Número de workers para procesamiento paralelo.
    patterns_output_path : str, optional (default=None)
        Ruta donde guardar el JSON de patrones extraídos después del fit.
    """
    
    def __init__(
        self,
        pep: str = None,
        general_prompt: str = None,
        model_config: Optional[dict] = None,
        temperature: float = 0.0,
        max_tokens: int = 500,
        enable_reasoning: bool = False,
        reasoning_budget: int = 0,
        n_workers: int = 4,
        patterns_output_path: Optional[str] = None,
    ):
        self.pep = pep
        self.general_prompt = general_prompt
        self.model_config = model_config
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.enable_reasoning = enable_reasoning
        self.reasoning_budget = reasoning_budget
        self.n_workers = n_workers
        self.patterns_output_path = patterns_output_path
        
        # Configuración de logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Inicialización de atributos internos
        self._pandarallel_initialized = False

        # Intentar inicializar pandarallel de forma centralizada
        try:
            from pandarallel import pandarallel
            pandarallel.initialize(nb_workers=self.n_workers, progress_bar=True)
            self._use_parallel = True
            self.logger.info(f"Procesamiento paralelo inicializado con {self.n_workers} workers")
        except Exception as e:
            self._use_parallel = False
            self.logger.warning(f"Pandarallel no disponible o fallo en la inicialización: {e}. Se usará procesamiento secuencial.")
        
        # Imprime todos los parámetros
        self.logger.info("Inicializando NubiralClassifier con los siguientes parámetros:")
        for param_name, param_value in self.get_params().items():
            # Omitir la impresión completa de prompts largos
            if param_name in ['pep', 'general_prompt'] and param_value is not None:
                if isinstance(param_value, str):
                    preview = param_value[:50] + '...' if len(param_value) > 50 else param_value
                    self.logger.info(f"  - {param_name}: {preview}")
                else:
                    self.logger.info(f"  - {param_name}: {type(param_value)}")
            else:
                self.logger.info(f"  - {param_name}: {param_value}")

    def fit(self, X, y, force_training=False):
        """
        Ajusta el clasificador con los datos de entrenamiento o carga patrones existentes.
        
        Si patterns_output_path apunta a un archivo existente y force_training=False, 
        se cargarán los patrones en lugar de entrenar.
        
        Parámetros
        ----------
        X : array-like de shape (n_samples, )
            Textos de entrenamiento.
        y : array-like de shape (n_samples, )
            Etiquetas de categoría correspondientes.
            
        Retorna
        -------
        self : object
            Instancia ajustada del clasificador.
        """
        # Verificar si podemos cargar patrones existentes
        if self.patterns_output_path and os.path.exists(self.patterns_output_path) and not force_training:
            try:
                self.load_patterns_from_json(self.patterns_output_path)
                self.logger.info(f"Patrones cargados desde {self.patterns_output_path}. Omitiendo entrenamiento.")
                return self
            except Exception as e:
                self.logger.warning(f"Error al cargar patrones desde {self.patterns_output_path}: {e}")
                self.logger.info("Continuando con entrenamiento normal.")
        
        # Si llegamos aquí, necesitamos entrenar el modelo
        
        # Validación de entrada
        X, y = check_X_y(X, y, dtype=None, ensure_2d=False)
        
        # Registrar categorías únicas
        self.classes_ = pd.Series(y).unique()
        self.logger.info(f"Categorías detectadas: {self.classes_}")
        
        # Agrupar textos por categoría
        df = pd.DataFrame({'text': X, 'category': y})
        all_categories_info = df.groupby("category")["text"].apply(lambda x: " ".join(x)).to_dict()
        
        self.logger.info("Extrayendo patrones para cada categoría...")
        
        # Inicializar el diccionario para almacenar los patrones extraídos
        self.patterns_dict_ = {}
        
        # Procesar cada categoría
        for category, raw_text in tqdm(all_categories_info.items(), desc="Extrayendo patrones"):
            # Formatear el prompt PEP con la categoría y el texto agrupado
            formatted_prompt = self.pep.format(
                categoria=category,
                texto_categoria=raw_text
            )
            
            # Invocar el LLM para extraer patrones
            response = pep_llm.chat(
                prompt=formatted_prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                reasoning_budget=self.reasoning_budget,
                enable_reasoning=self.enable_reasoning
            )
            
            # Almacenar el resultado
            self.patterns_dict_[category] = response
        
        # Crear el contexto completo concatenando los patrones de todas las categorías
        self.all_patterns_context_ = "\n\n".join(
            [f"{k}: {v}" for k, v in self.patterns_dict_.items()]
        )
        
        # Guardar el JSON de patrones si se especificó una ruta
        if self.patterns_output_path:
            self.save_patterns_json(self.patterns_output_path)
        
        self.logger.info("Ajuste completado. El clasificador está listo para predecir.")
        return self

    
    def load_patterns_from_json(self, path: str):
        """
        Carga los patrones extraídos desde un archivo JSON.
    
        Parámetros
        ----------
        path : str
            Ruta del archivo JSON con los patrones.
    
        Retorna
        -------
        self : object
            Instancia del clasificador con patrones cargados.
        """
        try:
            with open(path, 'r', encoding='utf-8') as file:
                patterns_data = json.load(file)
            
            # Cargar el diccionario de patrones
            self.patterns_dict_ = patterns_data.get("patterns_dict", {})
            
            # Recrear siempre el all_patterns_context_ a partir de patterns_dict_
            self.all_patterns_context_ = "\n\n".join(
                [f"{k}: {v}" for k, v in self.patterns_dict_.items()]
            ) if self.patterns_dict_ else ""
            
            # Omitir la carga de general_prompt ya que se pasa al inicializar el modelo
            
            # Cargar las clases
            if "classes" in patterns_data:
                self.classes_ = np.array(patterns_data["classes"])
            
            self.logger.info(f"Patrones cargados exitosamente desde {path}")
            return self
        except Exception as e:
            self.logger.error(f"Error al cargar patrones desde {path}: {e}")
            raise


    def save_patterns_json(self, path: str):
        """
        Guarda los patrones extraídos en un archivo JSON.
        
        Parámetros
        ----------
        path : str
            Ruta donde guardar el archivo JSON.
        """
        # Obtener el directorio de la ruta
        dirname = os.path.dirname(path)
        
        # Si hay un directorio especificado, asegurarse de que exista
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        
        # Preparar el diccionario para guardar
        patterns_output = {
            "patterns_dict": self.patterns_dict_,
            "classes": self.classes_.tolist() if hasattr(self, 'classes_') else []
        }
        
        # Guardar en un archivo JSON
        with open(path, 'w', encoding='utf-8') as file:
            json.dump(patterns_output, file, ensure_ascii=False, indent=4)
            
        self.logger.info(f"Patrones extraídos guardados en {path}")
    

    def _extract_choice(self, text: str) -> str:
        """
        Extrae el valor asociado a "choice" dentro de un bloque JSON en el texto.
        
        Parámetros
        ----------
        text : str
            El texto de la respuesta del LLM.
            
        Retorna
        -------
        str
            La categoría extraída o "ERROR" si no se puede extraer.
        """
        pattern = r'"choice"\s*:\s*"([^"]+)"'
        match = re.search(pattern, text)
        if match:
            return match.group(1)
        self.logger.warning("No se pudo extraer la categoría de la respuesta del LLM")
        return "ERROR"
    
    def _infer_single(self, text: str) -> str:
        """
        Realiza la inferencia para un único texto.
        
        Parámetros
        ----------
        text : str
            El texto a clasificar.
            
        Retorna
        -------
        str
            La categoría predicha.
        """
        try:
            check_is_fitted(self, ["patterns_dict_", "all_patterns_context_"])
            
            # Inicializar LLMClient
            llm_client = LLMClient(self.model_config or ModelConfig.CLAUDE_3_7_SONNET.value)
            
            # Formatear el prompt final
            final_prompt = self.general_prompt.format(
                all_context=self.all_patterns_context_,
                user_text=text
            )
            
            # Invocar el LLM
            response = llm_client.chat(
                prompt=final_prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                enable_reasoning=self.enable_reasoning
            )
            
            # Extraer la categoría de la respuesta
            choice = self._extract_choice(response)
            
            # Validar si la categoría está entre las clases conocidas
            if choice not in self.classes_ and choice != "ERROR":
                self.logger.warning(f"Categoría predicha '{choice}' no está entre las categorías conocidas {self.classes_}")
                return "ERROR"
                
            return choice
        except Exception as e:
            self.logger.error(f"Error durante la inferencia: {e}")
            return "ERROR"

    def predict(self, X):
        """
        Predice la categoría para cada texto de entrada.
        """
        check_is_fitted(self, ["patterns_dict_", "all_patterns_context_"])
        
        if isinstance(X, str):
            return self._infer_single(X)
        elif hasattr(X, '__iter__') and not isinstance(X, (str, bytes)):
            if len(X) == 0:
                return []
            elif len(X) == 1:
                return [self._infer_single(X[0])]
            else:
                self.logger.info(f"Procesando {len(X)} textos...")
                X_series = pd.Series(X)
                
                if self._use_parallel:
                    # Usar pandarallel si fue inicializado con éxito
                    predictions = X_series.parallel_apply(self._infer_single)
                else:
                    # Usar tqdm progress_apply en modo secuencial
                    tqdm.pandas(desc="Predicción")
                    predictions = X_series.progress_apply(self._infer_single)
                    
                return predictions.values
        else:
            raise ValueError("La entrada debe ser un string o un iterable de strings")

    
    def save(self, path: str):
        """
        Guarda el modelo entrenado en un archivo.
        
        Parámetros
        ----------
        path : str
            Ruta donde guardar el modelo.
        """
        # Extraer el directorio del path
        directory = os.path.dirname(path)
        
        # Crear el directorio y cualquier subdirectorio necesario si no existen
        if directory:
            os.makedirs(directory, exist_ok=True)
        
        # Guardar el modelo
        try:
            with open(path, 'wb') as file:
                pickle.dump(self, file)
            self.logger.info(f"Modelo guardado correctamente en {path}")
        except Exception as e:
            self.logger.error(f"Error al guardar el modelo en {path}: {str(e)}")
            raise
            
    @classmethod
    def load(cls, path: str):
        """
        Carga un modelo previamente guardado.
        
        Parámetros
        ----------
        path : str
            Ruta del modelo guardado.
            
        Retorna
        -------
        NubiralClassifier
            Modelo cargado.
        """
        with open(path, 'rb') as file:
            model = pickle.load(file)
        logging.getLogger(__name__).info(f"Modelo cargado desde {path}")
        return model