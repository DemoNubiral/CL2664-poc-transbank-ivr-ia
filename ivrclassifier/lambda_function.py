import os
import re
import json
import logging
import numpy as np
import joblib  # Usando joblib
from typing import Optional
import pandas as pd
from tqdm import tqdm
# pandarallel se maneja dentro del __init__
from src.bedrock_models_v2 import LLMClient, ModelConfig  # Asumiendo que LLMClient se usa aquí

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted


class IvrClassifier(BaseEstimator, ClassifierMixin):
    """
    Clasificador basado en LLM que utiliza técnicas de extracción de patrones para clasificar textos.

    El LLM para extraer patrones (`pep_llm`) se pasa directamente al método `fit`.

    Parámetros (__init__)
    --------------------
    pep : str
        Prompt plantilla para la extracción de patrones (debe contener {categoria} y {texto_categoria}).
    general_prompt : str
        Plantilla del prompt final a enviar al LLM de INFERENCIA (debe contener {all_context} y {user_text}).
    model_config : dict, optional (default=None)
        Configuración del modelo LLM a utilizar en INFERENCIA (`predict`).
        Si es None, se usará la configuración por defecto (ej: Claude 3.7 Sonnet).
    temperature : float, optional (default=0.0)
        Temperatura para la generación de texto en el LLM DE INFERENCIA.
    max_tokens : int, optional (default=500)
        Número máximo de tokens a generar en la respuesta del LLM DE INFERENCIA.
    enable_reasoning : bool, optional (default=False)
        Si se debe habilitar el razonamiento explícito en el LLM DE INFERENCIA.
    reasoning_budget : int, optional (default=0)
        Presupuesto de tokens para el razonamiento en el LLM DE INFERENCIA, si está habilitado.
    n_workers : int, optional (default=4)
        Número de workers para procesamiento paralelo con pandarallel (si está disponible).
    patterns_output_path : str, optional (default=None)
        Ruta opcional donde guardar el JSON de patrones extraídos durante el `fit`.
    """

    def __init__(
        self,
        pep: str = None,  # Prompt para generar patrones (usado en fit)
        general_prompt: str = None,  # Prompt para inferencia (usado en predict)
        model_config: Optional[dict] = None,  # Config LLM para INFERENCIA
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
        # Es mejor configurar logging fuera de la clase o usar un logger pasado
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        # Obtener logger específico para esta clase
        self.logger = logging.getLogger(self.__class__.__name__)

        # Inicialización de atributos internos
        self._pandarallel_initialized = False
        self._use_parallel = False  # Por defecto no usar paralelo

        # Intentar inicializar pandarallel de forma centralizada si n_workers > 1
        if self.n_workers > 1:
            try:
                from pandarallel import pandarallel

                pandarallel.initialize(nb_workers=self.n_workers, progress_bar=True)
                self._use_parallel = True
                self.logger.info(
                    f"Procesamiento paralelo inicializado con {self.n_workers} workers"
                )
            except ImportError:
                self.logger.warning(
                    "Pandarallel no instalado. Se usará procesamiento secuencial."
                )
            except Exception as e:
                self.logger.warning(
                    f"Pandarallel disponible pero fallo en la inicialización: {e}. Se usará procesamiento secuencial."
                )
        else:
            self.logger.info("n_workers <= 1, se usará procesamiento secuencial.")

        # Imprime parámetros de inicialización
        self.logger.info(f"Inicializando {self.__class__.__name__} con parámetros:")
        params = self.get_params()
        for param_name, param_value in params.items():
            if param_name in ["pep", "general_prompt"] and isinstance(param_value, str):
                preview = param_value[:50] + "..." if len(param_value) > 50 else param_value
                self.logger.info(f"  - {param_name}: '{preview}'")
            else:
                self.logger.info(f"  - {param_name}: {param_value}")

    def fit(self, X, y, pep_llm: LLMClient, force_training=False):
        """
        Ajusta el clasificador extrayendo patrones usando un LLM proporcionado.

        Si `patterns_output_path` apunta a un archivo JSON existente y `force_training=False`,
        se cargarán los patrones desde allí en lugar de usar `pep_llm`.

        Parámetros
        ----------
        X : array-like de shape (n_samples,)
            Textos de entrenamiento.
        y : array-like de shape (n_samples,)
            Etiquetas de categoría correspondientes.
        pep_llm : LLMClient
            Instancia del cliente LLM configurada para ser usada en la extracción
            de patrones durante este `fit`. **No se guarda con el modelo.**
        force_training : bool, optional (default=False)
            Si es True, fuerza la extracción de patrones incluso si existe
            `patterns_output_path`.

        Retorna
        -------
        self : object
            Instancia ajustada del clasificador (con `patterns_dict_`, `classes_`, etc.).
        """
        self.logger.info("Iniciando proceso fit...")

        # Input validation para X, y
        X, y = check_X_y(X, y, dtype=str, ensure_2d=False)  # Asegurar que X sea string

        # Verificar si podemos cargar patrones existentes
        can_load_patterns = (
            self.patterns_output_path
            and os.path.exists(self.patterns_output_path)
            and not force_training
        )

        if can_load_patterns:
            try:
                self.logger.info(
                    f"Intentando cargar patrones desde {self.patterns_output_path}..."
                )
                self.load_patterns_from_json(self.patterns_output_path)
                # load_patterns_from_json debe definir self.classes_ y self.patterns_dict_
                self.logger.info(
                    f"Patrones cargados exitosamente desde {self.patterns_output_path}. Omitiendo extracción con LLM."
                )
                self._is_fitted = True  # Marcar como ajustado
                return self
            except Exception as e:
                self.logger.warning(
                    f"Error al cargar patrones desde {self.patterns_output_path}: {e}. Se procederá con la extracción."
                )

        # Si llegamos aquí, necesitamos extraer patrones usando pep_llm
        self.logger.info("Extrayendo patrones usando el LLM proporcionado (`pep_llm`).")
        if not self.pep:
            raise ValueError(
                "El atributo 'pep' (prompt de extracción de patrones) no puede ser None para entrenar."
            )
        if not isinstance(pep_llm, LLMClient):
            raise TypeError("El argumento 'pep_llm' debe ser una instancia de LLMClient.")

        # Registrar categorías únicas y asegurar consistencia
        df = pd.DataFrame({"text": X, "category": y})
        unique_classes = df["category"].unique()
        self.classes_ = np.array(sorted(unique_classes))  # Guardar ordenado
        self.logger.info(f"Categorías detectadas y ordenadas: {self.classes_.tolist()}")

        # Agrupar textos por categoría
        all_categories_info = df.groupby("category")["text"].apply(lambda texts: " ".join(texts)).to_dict()

        self.logger.info(f"Extrayendo patrones para {len(self.classes_)} categorías...")

        # Inicializar el diccionario para almacenar los patrones extraídos
        self.patterns_dict_ = {}

        # Procesar cada categoría
        # Usar self.classes_ para asegurar el orden y la inclusión de todas las categorías
        for category in tqdm(self.classes_, desc="Extrayendo patrones"):
            raw_text = all_categories_info.get(category, "")  # Obtener texto, default a vacío si falta
            if not raw_text:
                self.logger.warning(
                    f"No se encontraron textos para la categoría '{category}'. Se generará un patrón vacío."
                )
                self.patterns_dict_[category] = ""  # O algún placeholder
                continue  # Pasar a la siguiente categoría

            # Formatear el prompt PEP con la categoría y el texto agrupado
            try:
                formatted_prompt = self.pep.format(categoria=category, texto_categoria=raw_text)
            except KeyError as e:
                self.logger.error(
                    f"Error al formatear el prompt 'pep'. Asegúrate de que contenga {{categoria}} y {{texto_categoria}}. Detalle: {e}"
                )
                raise ValueError("Error en el formato del prompt 'pep'.") from e

            # Invocar el LLM de extracción de patrones (`pep_llm`)
            try:
                # Usar los parámetros de la instancia aquí puede no ser lo ideal
                # Podrías querer pasar parámetros específicos para PEP
                # Por ejemplo, podrías añadir pep_temperature, pep_max_tokens a __init__
                # O definir valores fijos aquí
                pep_temp = 0.1  # Ejemplo: temperatura específica para PEP
                pep_max_tokens = 1000  # Ejemplo: más tokens para patrones

                response = pep_llm.chat(
                    prompt=formatted_prompt,
                    temperature=pep_temp,  # Usar temp específica para PEP
                    max_tokens=pep_max_tokens,  # Usar max_tokens específico para PEP
                    # Pasar enable_reasoning/budget si pep_llm los soporta y son deseados
                    # enable_reasoning=self.enable_reasoning, # O pep_enable_reasoning
                    # reasoning_budget=self.reasoning_budget, # O pep_reasoning_budget
                )
                self.patterns_dict_[category] = response.strip() if response else ""
                self.logger.debug(
                    f"Patrón extraído para '{category}': {self.patterns_dict_[category][:100]}..."
                )
            except Exception as e:
                self.logger.error(
                    f"Error al invocar pep_llm para la categoría '{category}': {e}", exc_info=True
                )
                # Decide qué hacer: ¿asignar patrón vacío? ¿fallar?
                self.patterns_dict_[category] = "ERROR_EXTRACCION_PATRON"
                # Opcional: relanzar si un error aquí es crítico
                # raise RuntimeError(f"Fallo al extraer patrón para {category}") from e

        # Crear el contexto completo concatenando los patrones de todas las categorías
        # Asegurar que el orden coincida con self.classes_
        context_parts = []
        for category in self.classes_:
            pattern = self.patterns_dict_.get(category, "")
            context_parts.append(f"Categoría: {category}\nPatrones:\n{pattern}\n")

        self.all_patterns_context_ = "\n---\n".join(context_parts).strip()
        self.logger.debug(
            f"Contexto de patrones completo generado:\n{self.all_patterns_context_[:500]}..."
        )

        # Guardar el JSON de patrones si se especificó una ruta
        if self.patterns_output_path:
            self.logger.info(f"Guardando patrones extraídos en {self.patterns_output_path}...")
            self.save_patterns_json(self.patterns_output_path)

        self._is_fitted = True  # Marcar como ajustado al final
        self.logger.info("Ajuste completado. El clasificador está listo para predecir.")
        return self

    def load_patterns_from_json(self, path: str):
        """
        Carga los patrones extraídos y las clases desde un archivo JSON.
        Establece `patterns_dict_`, `classes_`, y `all_patterns_context_`.

        Parámetros
        ----------
        path : str
            Ruta del archivo JSON con los patrones.

        Retorna
        -------
        self : object
            Instancia del clasificador con patrones cargados.
        """
        self.logger.debug(f"Cargando patrones desde archivo JSON: {path}")
        try:
            with open(path, "r", encoding="utf-8") as file:
                patterns_data = json.load(file)

            # Cargar el diccionario de patrones
            self.patterns_dict_ = patterns_data.get("patterns_dict", {})
            if not isinstance(self.patterns_dict_, dict):
                raise ValueError("El campo 'patterns_dict' en el JSON no es un diccionario.")

            # Cargar las clases (importante que coincidan con las keys de patterns_dict)
            if "classes" in patterns_data and patterns_data["classes"] is not None:
                self.classes_ = np.array(sorted(patterns_data["classes"]))  # Cargar y ordenar
                # Validar consistencia
                if set(self.classes_) != set(self.patterns_dict_.keys()):
                    self.logger.warning(
                        "Las claves en 'patterns_dict' no coinciden exactamente con 'classes' en el JSON. Se usarán las claves de 'patterns_dict'."
                    )
                    self.classes_ = np.array(sorted(self.patterns_dict_.keys()))
            else:
                # Si no hay 'classes', derivarlas de las keys del diccionario y ordenar
                self.logger.warning(
                    "No se encontró el campo 'classes' en el JSON. Se derivarán de las claves de 'patterns_dict'."
                )
                self.classes_ = np.array(sorted(self.patterns_dict_.keys()))

            self.logger.info(f"Clases cargadas y ordenadas: {self.classes_.tolist()}")

            # Recrear siempre el all_patterns_context_ a partir de patterns_dict_ y classes_ (para asegurar orden)
            context_parts = []
            for category in self.classes_:
                pattern = self.patterns_dict_.get(category, "")  # Obtener patrón
                context_parts.append(f"Categoría: {category}\nPatrones:\n{pattern}\n")
            self.all_patterns_context_ = "\n---\n".join(context_parts).strip()

            self.logger.info(f"Patrones y clases cargados exitosamente desde {path}")
            self._is_fitted = True  # Marcar como ajustado
            return self
        except FileNotFoundError:
            self.logger.error(f"Error al cargar patrones: Archivo no encontrado en {path}")
            raise
        except json.JSONDecodeError as e:
            self.logger.error(f"Error al cargar patrones: El archivo JSON en {path} es inválido. {e}")
            raise ValueError(f"Archivo JSON inválido: {path}") from e
        except Exception as e:
            self.logger.error(f"Error inesperado al cargar patrones desde {path}: {e}", exc_info=True)
            raise

    def save_patterns_json(self, path: str):
        """
        Guarda los patrones (`patterns_dict_`) y clases (`classes_`) actuales en un archivo JSON.

        Parámetros
        ----------
        path : str
            Ruta donde guardar el archivo JSON.
        """
        # Asegurarse de que el modelo esté ajustado (tenga patrones y clases)
        if not hasattr(self, "patterns_dict_") or not hasattr(self, "classes_"):
            self.logger.warning(
                "El método 'fit' no parece haber sido completado. No se guardarán patrones JSON."
            )
            return

        # Obtener el directorio de la ruta
        dirname = os.path.dirname(path)

        # Si hay un directorio especificado, asegurarse de que exista
        if dirname:
            try:
                os.makedirs(dirname, exist_ok=True)
            except OSError as e:
                self.logger.error(
                    f"No se pudo crear el directorio {dirname} para guardar patrones JSON: {e}"
                )
                return  # No continuar si no se puede crear el directorio

        # Preparar el diccionario para guardar (usar atributos actuales)
        patterns_output = {
            "patterns_dict": self.patterns_dict_,
            "classes": self.classes_.tolist() if self.classes_ is not None else None,
        }

        # Guardar en un archivo JSON
        try:
            with open(path, "w", encoding="utf-8") as file:
                json.dump(patterns_output, file, ensure_ascii=False, indent=4)
            self.logger.info(f"Patrones y clases guardados en formato JSON en {path}")
        except TypeError as e:
            self.logger.error(
                f"Error de tipo al serializar patrones a JSON en {path}. Asegúrate de que `patterns_dict_` contenga tipos serializables. Error: {e}"
            )
        except IOError as e:
            self.logger.error(f"Error de E/S al guardar patrones JSON en {path}: {e}")
        except Exception as e:
            self.logger.error(
                f"Error inesperado al guardar patrones JSON en {path}: {e}", exc_info=True
            )

    def _extract_choice(self, text: str) -> str:
        """
        Extrae el valor asociado a "choice" dentro de un bloque JSON en el texto.
        Intenta parsear JSON primero, luego usa regex como fallback.

        Parámetros
        ----------
        text : str
            El texto de la respuesta del LLM.

        Retorna
        -------
        str
            La categoría extraída o un string indicando error (ej: "ERROR_EXTRACCION").
        """
        if not text or not isinstance(text, str):
            self.logger.warning("Entrada inválida para _extract_choice (None o no string).")
            return "ERROR_EXTRACCION_INPUT"

        # Intenta encontrar un bloque JSON completo (flexible con espacios al inicio/final)
        json_match = re.search(r"^\s*(\{.*?\})\s*$", text, re.DOTALL)
        if json_match:
            json_block = json_match.group(1)
            try:
                # Intentar parsear el bloque JSON encontrado
                json_data = json.loads(json_block)
                if isinstance(json_data, dict) and "choice" in json_data:
                    choice = json_data["choice"]
                    # Validar que sea un string no vacío
                    if isinstance(choice, str) and choice.strip():
                        self.logger.debug(f"Categoría extraída vía JSON: '{choice}'")
                        return choice.strip()
                    else:
                        self.logger.warning(
                            f"Campo 'choice' encontrado en JSON pero no es un string válido o está vacío: {choice}"
                        )
                else:
                    self.logger.warning(
                        f"Bloque JSON parseado pero no contiene la clave 'choice' o no es un diccionario: {json_data}"
                    )
            except json.JSONDecodeError:
                # Si falla el parseo del bloque JSON, loguear y continuar al fallback regex
                self.logger.warning(f"Se encontró un bloque parecido a JSON pero falló el parseo: '{json_block[:100]}...'")

        # Fallback a la regex si no se encontró/parseó JSON con 'choice' válido
        self.logger.debug("Intentando extracción de 'choice' con regex como fallback...")
        pattern = r'"choice"\s*:\s*"([^"]+)"'
        match = re.search(pattern, text)
        if match:
            choice = match.group(1).strip()
            if choice:
                self.logger.debug(f"Categoría extraída vía regex: '{choice}'")
                return choice
            else:
                self.logger.warning("Regex encontró 'choice' pero el valor capturado está vacío.")

        # Si ninguna estrategia funcionó
        self.logger.warning(
            f"No se pudo extraer 'choice' de la respuesta mediante JSON o regex: '{text[:200]}...'"
        )
        return "ERROR_EXTRACCION"

    def _infer_single(self, text: str) -> str:
        """
        Realiza la inferencia para un único texto usando el LLM de inferencia.

        Parámetros
        ----------
        text : str
            El texto a clasificar.

        Retorna
        -------
        str
            La categoría predicha o un string indicando error/desconocido.
        """
        # 1. Validar que el modelo esté ajustado
        try:
            check_is_fitted(self, ["patterns_dict_", "all_patterns_context_", "classes_"])
        except NotFittedError:
            self.logger.error(
                "Error de inferencia: El método 'predict' fue llamado antes de que el modelo fuera ajustado ('fit')."
            )
            return "ERROR_MODELO_NO_AJUSTADO"

        # 2. Validar entrada
        if not text or not isinstance(text, str):
            self.logger.warning(
                f"Entrada inválida para _infer_single: '{text}'. Se requiere un string no vacío."
            )
            return "ERROR_INPUT_INVALIDO"

        # 3. Inicializar LLMClient para inferencia (se hace en cada llamada para evitar serialización)
        try:
            llm_config = self.model_config or ModelConfig.CLAUDE_3_7_SONNET.value
            llm_client = LLMClient(llm_config)
            self.logger.debug(f"LLMClient de inferencia inicializado con config: {llm_config}")
        except Exception as e:
            self.logger.error(
                f"Error al inicializar LLMClient para inferencia: {e}", exc_info=True
            )
            return "ERROR_INICIALIZACION_LLM"

        # 4. Formatear el prompt final
        if not self.general_prompt:
            self.logger.error("Error de inferencia: El atributo 'general_prompt' no está definido.")
            return "ERROR_PROMPT_INFERENCIA_FALTANTE"
        try:
            final_prompt = self.general_prompt.format(
                all_context=self.all_patterns_context_, user_text=text
            )
        except KeyError as e:
            self.logger.error(
                f"Error al formatear 'general_prompt'. Asegúrate de que contenga {{all_context}} y {{user_text}}. Detalle: {e}"
            )
            return "ERROR_FORMATO_PROMPT_INFERENCIA"
        except Exception as e:
            self.logger.error(
                f"Error inesperado al formatear 'general_prompt': {e}", exc_info=True
            )
            return "ERROR_FORMATO_PROMPT_INFERENCIA"

        # 5. Invocar el LLM de inferencia
        try:
            self.logger.debug(f"Invocando LLM de inferencia para texto: '{text[:50]}...'")
            response = llm_client.chat(
                prompt=final_prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                enable_reasoning=self.enable_reasoning,
            )
            if not response:
                self.logger.warning("El LLM de inferencia devolvió una respuesta vacía.")
                return "ERROR_RESPUESTA_LLM_VACIA"

            self.logger.debug(f"Respuesta recibida del LLM de inferencia: '{response[:200]}...'")
        except Exception as e:
            self.logger.error(
                f"Error al invocar el LLM de inferencia: {e}", exc_info=True
            )
            return "ERROR_INVOCACION_LLM"

        # 6. Extraer la categoría de la respuesta
        choice = self._extract_choice(response)

        # 7. Validar la categoría extraída
        if choice.startswith("ERROR_"):
            return choice

        known_classes_list = self.classes_.tolist() if hasattr(self, "classes_") and self.classes_ is not None else []

        if not known_classes_list:
            self.logger.warning(
                "No hay clases conocidas definidas en el modelo ('self.classes_'). Se devolverá la categoría extraída sin validación."
            )
            return choice

        if choice not in known_classes_list:
            self.logger.warning(
                f"Categoría predicha '{choice}' no está entre las categorías conocidas {known_classes_list}"
            )
            return "CATEGORIA_DESCONOCIDA"

        # 8. Devolver la categoría validada
        self.logger.info(f"Texto '{text[:50]}...' clasificado como: '{choice}'")
        return choice

    def predict(self, X):
        """
        Predice la categoría para un texto o una secuencia de textos.

        Parámetros
        ----------
        X : str or iterable of str
            Texto o textos a clasificar.

        Retorna
        -------
        str or np.ndarray
            Categoría predicha (si X es str) o array numpy de categorías predichas (si X es iterable).
        """
        try:
            check_is_fitted(self, ["patterns_dict_", "all_patterns_context_", "classes_"])
        except NotFittedError as e:
            self.logger.error("Error: Intento de predecir con un modelo no ajustado.")
            raise NotFittedError(
                "El modelo debe ser ajustado ('fit') antes de llamar a 'predict'."
            ) from e

        if isinstance(X, str):
            return self._infer_single(X)
        elif hasattr(X, "__iter__") and not isinstance(X, (str, bytes)):
            input_list = list(X)
            n_samples = len(input_list)

            if n_samples == 0:
                self.logger.info("Entrada para 'predict' es una secuencia vacía. Devolviendo array vacío.")
                return np.array([])

            self.logger.info(f"Iniciando predicción para {n_samples} textos...")

            X_series = pd.Series(input_list, dtype=object)

            use_parallel_processing = self._use_parallel and n_samples > 1

            if use_parallel_processing:
                try:
                    self.logger.info(
                        f"Usando procesamiento paralelo (pandarallel con {self.n_workers} workers)..."
                    )
                    predictions = X_series.parallel_apply(self._infer_single)
                except Exception as e:
                    self.logger.warning(
                        f"Fallo al usar pandarallel ({e}), volviendo a procesamiento secuencial con tqdm."
                    )
                    tqdm.pandas(desc=f"Predicción (Secuencial, {n_samples} textos)")
                    predictions = X_series.progress_apply(self._infer_single)
            else:
                if n_samples > 1:
                    self.logger.info("Usando procesamiento secuencial (tqdm)...")
                    tqdm.pandas(desc=f"Predicción (Secuencial, {n_samples} textos)")
                    predictions = X_series.progress_apply(self._infer_single)
                else:
                    self.logger.info("Procesando un solo texto de la secuencia.")
                    predictions = X_series.apply(self._infer_single)

            return predictions.values.astype(object)
        else:
            raise TypeError(
                f"La entrada X debe ser un string o un iterable de strings. Se recibió tipo: {type(X)}"
            )

    def save(self, path: str):
        """
        Guarda la instancia actual del clasificador (self) usando joblib.
        Solo guarda el estado necesario para la inferencia.
        El LLM de extracción de patrones (`pep_llm`) NO se guarda.

        Parámetros
        ----------
        path : str
            Ruta donde guardar el archivo del modelo (ej: 'model.joblib').
        """
        try:
            check_is_fitted(self, ["patterns_dict_", "all_patterns_context_", "classes_"])
            self.logger.info("Modelo ajustado, procediendo a guardar.")
        except NotFittedError:
            self.logger.warning(
                "Intentando guardar un modelo que no ha sido ajustado ('fit'). El modelo guardado podría no ser útil."
            )

        directory = os.path.dirname(path)

        if directory:
            try:
                os.makedirs(directory, exist_ok=True)
                self.logger.debug(f"Directorio asegurado: {directory}")
            except OSError as e:
                self.logger.error(f"No se pudo crear el directorio {directory} para guardar el modelo: {e}")
                raise

        try:
            joblib.dump(self, path, compress=3)
            self.logger.info(f"Modelo guardado exitosamente con joblib (comprimido) en: {path}")
        except Exception as e:
            self.logger.error(f"Error crítico al guardar el modelo con joblib en {path}: {e}", exc_info=True)
            raise

    @classmethod
    def load(cls, path: str):
        """
        Carga una instancia de IvrClassifier desde un archivo guardado con joblib.

        Parámetros
        ----------
        path : str
            Ruta del archivo del modelo guardado (ej: 'model.joblib').

        Retorna
        -------
        IvrClassifier
            Instancia del modelo cargado.
        """
        logger = logging.getLogger(cls.__name__)
        logger.info(f"Intentando cargar modelo con joblib desde: {path}")

        try:
            model = joblib.load(path)

            if not isinstance(model, cls):
                logger.error(
                    f"Error: El archivo cargado {path} no contiene una instancia de {cls.__name__}, sino de {type(model).__name__}."
                )
                raise TypeError(f"El archivo {path} no es un modelo {cls.__name__} válido.")

            logger.info(f"Modelo cargado exitosamente con joblib desde {path}")

            try:
                check_is_fitted(model, ["patterns_dict_", "all_patterns_context_", "classes_"])
                logger.info("El modelo cargado parece estar ajustado.")
            except NotFittedError:
                logger.warning("El modelo cargado desde {path} no parece estar ajustado (falta estado interno).")

            return model

        except FileNotFoundError:
            logger.error(f"Error crítico: No se encontró el archivo del modelo en la ruta especificada: {path}")
            raise
        except (
            joblib.externals.loky.process_executor.TerminatedWorkerError,
            EOFError,
            ImportError,
            TypeError,
        ) as e:
            logger.error(
                f"Error crítico al deserializar el modelo desde {path} con joblib. El archivo podría estar corrupto o ser incompatible. Error: {e}",
                exc_info=True,
            )
            raise RuntimeError(f"Fallo al cargar el modelo desde {path}. Archivo corrupto o incompatible.") from e
        except Exception as e:
            logger.error(
                f"Error inesperado al cargar el modelo con joblib desde {path}: {e}", exc_info=True
            )
            raise
