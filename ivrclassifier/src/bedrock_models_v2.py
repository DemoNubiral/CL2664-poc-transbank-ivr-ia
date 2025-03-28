import os
import boto3
import logging
import time
import argparse
from enum import Enum
from rich import print  # Importa rich para prints con formato
from rich.console import Console

console = Console()

DEFAULT_REGION = "us-east-1"

class DummyLogger:
    """Logger que no hace nada."""
    def info(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass

class ModelConfig(Enum):
    """
    Configuraciones predefinidas para los modelos Anthropic.
    """
    CLAUDE_3_7_SONNET = {
        "model_id": "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
        "model_params": {
            "temperature": 0.0,
            "max_tokens": 8192,
            "enable_reasoning": True,
            "reasoning_budget": 4096,
        },
    }
    CLAUDE_3_5_SONNET = {
        "model_id": "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
        "model_params": {
            "temperature": 0.0,
            "max_tokens": 4000,
        },
    }
    CLAUDE_3_5_HAIKU = {
        "model_id": "us.anthropic.claude-3-5-haiku-20241022-v1:0",
        "model_params": {
            "temperature": 0.0,
            "max_tokens": 4000,
        },
    }

class LLMClient:
    """
    Clase para inicializar y gestionar la comunicación con el modelo Anthropic usando AWS Bedrock.
    Permite invocar el modelo en modo estándar o (si es compatible) con razonamiento extendido.

    Parámetros:
        model_config (dict): Configuración del modelo (ID y parámetros por defecto).
        region (str): Región de AWS para el cliente Bedrock.
        logger (logging.Logger): Logger opcional para depuración.
    """

    def __init__(self, model_config, region=DEFAULT_REGION, logger=None):
        # Si no se pasa logger, se asigna un DummyLogger que no imprime nada.
        self.logger = logger if logger is not None else DummyLogger()
        self.model_config = model_config
        self.region = region
        try:
            self.client = boto3.client(
                "bedrock-runtime",
                region_name=self.region,
            )
            self.logger.info("Cliente Bedrock-runtime inicializado correctamente.")
        except Exception as e:
            self.logger.error(f"Error al inicializar el cliente: {str(e)}")
            self.client = None

    def invoke(
        self,
        prompt,
        system_prompt=None,
        enable_reasoning=None,
        reasoning_budget=None,
        temperature=None,
        max_tokens=None,
    ):
        """
        Invoca al modelo con la configuración especificada.

        Parámetros:
            prompt (str): Texto a enviar al modelo.
            system_prompt (str): Mensaje del sistema para configurar el comportamiento del modelo.
                                 Por defecto: "You're a helpful AI assistant."
            enable_reasoning (bool): Si es True, activa el modo de razonamiento extendido (solo para modelos que lo soportan).
            reasoning_budget (int): Presupuesto de tokens para la fase de razonamiento (mínimo 1024 tokens).
            temperature (float): Temperatura para la generación (si se omite se usa la configuración por defecto).
            max_tokens (int): Tokens máximos para la respuesta (si se omite se usa la configuración por defecto).

        Retorna:
            dict: Respuesta completa del API, incluyendo el tiempo transcurrido.
        """
        if not self.client:
            self.logger.error("El cliente no está inicializado.")
            return None

        # Extraer valores por defecto del modelo
        model_defaults = self.model_config.get("model_params", {})
        # Para modelos que soportan razonamiento extendido se toman los valores, sino se ignoran.
        if "claude-3-7-sonnet" in self.model_config["model_id"].lower():
            if enable_reasoning is None:
                enable_reasoning = model_defaults.get("enable_reasoning", False)
            if reasoning_budget is None:
                reasoning_budget = model_defaults.get("reasoning_budget", 1024)
        else:
            # Para CLAUDE_3_5_SONNET se ignoran enable_reasoning y reasoning_budget
            enable_reasoning = False
            reasoning_budget = None

        gen_temperature = (
            temperature
            if temperature is not None
            else model_defaults.get("temperature", 1.0)
        )
        gen_max_tokens = (
            max_tokens
            if max_tokens is not None
            else model_defaults.get("max_tokens", 4000)
        )

        # Permitir pasar un system_prompt personalizado; si no se provee, usar el valor por defecto.
        if system_prompt is None:
            system_prompt = "You're a helpful AI assistant."
        formatted_system_prompt = [{"text": system_prompt}]

        # Definir el mensaje del usuario
        messages = [{"role": "user", "content": [{"text": prompt}]}]

        # Configuración básica de inferencia
        inference_config = {
            "temperature": gen_temperature,
            "maxTokens": gen_max_tokens,
        }

        request_params = {
            "modelId": self.model_config["model_id"],
            "messages": messages,
            "system": formatted_system_prompt,
            "inferenceConfig": inference_config,
        }

        # Si el modelo soporta razonamiento extendido, se incluyen esos parámetros
        if enable_reasoning and reasoning_budget is not None:
            request_params["inferenceConfig"]["temperature"] = 1.0  # Forzar temperature=1.0 en modo razonamiento extendido
            if gen_max_tokens <= reasoning_budget:
                adjusted_max_tokens = reasoning_budget + 1
                self.logger.info(
                    f"Ajustando max_tokens de {gen_max_tokens} a {adjusted_max_tokens} para sobrepasar el presupuesto de razonamiento"
                )
                request_params["inferenceConfig"]["maxTokens"] = adjusted_max_tokens

            request_params["additionalModelRequestFields"] = {
                "reasoning_config": {
                    "type": "enabled",
                    "budget_tokens": reasoning_budget,
                }
            }
        else:
            if enable_reasoning:
                self.logger.warning(
                    "El modelo seleccionado no soporta razonamiento extendido. Se ignorarán los parámetros de razonamiento."
                )

        # Imprimir la configuración de la solicitud para chequear si se ha activado razonamiento
        self.logger.info(f"Request parameters: {request_params}")

        # Invocar el modelo y medir el tiempo de respuesta
        start_time = time.time()
        response = self.client.converse(**request_params)
        elapsed_time = time.time() - start_time
        response["_elapsed_time"] = elapsed_time

        return response

    def chat(
        self,
        prompt,
        system_prompt=None,
        enable_reasoning=None,
        reasoning_budget=None,
        temperature=None,
        max_tokens=None,
        append_reasoning=False
    ):
        """
        Wrapper de la función invoke que retorna únicamente el texto de la respuesta, soportando
        diferentes formatos de JSON y con opción de incluir el texto de razonamiento.

        Parámetros:
            prompt (str): Texto de entrada a enviar al modelo.
            system_prompt (str): Mensaje del sistema para configurar el comportamiento del modelo.
                                 Por defecto: "You're a helpful AI assistant."
            enable_reasoning (bool, opcional): Activa el razonamiento extendido (si es soportado).
            reasoning_budget (int, opcional): Presupuesto de tokens para el razonamiento extendido.
            temperature (float, opcional): Temperatura para la generación.
            max_tokens (int, opcional): Número máximo de tokens para la respuesta.
            append_reasoning (bool, opcional): Si es True, se incluye el texto presente en "reasoningContent".

        Retorna:
            str: El contenido concatenado de los valores extraídos de "text".
        """
        response = self.invoke(
            prompt=prompt,
            system_prompt=system_prompt,
            enable_reasoning=enable_reasoning,
            reasoning_budget=reasoning_budget,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        contents = response.get("output", {}).get("message", {}).get("content", [])
        textos = []

        for item in contents:
            # Se extrae siempre el texto directo, si existe.
            if "text" in item:
                textos.append(item["text"])
            # Si se solicita incluir razonamiento y está presente, se extrae el texto de razonamiento.
            if append_reasoning and "reasoningContent" in item:
                razonamiento = item.get("reasoningContent", {}).get("reasoningText", {}).get("text", "")
                if razonamiento:
                    textos.append(razonamiento)

        return "\n".join(textos)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Declarar las variables manualmente
    model_choice = (
        "claude_3_7_sonnet"  # Opciones: "claude_3_7_sonnet" o "claude_3_5_sonnet"
    )
    enable_reasoning = True  # Solo aplica para Claude 3.7 Sonnet
    reasoning_budget = 2**12  # Presupuesto de tokens para razonamiento extendido (solo para Claude 3.7)
    max_tokens = 2**13      # Cantidad máxima de tokens para la respuesta

    prompt = """
    Mejora este prompt para llm, quizás no fue un error si no que el usuario simplemente nos saludó o algo así. En el caso que no sea un error contale que eres NubiBot, una IA que lo ayudará a extraer información de pozos petroliferos.

    FAILURE_MESSAGE_PROMPT = `
    Genera un mensaje adecuado y educado para el cliente explicando que hubo el siguiente error: {error}.
    Puede ser que no haya habido un error, revisa con el mensaje del usuario:
    {user_message}
    Máximo un párrafo.
    Finaliza el mensaje con un: 'Disculpas, NubiBot.'
    `
    """

    # Seleccionar la configuración del modelo según la variable model_choice
    if model_choice == "claude_3_7_sonnet":
        model_config = ModelConfig.CLAUDE_3_7_SONNET.value
    elif model_choice == "claude_3_5_sonnet":
        model_config = ModelConfig.CLAUDE_3_5_SONNET.value
    elif model_choice == "claude_3_5_haiku":
        model_config = ModelConfig.CLAUDE_3_5_HAIKU.value
    else:
        console.print(
            "[bold red]Modelo no reconocido. Selecciona uno de los modelos disponibles.[/bold red]"
        )
        exit()

    # Para el modelo 3.7 se sobrescriben enable_reasoning y reasoning_budget
    if model_choice == "claude_3_7_sonnet":
        model_config["model_params"]["enable_reasoning"] = enable_reasoning
        model_config["model_params"]["reasoning_budget"] = reasoning_budget
    model_config["model_params"]["max_tokens"] = max_tokens

    # Comprobar y mostrar si el razonamiento extendido está activado (solo para Claude 3.7)
    if model_choice == "claude_3_7_sonnet":
        if model_config["model_params"].get("enable_reasoning", False):
            console.print(
                "[bold green]Razonamiento extendido ACTIVADO para Claude 3.7 Sonnet.[/bold green]"
            )
        else:
            console.print(
                "[bold red]Razonamiento extendido NO activado para Claude 3.7 Sonnet.[/bold red]"
            )
    else:
        console.print(
            "[bold blue]Modelo Claude 3.5 Sonnet seleccionado; razonamiento extendido no es soportado.[/bold blue]"
        )

    # Crear la instancia del cliente sin pasar logger (no se imprimirá nada)
    llm_client = LLMClient(model_config, region=DEFAULT_REGION, logger=None)

    # Invocar el modelo usando los parámetros correspondientes
    response = llm_client.chat(
        prompt=prompt,
        # Puedes también pasar un system_prompt personalizado:
        # system_prompt="Eres NubiBot, una IA experta en extracción de información de pozos petrolíferos.",
        # enable_reasoning=model_config["model_params"].get("enable_reasoning", False),
        # reasoning_budget=model_config["model_params"].get("reasoning_budget"),
        # max_tokens=model_config["model_params"].get("max_tokens")
    )

    console.print("\n[bold magenta]Respuesta del modelo:[/bold magenta]")
    console.print(response)
