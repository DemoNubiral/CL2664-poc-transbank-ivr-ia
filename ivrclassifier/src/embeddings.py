import json
import logging
import boto3
from botocore.exceptions import ClientError
import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

MODEL_TITAN_ID = "amazon.titan-embed-text-v2:0"
MODEL_COHERE_ID = "cohere.embed-multilingual-v3"


def generate_cohere_embeddings(text, input_type="classification"):
    """
    Genera embeddings para el/los texto(s) usando el modelo Cohere Embed a través de AWS Bedrock
    y retorna únicamente los embeddings.

    Args:
        text (str o list): Texto o lista de textos a embedir.
        input_type (str): Tipo de entrada (por defecto "classification").
        embedding_types (list): Tipos de embeddings a solicitar (por defecto ["int8", "float"]).

    Returns:
        dict: Diccionario con los embeddings generados.
    """
    # Convertir a lista si es un único string
    texts = [text] if isinstance(text, str) else text

    # Construir el cuerpo de la solicitud
    body = json.dumps(
        {
            "texts": texts,
            "input_type": input_type,
        }
    )

    # logger.info("Generando embeddings para el texto usando el modelo %s", MODEL_ID)

    accept = "*/*"
    content_type = "application/json"
    bedrock = boto3.client(service_name="bedrock-runtime")

    try:
        response = bedrock.invoke_model(
            body=body, modelId=MODEL_COHERE_ID, accept=accept, contentType=content_type
        )
        # Decodificar el contenido de la respuesta (bytes a dict)
        response_body = json.loads(response["body"].read())
        # logger.info("Embeddings generados correctamente.")
        # Retornar únicamente los embeddings
        return response_body.get("embeddings")[0]
    except ClientError as err:
        message = err.response["Error"]["Message"]
        logger.error("Error del cliente: %s", message)
        raise


def generate_titan_embeddings(text, embedding_types=["float"]):
    """
    Genera el vector embedding para un texto usando Amazon Titan Text Embeddings V2.

    Args:
        text (str o list): Un único texto o una lista de textos. Si se provee más de uno, se usa el primero.
        embedding_types (list): Tipos de embedding a solicitar (por defecto ["float"]).

    Returns:
        dict: El embedding generado (la representación vectorial) obtenido del endpoint.
    """
    # Convertir a lista si se pasa un string
    texts = [text] if isinstance(text, str) else text

    if len(texts) > 1:
        logger.warning(
            "Se han proporcionado múltiples textos. Solo se utilizará el primero."
        )
    input_text = texts[0]

    # Construir el cuerpo de la solicitud
    body = json.dumps({"inputText": input_text, "embeddingTypes": embedding_types})

    try:
        bedrock = boto3.client(service_name="bedrock-runtime")
        response = bedrock.invoke_model(
            body=body,
            modelId=MODEL_TITAN_ID,
            accept="application/json",
            contentType="application/json",
        )
        response_body = json.loads(response.get("body").read())
        return response_body["embedding"]
    except ClientError as err:
        logger.error("Error generando el embedding: %s", err)
        raise
