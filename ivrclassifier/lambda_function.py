import json
import os
import boto3
from aws_lambda_powertools import Logger
from src.ivr_classifier import IvrClassifier

# Configuración de logging con aws-lambda-powertools
logger = Logger(service="ivr-classifier.")

# Configuración S3 - Obligatoria para cargar el modelo
S3_BUCKET = os.environ.get('MODEL_S3_BUCKET')
S3_KEY = os.environ.get('MODEL_S3_KEY')
MODEL_LOCAL_PATH = '/tmp/model.pkl'

# Variable global para almacenar el modelo cargado
model = None

def download_model_from_s3():
    """Descarga el modelo desde S3 al sistema de archivos local de Lambda"""
    if not S3_BUCKET or not S3_KEY:
        error_msg = "Las variables de entorno MODEL_S3_BUCKET y MODEL_S3_KEY son obligatorias"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    try:
        logger.info(f"Descargando modelo desde S3: {S3_BUCKET}/{S3_KEY}")
        s3_client = boto3.client('s3')
        s3_client.download_file(S3_BUCKET, S3_KEY, MODEL_LOCAL_PATH)
        logger.info(f"Modelo descargado correctamente a {MODEL_LOCAL_PATH}")
        return MODEL_LOCAL_PATH
    except Exception as e:
        logger.exception("Error al descargar el modelo desde S3")
        raise

def load_model():
    """Carga el modelo desde S3 y lo inicializa"""
    try:
        local_path = download_model_from_s3()
        logger.info(f"Cargando modelo desde {local_path}")
        loaded_model = IvrClassifier.load(local_path)
        logger.info("Modelo cargado en memoria correctamente")
        return loaded_model
    except Exception as e:
        logger.exception("Error al cargar el modelo")
        raise

def initialize():
    """Inicializa recursos globales"""
    global model
    if model is None:
        logger.info("Inicializando clasificador...")
        model = load_model()

# Cargar el modelo cuando se inicializa la función Lambda
try:
    initialize()
except Exception:
    logger.exception("Error durante la inicialización")
    # No elevamos la excepción aquí para permitir que Lambda se inicie

def lambda_handler(event, context):
    """
    Handler de AWS Lambda para clasificar textos.
    """
    try:
        logger.info("Recibido evento", extra={"event": event})
        
        if model is None:
            logger.info("Modelo no está cargado, intentando inicializar...")
            try:
                initialize()
                if model is None:
                    return format_response(500, {"error": "No se pudo cargar el modelo desde S3"})
            except Exception as e:
                logger.exception("Error al inicializar el modelo")
                return format_response(500, {"error": f"Error al cargar el modelo: {str(e)}"})
        
        text_to_classify = None
        
        if isinstance(event, dict) and 'text' in event:
            text_to_classify = event['text']
        elif isinstance(event, dict) and 'Records' in event:
            try:
                bucket_name = event['Records'][0]['s3']['bucket']['name']
                file_key = event['Records'][0]['s3']['object']['key']
                
                s3_client = boto3.client('s3')
                response = s3_client.get_object(Bucket=bucket_name, Key=file_key)
                text_to_classify = response['Body'].read().decode('utf-8').strip()
                logger.info("Texto extraído del archivo", extra={"file_key": file_key, "text": text_to_classify[:100]})
            except Exception:
                logger.exception("Error al procesar el archivo de texto desde S3")
                return format_response(500, {"error": "Error al procesar archivo desde S3"})
        elif isinstance(event, dict) and 'body' in event:
            try:
                body = event['body']
                body_dict = json.loads(body) if isinstance(body, str) else body
                text_to_classify = body_dict.get('text')
            except Exception:
                logger.exception("Error al procesar el body")
                return format_response(400, {"error": "Formato de solicitud inválido"})
        
        if not text_to_classify:
            return format_response(400, {"error": "No se proporcionó texto para clasificar"})
        
        logger.info("Clasificando texto", extra={"text_snippet": text_to_classify[:100]})
        prediction = model.predict(text_to_classify)
        
        response_data = {
            "text": text_to_classify[:100] + "..." if len(text_to_classify) > 100 else text_to_classify,
            "category": prediction if isinstance(prediction, str) else prediction.tolist() if hasattr(prediction, 'tolist') else prediction
        }
        
        return format_response(200, response_data)
    except Exception:
        logger.exception("Error en lambda_handler")
        return format_response(500, {"error": "Error interno"})

def format_response(status_code, body_dict):
    """Formatea la respuesta para API Gateway"""
    return {
        "statusCode": status_code,
        "body": json.dumps(body_dict, ensure_ascii=False),
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Content-Type",
            "Access-Control-Allow-Methods": "OPTIONS,POST,GET"
        }
    }