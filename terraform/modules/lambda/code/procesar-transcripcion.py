import json

def lambda_handler(event, context):
    """
    Función básica de Lambda en Python.
    
    Args:
        event (dict): Datos de entrada que activa la Lambda (por ejemplo, evento de S3 o API Gateway).
        context (object): Información del entorno de ejecución proporcionada por AWS.
    
    Returns:
        dict: Respuesta con un mensaje y el evento recibido.
    """
    print("Event received:", event)
    
    # Mensaje personalizado
    message = "Hello from AWS Lambda!"
    
    # Respuesta a devolver
    response = {
        "statusCode": 200,
        "body": json.dumps({
            "message": message,
            "event": event
        })
    }
    
    return response