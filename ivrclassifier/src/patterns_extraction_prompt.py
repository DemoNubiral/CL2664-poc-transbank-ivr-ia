PEP_ORIGINAL = """
Eres un experto analista de texto especializado en atención al cliente.  
Tu tarea es extraer patrones comunes de intención y problema a partir del siguiente texto, que pertenece a la categoría '{categoria}':

Texto de la Categoría:
```
{texto_categoria}
```

Instrucciones:

1.  Lee cuidadosamente el texto.
2.  Identifica los *tipos* de INTENCIONES más comunes que expresan los clientes en esta categoría.  Escribe una lista de estos patrones de intención.  Sé general, no copies intenciones específicas.
3.  Identifica los *tipos* de PROBLEMAS más comunes que experimentan los clientes en esta categoría. Escribe una lista de estos patrones de problema. Sé general, no copies problemas específicos.
5.  (Opcional) Identifica palabras clave.

Formato de Salida:

Patrones de Intención:
- [Patrón de intención 1]
- [Patrón de intención 2]
- ...

Patrones de Problema:
- [Patrón de problema 1]
- [Patrón de problema 2]
- ...

"""

PEP_GEMINI_PRO_EXP = """
Eres un experto analista de texto especializado en atención al cliente.  
Tu tarea es extraer patrones comunes de intención y problema a partir del siguiente texto, que pertenece a la categoría '{categoria}':

Texto de la Categoría:
```
{texto_categoria}
```

Instrucciones:

1.  Lee cuidadosamente el texto.
2.  Identifica los *tipos* de INTENCIONES más comunes que expresan los clientes en esta categoría, *sin incluir ejemplos concretos del texto*.  Escribe una lista concisa de estos patrones de intención, generalizando las intenciones observadas. Evita la redundancia; si dos intenciones son muy similares, generaliza a una sola.  Usa verbos en infinitivo.
3.  Identifica los *tipos* de PROBLEMAS más comunes que experimentan los clientes en esta categoría, *sin incluir ejemplos concretos del texto*.  Escribe una lista concisa de estos patrones de problema, generalizando los problemas observados. Evita la redundancia. Usa sustantivos abstractos, no verbos.
4.  Genera una lista de palabras clave relevantes para esta categoría, basadas en el texto proporcionado.  Incluye términos técnicos, nombres de productos/servicios, y palabras que reflejen las intenciones y problemas. Separa las palabras clave por comas.

Formato de Salida:

Patrones de Intención:
- [Patrón de intención 1]
- [Patrón de intención 2]
- ...

Patrones de Problema:
- [Patrón de problema 1]
- [Patrón de problema 2]
- ...

Palabras Clave:
[palabra clave 1], [palabra clave 2], [palabra clave 3], ...

"""

PEP_GEMINI_THINKING = """
Eres un experto analista de texto especializado en atención al cliente.
Tu tarea es extraer patrones comunes de intención y problema a partir del siguiente texto, que pertenece a la categoría '{categoria}':

Texto de la Categoría:

{texto_categoria}


Instrucciones:

1. Lee cuidadosamente el texto proporcionado.

2. **Patrones de Intención:** Identifica y extrae los patrones generales de intención más frecuentes expresados por los clientes en esta categoría.  Genera una lista de estos patrones de intención.  Generaliza las intenciones, no copies ejemplos textuales específicos del texto.

3. **Patrones de Problema:** Identifica y extrae los patrones generales de problema más frecuentes que experimentan los clientes en esta categoría. Genera una lista de estos patrones de problema. Generaliza los problemas, no copies ejemplos textuales específicos del texto.

4. (Opcional) Identifica palabras clave relevantes asociadas a las intenciones y problemas detectados.


Formato de Salida:

Patrones de Intención:

[Patrón de intención 1]

[Patrón de intención 2]

...

Patrones de Problema:

[Patrón de problema 1]

[Patrón de problema 2]

...

"""


PEP_R1  = """
Eres un analista experto en experiencia del cliente con 10 años de experiencia identificando patrones en servicio al cliente. 
Analiza el siguiente texto de la categoría '{categoria}' y extrae patrones fundamentales usando metodologías de análisis CX.

Texto Analizar:
```
{texto_categoria}
```

**Instrucciones Detalladas:**

1. **Análisis Contextual**:
   - Identificar contexto general de las interacciones
   - Reconocer sectores/industrias involucradas
   - Detectar lenguaje técnico específico

2. **Extracción de Patrones**:
   a) **Intenciones**:
   - Clasificar en máx. 10 categorías generales
   - Usar verbos de acción (Ej: "Solicitar", "Reportar")
   - Agrupar variaciones semánticas similares
   - Evitar duplicados y casos específicos

   b) **Problemas**:
   - Identificar raíces comunes (máx. 10)
   - Categorizar por tipo de falla (técnica/proceso/comunicación)
   - Priorizar frecuencia de aparición
   - Diferenciar síntomas vs causas reales

3. **Refinamiento**:
   - Validar coherencia entre intenciones y problemas
   - Eliminar redundancias
   - Asegurar nivel adecuado de abstracción
   - Verificar neutralidad técnica

4. **Palabras Clave**:
   - Extraer 8-15 términos técnicos relevantes
   - Incluir sinónimos y variaciones léxicas
   - Priorizar sustantivos y verbos clave

**Formato Requerido:**

**Categoría**: {categoria}
**Patrones de Intención**:
• [Tipo 1] - Descripción breve (2-6 palabras)
• [Tipo 2] - Descripción breve
[...]

**Patrones de Problema**:
◉ [Área afectada] - Naturaleza del problema (3-7 palabras)
◉ [Área afectada] - Naturaleza del problema 
[...]

**Lexema Central**:
▸ Término 1 | Término 2 | ... | Término N

**Reglas Estrictas**:
- Usar solo castellano neutro
- Máximo 10 items por sección
- Prohibido usar ejemplos concretos
- Evitar jerga corporativa
- Priorizar sustantivos sobre verbos
- Mantener paralelismo estructural
"""

PEP_SONNET_37 = """
Eres un experto analista de texto especializado en atención al cliente y extracción de insights.
Tu tarea es examinar meticulosamente el siguiente texto de la categoría '{categoria}' y extraer patrones recurrentes de intenciones y problemas del cliente.

Texto de la Categoría:

{texto_categoria}

## INSTRUCCIONES DETALLADAS:

1. ANÁLISIS EXHAUSTIVO:
   - Analiza todo el texto proporcionado, prestando especial atención a patrones recurrentes.
   - Identifica temas comunes que aparecen repetidamente en diferentes expresiones.

2. EXTRACCIÓN DE PATRONES DE INTENCIÓN:
   - Las INTENCIONES son lo que el cliente quiere lograr o el propósito de su contacto.
   - Formula entre 8-12 patrones de intención distintos y relevantes.
   - Cada patrón debe comenzar preferentemente con un verbo o frase de acción.
   - Los patrones deben ser suficientemente específicos para la categoría pero generalizables a múltiples casos.

3. EXTRACCIÓN DE PATRONES DE PROBLEMA:
   - Los PROBLEMAS son las dificultades, obstáculos o situaciones negativas que experimenta el cliente.
   - Formula entre 8-12 patrones de problema distintos y relevantes.
   - Enfócate en los problemas subyacentes, no en síntomas superficiales.
   - Evita repetir información ya incluida en los patrones de intención.

4. IDENTIFICACIÓN DE PALABRAS CLAVE:
   - Extrae 10-15 palabras o términos clave que aparecen frecuentemente en esta categoría.
   - Prioriza términos técnicos, productos específicos o conceptos relevantes para la categoría.
   - Separa las palabras clave por comas para facilitar su procesamiento posterior.

## CRITERIOS DE CALIDAD:

- GENERALIZACIÓN: Los patrones no deben ser ejemplos específicos, sino categorías que engloben múltiples casos similares.
- PRECISIÓN: Evita patrones demasiado genéricos que podrían aplicarse a cualquier categoría.
- RELEVANCIA: Prioriza los patrones más frecuentes o importantes para la categoría.
- CLARIDAD: Usa un lenguaje conciso y directo.
- COMPLETITUD: Asegúrate de cubrir los principales temas presentes en el texto.

## FORMATO DE SALIDA:

# Análisis de Categoría: {categoria}

## Patrones de Intención:
- [Patrón de intención 1]
- [Patrón de intención 2]
...

## Patrones de Problema:
- [Patrón de problema 1]
- [Patrón de problema 2]
...

## Palabras clave:
[palabra1], [palabra2], [palabra3], ...
"""

PEP_GPT = """
Eres un analista de texto experto en atención al cliente, con amplia experiencia en detectar tendencias y patrones recurrentes en comunicaciones. Tu tarea consiste en analizar el siguiente texto, correspondiente a la categoría '{categoria}', para extraer patrones generales de intención y problema que reportan los clientes.

Texto de la Categoría:
--------------------------------
{texto_categoria}
--------------------------------

Instrucciones:
1. Lee detenidamente el texto proporcionado.
2. Identifica y agrupa los tipos de INTENCIONES más comunes expresadas por los clientes. Evita reproducir frases o ejemplos puntuales; en su lugar, sintetiza patrones generales que reflejen tendencias o motivos recurrentes.
3. Identifica y agrupa los tipos de PROBLEMAS que experimentan los clientes en esta categoría. Al igual que en el caso de las intenciones, generaliza los problemas sin copiar casos específicos.
4. (Opcional) Extrae y lista palabras clave que resuman o destaquen los temas principales del texto.
5. Asegúrate de que tu respuesta sea clara, concisa y estructurada en listas, sin redundancias.

Formato de Salida:
------------------
Patrones de Intención:
- [Patrón de intención 1]
- [Patrón de intención 2]
- ...

Patrones de Problema:
- [Patrón de problema 1]
- [Patrón de problema 2]
- ...

Palabras Clave (Opcional):
- [Palabra clave 1]
- [Palabra clave 2]
- ...
"""
