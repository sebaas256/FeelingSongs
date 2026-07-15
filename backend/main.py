from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os 
import requests
from dotenv import load_dotenv
from pinecone import Pinecone 

#Cargar las variables de entorno desde el arvhibo .env 
load_dotenv()

HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

HF_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
HF_API_URL = f"https://router.huggingface.co/hf-inference/models/{HF_MODEL}/pipeline/feature-extraction"
HF_HEADERS = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}

#conexion a pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("buscador-musical")

#Inicializar la aplicacion FastAPI
app = FastAPI(title="FeelingSogns API")

#Configurar CORS (esto permite que astro por medio del puerto 4321 se comunique con el backend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],  #con esto se permite get, post, etc.
    allow_headers=["*"]
)

#Definir el modelo de datos que recibiremos del front
class SearchQuery(BaseModel):
    emocion: str
    
#Ruta test
@app.get("/")
def read_root():
    return {"mensaje": "FeelingSongs API esta corriendo correctamente"}

def obtener_vector(texto):
    response = requests.post(HF_API_URL, headers=HF_HEADERS, json={"inputs": [texto]})
    if response.status_code != 200:
        print(f"Error en Hugging Face ({response.status_code}): {response.text}")
        return None
    return response.json()[0]

MIN_SCORE = 0.25

def escalar_a_porcentaje(resultados, min_display=60, max_display=95):
    """Remapea los scores crudos (0.0-1.0) de una búsqueda a un rango
    visualmente coherente con la UI, sin pretender que sean un porcentaje
    absoluto de 'verdad' — es un ranking relativo dentro de esa búsqueda."""
    scores = [r["score"] for r in resultados]
    score_min, score_max = min(scores), max(scores)
    rango = score_max - score_min

    porcentajes = []
    for r in resultados:
        if rango == 0:
            porcentajes.append(max_display)
        else:
            p = min_display + (r["score"] - score_min) / rango * (max_display - min_display)
            porcentajes.append(round(p))
    return porcentajes

@app.post("/api/buscar")
def buscar_cancion(query: SearchQuery):
    print(f"El usuario esta buscando la emocion: {query.emocion}")

    vector = obtener_vector(query.emocion)
    if vector is None:
        return {"query_recibida": query.emocion, "resultados": [], "error": "No se pudo generar el embedding"}

    resultado_pinecone = index.query(vector=vector, top_k=6, include_metadata=True)
    matches_validos = [m for m in resultado_pinecone["matches"] if m["score"] >= MIN_SCORE]

    if not matches_validos:
        return {"query_recibida": query.emocion, "resultados": []}

    porcentajes = escalar_a_porcentaje(matches_validos)

    resultados = []
    for match, porcentaje in zip(matches_validos, porcentajes):
        meta = match["metadata"]
        resultados.append({
            "titulo": meta.get("titulo"),
            "artista": meta.get("artista"),
            "anio": meta.get("anio"),
            "match": f"{porcentaje}%",
        })

    return {"query_recibida": query.emocion, "resultados": resultados}