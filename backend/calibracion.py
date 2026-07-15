# calibrar.py — ejecuta esto una vez para ver tu rango real de scores
import os, requests
from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv()
HF_API_URL = "https://router.huggingface.co/hf-inference/models/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2/pipeline/feature-extraction"
headers = {"Authorization": f"Bearer {os.getenv('HUGGINGFACE_API_KEY')}"}
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("buscador-musical")

def vector_de(texto):
    r = requests.post(HF_API_URL, headers=headers, json={"inputs": [texto]})
    if r.status_code != 200:
        print(f"  >> Error HF ({r.status_code}): {r.text}")
        return None
    return r.json()[0]

pruebas = [
    "Nostalgia de los 80s",
    "Euforia nocturna en la ciudad",
    "receta de pasta con tomate",
    "cómo arreglar una llanta pinchada",
    "resultado del partido de fútbol de ayer",   # nuevo control limpio
    "cómo instalar Python en Windows",           # nuevo control limpio
]

for texto in pruebas:
    res = index.query(vector=vector_de(texto), top_k=5, include_metadata=True)
    print(f"\n--- {texto} ---")
    for m in res["matches"]:
        print(f"  {m['score']:.4f}  {m['metadata']['titulo']}")