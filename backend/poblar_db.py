import os
import requests
import time
from dotenv import load_dotenv
from pinecone import Pinecone

# Cargar variables de entorno
load_dotenv()
HF_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Configurar Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("buscador-musical")

# Modelo MULTILINGÜE (antes usábamos all-MiniLM-L6-v2, que es solo inglés
# y no distinguía bien el significado de texto en español). Este modelo
# también genera vectores de 384 dimensiones, así que el índice no cambia.
HF_API_URL = "https://router.huggingface.co/hf-inference/models/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2/pipeline/feature-extraction"
headers = {"Authorization": f"Bearer {HF_API_KEY}"} 

# Catálogo de 50 canciones de los 70s y 80s.
# Las descripciones fueron revisadas contra el significado real/documentado
# de cada canción (entrevistas, Songfacts, etc.) en vez de la lectura pop
# superficial, y se varió la estructura de la frase en cada una para que
# los embeddings las puedan diferenciar mejor entre sí.
canciones = [
    {"id": "1", "titulo": "Looking for Love", "artista": "Whitesnake", "anio": 1987, "emocion": "David Coverdale canta a la soledad de quien todavía no encuentra pareja: una power ballad de hard rock sobre la vulnerabilidad de seguir buscando el amor sin perder la esperanza."},
    {"id": "2", "titulo": "Heaven", "artista": "Bryan Adams", "anio": 1984, "emocion": "Soft rock de devoción absoluta, donde estar junto a la persona amada se describe literalmente como tocar el paraíso; un amor tranquilo y sin condiciones."},
    {"id": "3", "titulo": "Wish You Were Here", "artista": "Pink Floyd", "anio": 1975, "emocion": "No es una canción de amor romántico, sino un lamento de Roger Waters por Syd Barrett, el fundador de la banda que se perdió en la enfermedad mental; también critica la frialdad y el vacío del negocio de la música."},
    {"id": "4", "titulo": "Alone Again", "artista": "Dokken", "anio": 1985, "emocion": "Heavy metal melódico centrado en el vacío que deja una ruptura reciente, esa mezcla de arrepentimiento y soledad que aparece cuando todos se han ido y solo queda el silencio."},
    {"id": "5", "titulo": "Don't Dream It's Over", "artista": "Crowded House", "anio": 1986, "emocion": "Pop rock de resistencia emocional: Neil Finn habla de sostener la esperanza y la conexión con alguien cuando fuerzas externas —una relación tensa, el mundo mismo— empujan a rendirse."},
    {"id": "6", "titulo": "Waiting for a Girl Like You", "artista": "Foreigner", "anio": 1981, "emocion": "Balada de sintetizadores sobre el asombro incrédulo de haber encontrado por fin, después de mucho esperar, a la persona exacta que se había estado buscando toda la vida."},
    {"id": "7", "titulo": "Total Eclipse of the Heart", "artista": "Bonnie Tyler", "anio": 1983, "emocion": "Jim Steinman escribió esta power ballad originalmente para un musical sobre vampiros; sobrevivió como un himno de necesidad desesperada en la oscuridad, un corazón que se apaga sin la otra persona."},
    {"id": "8", "titulo": "Can't Fight This Feeling", "artista": "REO Speedwagon", "anio": 1984, "emocion": "Balada rock sobre rendirse ante lo inevitable: enamorarse en secreto de una amistad cercana y finalmente no poder seguir escondiendo lo que se siente."},
    {"id": "9", "titulo": "Telephone Line", "artista": "Electric Light Orchestra", "anio": 1976, "emocion": "Rock sinfónico construido alrededor de una espera angustiante junto al teléfono, marcando un número que nadie contesta, mientras la noche avanza y la esperanza se apaga poco a poco."},
    {"id": "10", "titulo": "Wishing (If I Had a Photograph of You)", "artista": "A Flock of Seagulls", "anio": 1982, "emocion": "Synth-pop atmosférico sobre la distancia y el anhelo: no tener ni siquiera una imagen a la que aferrarse de alguien que ya no está cerca."},
    {"id": "11", "titulo": "Take On Me", "artista": "a-ha", "anio": 1985, "emocion": "Synth-pop urgente y luminoso, una invitación directa a arriesgarse por un romance fugaz antes de que el momento se escape, con la energía de una persecución imposible de ignorar."},
    {"id": "12", "titulo": "Careless Whisper", "artista": "George Michael", "anio": 1984, "emocion": "Balada con saxofón sobre la culpa que sigue a una infidelidad: la sensación de que unos pies que antes bailaban con inocencia ya nunca podrán volver a hacerlo con la misma libertad."},
    {"id": "13", "titulo": "Drive", "artista": "The Cars", "anio": 1984, "emocion": "Ric Ocasek explicó que esta canción nace de la preocupación real por alguien que se está autodestruyendo; un synth-pop frágil sobre observar impotente cómo una persona pierde el control de su propia vida."},
    {"id": "14", "titulo": "Africa", "artista": "Toto", "anio": 1982, "emocion": "Inspirada en un documental nocturno y en relatos de misioneros, cuenta la historia de un hombre dividido entre su vocación de servicio al continente africano y el anhelo de compañía y amor personal."},
    {"id": "15", "titulo": "Time After Time", "artista": "Cyndi Lauper", "anio": 1983, "emocion": "Pop suave sobre la lealtad incondicional: la promesa de estar ahí para atrapar a alguien si tropieza, sin importar cuánto tiempo o distancia se interponga."},
    {"id": "16", "titulo": "Every Breath You Take", "artista": "The Police", "anio": 1983, "emocion": "Sting escribió esto durante su propio divorcio; a pesar de sonar romántica, la letra describe vigilancia obsesiva y posesión, el lado oscuro de no poder soltar a alguien."},
    {"id": "17", "titulo": "Here I Go Again", "artista": "Whitesnake", "anio": 1982, "emocion": "Hard rock de autosuficiencia: aceptar el destino de vagar en solitario como un lobo, encontrando fuerza y libertad en la independencia en lugar de lamentarla."},
    {"id": "18", "titulo": "Missing You", "artista": "John Waite", "anio": 1984, "emocion": "Soft rock sobre la negación: repetirse a uno mismo que ya se superó a un ex, mientras cada frase delata que en realidad todavía duele muchísimo."},
    {"id": "19", "titulo": "Separate Ways (Worlds Apart)", "artista": "Journey", "anio": 1983, "emocion": "Rock de arena sobre el momento exacto en que dos caminos se separan para siempre, aceptando el fin de una relación mientras aún se desea lo mejor para la otra persona."},
    {"id": "20", "titulo": "Everybody Wants to Rule the World", "artista": "Tears for Fears", "anio": 1985, "emocion": "New wave con un trasfondo político: una mirada cínica a cómo el ansia de poder y control corrompe incluso a quienes prometen actuar por el bien común."},
    {"id": "21", "titulo": "Against All Odds (Take a Look at Me Now)", "artista": "Phil Collins", "anio": 1984, "emocion": "Balada de piano desnuda, escrita para una película, donde alguien ruega por una última oportunidad sabiendo que probablemente ya es demasiado tarde para recuperar lo perdido."},
    {"id": "22", "titulo": "Enjoy the Silence", "artista": "Depeche Mode", "anio": 1990, "emocion": "Synth-pop que empezó como una balada lenta: la idea de que las palabras solo traen conflicto y que el verdadero refugio del amor está en el silencio compartido."},
    {"id": "23", "titulo": "True", "artista": "Spandau Ballet", "anio": 1983, "emocion": "Sophisti-pop suave sobre la timidez de no saber cómo confesarle a alguien lo que realmente se siente, atrapado entre el deseo de ser sincero y el miedo a decirlo mal."},
    {"id": "24", "titulo": "Alone", "artista": "Heart", "anio": 1987, "emocion": "Power ballad sobre el amor guardado en secreto durante demasiado tiempo: noches de insomnio imaginando el valor que hará falta para finalmente decir la verdad."},
    {"id": "25", "titulo": "I Just Died in Your Arms", "artista": "Cutting Crew", "anio": 1986, "emocion": "Pop rock sobre un encuentro casual que se vuelve emocionalmente abrumador de forma inesperada, dejando una mezcla extraña de placer intenso y arrepentimiento inmediato."},
    {"id": "26", "titulo": "Sweet Dreams (Are Made of This)", "artista": "Eurythmics", "anio": 1983, "emocion": "Synth-pop hipnótico y algo siniestro sobre cómo todo el mundo busca algo distinto —placer, poder, control, sumisión— y aun así el consejo final es mantener la cabeza en alto y seguir adelante."},
    {"id": "27", "titulo": "With or Without You", "artista": "U2", "anio": 1987, "emocion": "Rock atmosférico sobre una tensión irresoluble entre el compromiso y la libertad personal, no poder vivir plenamente ni con esa persona ni sin ella."},
    {"id": "28", "titulo": "I Still Haven't Found What I'm Looking For", "artista": "U2", "anio": 1987, "emocion": "Rock con raíces de góspel sobre una búsqueda espiritual inquieta, la sensación de tener fe y aun así sentir que algo esencial todavía falta por encontrar."},
    {"id": "29", "titulo": "Forever Young", "artista": "Alphaville", "anio": 1984, "emocion": "Detrás de su sonido festivo, esta canción synth-pop nace del miedo real a una guerra nuclear en plena Guerra Fría: preferir idealizar la juventud eterna antes que envejecer bajo la amenaza constante del fin del mundo."},
    {"id": "30", "titulo": "Bette Davis Eyes", "artista": "Kim Carnes", "anio": 1981, "emocion": "Pop rock rasposo sobre la fascinación peligrosa que provoca una mujer seductora y calculadora, consciente de su propio poder para manipular a quien se acerque."},
    {"id": "31", "titulo": "Boys Don't Cry", "artista": "The Cure", "anio": 1979, "emocion": "Post-punk ágil sobre el arrepentimiento de haber ocultado las lágrimas por orgullo, reprimiendo el dolor de una ruptura causada por un error propio."},
    {"id": "32", "titulo": "Just Like Heaven", "artista": "The Cure", "anio": 1987, "emocion": "Robert Smith describió el origen de esta canción como un mareo real junto a quien luego sería su esposa; rock alternativo que transmite el vértigo eufórico de enamorarse hasta perder el equilibrio."},
    {"id": "33", "titulo": "In the Air Tonight", "artista": "Phil Collins", "anio": 1981, "emocion": "No trata de un ahogamiento ni de una venganza, como dice el mito popular: Collins confirmó que nace de la ira y la desolación crudas de su propio divorcio, la tensión que se acumula antes de que todo termine por romperse."},
    {"id": "34", "titulo": "Never Tear Us Apart", "artista": "INXS", "anio": 1988, "emocion": "Balada rock con saxofón sensual sobre una conexión que se siente inevitable, la certeza de que ninguna fuerza externa podrá separar a dos personas hechas la una para la otra."},
    {"id": "35", "titulo": "Holding Back the Years", "artista": "Simply Red", "anio": 1985, "emocion": "Mick Hucknall escribió esto a los 17 años sobre el abandono de su madre cuando él tenía tres años; una súplica soul por dejar atrás el peso de una infancia rota y poder por fin avanzar."},
    {"id": "36", "titulo": "Right Here Waiting", "artista": "Richard Marx", "anio": 1989, "emocion": "Balada de piano escrita para su entonces novia mientras ella filmaba en el extranjero: la promesa de esperar el tiempo que haga falta para volver a estar juntos."},
    {"id": "37", "titulo": "I Want to Know What Love Is", "artista": "Foreigner", "anio": 1984, "emocion": "Power ballad con coros góspel sobre el agotamiento de la soledad, buscando con sinceridad casi religiosa entender qué es realmente el amor verdadero."},
    {"id": "38", "titulo": "Keep On Loving You", "artista": "REO Speedwagon", "anio": 1980, "emocion": "Kevin Cronin la escribió tras perdonar una infidelidad de su entonces esposa; balada rock sobre la decisión consciente de seguir amando a alguien pese a haber sido herido."},
    {"id": "39", "titulo": "Still Loving You", "artista": "Scorpions", "anio": 1984, "emocion": "Power ballad desesperada donde alguien se arrodilla, literalmente, para rogar una última oportunidad tras haber cometido un error que casi termina con la relación."},
    {"id": "40", "titulo": "Purple Rain", "artista": "Prince", "anio": 1984, "emocion": "Balada que mezcla rock, góspel y R&B con un significado que el propio Prince nunca aclaró del todo: arrepentimiento, purificación y el deseo de guiar a alguien a través de la tormenta hasta un lugar mejor."},
    {"id": "41", "titulo": "Tainted Love", "artista": "Soft Cell", "anio": 1981, "emocion": "Synth-pop bailable pero oscuro sobre sentirse envenenado por un amor tóxico y la urgencia de huir de una relación que ya solo hace daño."},
    {"id": "42", "titulo": "Bizarre Love Triangle", "artista": "New Order", "anio": 1986, "emocion": "Dance rock electrónico sobre la confusión de un amor no correspondido de la forma que uno desearía, esa angustia dulce de no saber bien qué lugar se ocupa en el corazón de otra persona."},
    {"id": "43", "titulo": "True Colors", "artista": "Cyndi Lauper", "anio": 1986, "emocion": "Balada pop reconfortante escrita para consolar a un amigo triste, recordándole que su belleza verdadera sigue ahí incluso cuando él mismo no logra verla."},
    {"id": "44", "titulo": "All Out of Love", "artista": "Air Supply", "anio": 1980, "emocion": "Soft rock dramático sobre sentirse completamente vacío tras la partida de un amor, reconociendo los propios errores y rogando en voz baja una segunda oportunidad."},
    {"id": "45", "titulo": "Making Love Out of Nothing at All", "artista": "Air Supply", "anio": 1983, "emocion": "Escrita por Jim Steinman con su estilo grandilocuente habitual: alguien que sabe hacer casi cualquier cosa en la vida excepto explicar el misterio de cómo se ama profundamente a otra persona."},
    {"id": "46", "titulo": "I Can't Go for That (No Can Do)", "artista": "Daryl Hall & John Oates", "anio": 1981, "emocion": "Pop soul con bajo protagonista sobre poner límites firmes en una relación, negándose de forma tranquila pero terminante a ser controlado o manipulado."},
    {"id": "47", "titulo": "Open Arms", "artista": "Journey", "anio": 1981, "emocion": "Balada rock sobre la vulnerabilidad de volver a abrir el corazón después de un tiempo separados, recibiendo a alguien de vuelta con los brazos abiertos y sin rencor."},
    {"id": "48", "titulo": "Sweet Child O' Mine", "artista": "Guns N' Roses", "anio": 1987, "emocion": "Axl Rose escribió esto sobre su entonces novia Erin Everly; en contraste con la imagen ruda de la banda, es una canción tierna sobre una sonrisa que devuelve la inocencia perdida de la infancia."},
    {"id": "49", "titulo": "November Rain", "artista": "Guns N' Roses", "anio": 1991, "emocion": "Basada en un relato corto sobre una boda marcada por la tragedia, esta épica orquestal de rock habla de aceptar que el cambio y la pérdida son inevitables, incluso en el amor más grande."},
    {"id": "50", "titulo": "Wind of Change", "artista": "Scorpions", "anio": 1990, "emocion": "No es una escena genérica de ciudad: nació tras un concierto de la banda en Moscú en 1989 y celebra explícitamente la caída de la Unión Soviética y el fin de la Guerra Fría, con un mensaje de esperanza y hermandad entre pueblos antes enfrentados."},
]

def obtener_vector(texto):
    response = requests.post(HF_API_URL, headers=headers, json={"inputs": [texto]})
    if response.status_code != 200:
        print(f"Error en Hugging Face ({response.status_code}): {response.text}")
        return None
    return response.json()[0]

print(f"Iniciando la carga de {len(canciones)} canciones a la Inteligencia Artificial...")

vectores_a_guardar = []
for index_cancion, cancion in enumerate(canciones):
    print(f"Sintonizando [{index_cancion + 1}/{len(canciones)}]: {cancion['titulo']}...")
    vector = obtener_vector(cancion["emocion"])

    if vector:
        vectores_a_guardar.append((
            cancion["id"],
            vector,
            {
                "titulo": cancion["titulo"],
                "artista": cancion["artista"],
                "anio": cancion["anio"]
            }
        ))

    # Pequeña pausa para no saturar la API gratuita de Hugging Face
    time.sleep(0.5)

# Subimos todo de golpe a Pinecone (mismos IDs = sobrescribe los vectores viejos)
if vectores_a_guardar:
    index.upsert(vectors=vectores_a_guardar)
    print(f"¡Éxito supremo! Las {len(vectores_a_guardar)} canciones han sido vectorizadas y guardadas en Pinecone.")
else:
    print("No se pudo procesar ninguna canción.")