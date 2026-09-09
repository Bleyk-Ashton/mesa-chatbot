import streamlit as st
import faiss
import pandas as pd
import numpy as np
import re
from sentence_transformers import SentenceTransformer

st.set_page_config(
    page_title="Asistente Mesa Estratégica de Servicios - Externado",
    page_icon="logo_externado.png",
    layout="wide",
)

# ------------------------------------------------------------
# Carga (cacheada: solo se ejecuta una vez, aunque muchos usuarios entren)
# ------------------------------------------------------------
@st.cache_resource
def cargar_todo():
    modelo = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    indice = faiss.read_index("chatbot_index.faiss")
    tabla = pd.read_csv("chatbot_respuestas.csv")
    return modelo, indice, tabla

embedding_model, index, df_bot = cargar_todo()

UMBRAL_CONFIANZA = 0.78
MARGEN_MINIMO = 0.03
K_VECINOS = 8

FUERA_DE_ALCANCE = [
    (r"\bbecas?\b|soy imparable|fundaci[oó]n carolina",
     "Las solicitudes de becas y ayudas económicas no se gestionan por este canal. "
     "Te voy a escalar a un agente de la Mesa para que te oriente sobre el proceso de becas."),
    (r"\bsap\b",
     "El acceso o soporte de la plataforma SAP no se gestiona por este canal. "
     "Te voy a escalar a un agente para que te ayude con eso."),
    (r"consultorio jur[ií]dico|asesor[ií]a jur[ií]dica|asesor[ií]a legal",
     "Para asesoría jurídica debes contactar directamente al Consultorio Jurídico de la Universidad; "
     "no es un trámite que gestione este canal. Te voy a escalar para que te redirijan correctamente."),
    (r"entrevista.{0,20}admisi[oó]n|admisi[oó]n.{0,20}entrevista|cu[aá]ndo.{0,10}entrevista",
     "Las preguntas sobre el proceso de admisión (entrevistas, fechas) no se gestionan por este canal. "
     "Te voy a escalar a un agente para que te oriente."),
    (r"inscribir materias|inscripci[oó]n de materias|horario.{0,15}(clase|academic)",
     "La inscripción de materias y horarios académicos no se gestiona por este canal. "
     "Te voy a escalar a un agente para que te ayude."),
]

REGLAS_CATEGORIA = [
    (r"(estado|c[oó]mo va|como va|avance|seguimiento).{0,20}(mi )?(caso|ticket|solicitud|radicado|req\b|inc\b)"
     r"|\bREQ\s?-?\s?\d+|\bINC\s?-?\s?\d+"
     r"|no (me han|he) (resuelto|dado respuesta)|llevo.{0,15}(d[ií]as|semanas).{0,15}esperando",
     "Consulta y seguimiento de tickets o solicitudes"),
    (r"retenci[oó]n en la fuente|certificado.{0,15}retenci[oó]n|\bICA\b|\bIVA\b",
     "Certificados tributarios"),
    (r"educaci[oó]n estrella", "Educación Estrella"),
    (r"me devolvieron|devolvieron mi solicitud|qu[eé] debo corregir|nota cr[eé]dito|reembolso|anulaci[oó]n de factura",
     "Devoluciones y correcciones"),
    (r"base(s)? de datos (jur[ií]dic|legal)|banco(s)? de datos (jur[ií]dic|legal)|credenciales.{0,15}(jur[ií]dic|legal)",
     "Bancos de datos jurídicos"),
    (r"\bcarn[eé]t?\b", "Carné digital"),
    (r"\bbloqueo\b|bloqueada|bloqueado", "Bloqueos financieros"),
    (r"autenticador|autentificador|doble factor|segundo factor|\bMFA\b|c[oó]digo.{0,15}(no llega|no me llega|verificaci[oó]n)",
     "Restablecimiento del doble factor"),
    (r"contrase[nñ]a.{0,15}correo|correo.{0,15}contrase[nñ]a", "Acceso y recuperación del correo institucional"),
    (r"whatsapp", "WhatsApp y canales institucionales"),
    (r"paz y salvo", "Paz y Salvo para grado"),
    (r"\bavante\b", "Problemas de acceso a Avante"),
    (r"aula virtual|\bava\b", "Integración y aulas académicas"),
    (r"multa|pr[eé]stamo.{0,15}(libro|biblioteca)|libro.{0,15}(vencido|atraso)|proactivanet",
     "Proactivanet"),
    # --- Nuevas: categorías que dependían solo del embedding (agregadas tras el lote de pruebas) ---
    (r"derechos de grado|pago de grado|recibo.{0,15}grado|pagar.{0,15}graduarme|graduarme.{0,15}pagar|"
     r"pago.{0,20}grado.{0,15}(rechazado|no lleg[oó]|confirmaci[oó]n)",
     "Pago de derechos de grado"),
    (r"certificado.{0,15}(valor|financiero|pagado)|valor neto|cuanto.{0,15}he pagado|certificado.{0,15}pagos",
     "Certificados financieros"),
    (r"certificado de (notas|estudio|t[ií]tulo)|constancia.{0,20}(materia|curso|estudio)|certificaci[oó]n.{0,15}acad[eé]mic",
     "Certificados académicos"),
    (r"\bmatr[ií]cula\b|orden de matr[ií]cula|renovar.{0,15}matr[ií]cula|matr[ií]cula.{0,15}(equivocad|error|mal)",
     "Matrícula y órdenes de matrícula"),
    (r"orden de pago|fraccionar|pasarela de pago|pago en l[ií]nea|\bPSE\b|no me deja pagar|"
     r"pago (rechazado|no disponible)|no encuentro (la )?orden|link.{0,10}pago|recibo.{0,10}pago|"
     r"cuanto.{0,15}(debo )?pagar|precio.{0,15}(semestre|matr[ií]cula)|descuento",
     "Órdenes de pago"),
]

SALUDOS_Y_RUIDO = {
    "hola", "buenos dias", "buenas tardes", "buenas noches", "buenas",
    "gracias", "muchas gracias", "mil gracias", "adios", "chao", "hey",
    "ok", "vale", "listo", "buen dia", "hi", "hello",
}

def es_input_no_informativo(texto):
    t = texto.strip().lower()
    t_sin_tildes = (t.replace("á", "a").replace("é", "e").replace("í", "i")
                      .replace("ó", "o").replace("ú", "u"))
    if t_sin_tildes in SALUDOS_Y_RUIDO:
        return True
    if len(t) < 8:
        return True
    palabras_reales = re.findall(r"[a-zñáéíóúü]{3,}", t)
    if len(palabras_reales) < 2:
        return True
    return False

def corregir_tildes(texto):
    """Corrige tildes que a veces se pierden en los datos históricos de correos."""
    reemplazos = {
        "Carne digital": "Carné digital", "carne digital": "carné digital",
        "Bancos de datos juridicos": "Bancos de datos jurídicos",
        "Restablecimiento del doble factor": "Restablecimiento del doble factor",
        "Educacion Estrella": "Educación Estrella",
        "Matricula y ordenes de matricula": "Matrícula y órdenes de matrícula",
        "Ordenes de pago": "Órdenes de pago",
        "Acceso y recuperacion del correo institucional": "Acceso y recuperación del correo institucional",
    }
    for malo, bueno in reemplazos.items():
        texto = texto.replace(malo, bueno)
    return texto

def responder(pregunta_usuario):
    if not pregunta_usuario or not pregunta_usuario.strip():
        return "Escribe tu solicitud, por favor."

    for patron, mensaje in FUERA_DE_ALCANCE:
        if re.search(patron, pregunta_usuario, flags=re.IGNORECASE):
            return mensaje

    for patron, categoria in REGLAS_CATEGORIA:
        if re.search(patron, pregunta_usuario, flags=re.IGNORECASE):
            respuesta = df_bot.loc[df_bot["categoria_nombre"] == categoria, "respuesta_generalizada"].iloc[0]
            return f"**{categoria}**\n\n{respuesta}"

    if es_input_no_informativo(pregunta_usuario):
        return (
            "¡Hola! Cuéntame un poco más sobre tu solicitud (por ejemplo: acceso al correo, "
            "matrícula, certificados, pagos, carné digital) para poder orientarte."
        )

    emb = embedding_model.encode([pregunta_usuario], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(emb)
    scores, idxs = index.search(emb, K_VECINOS)
    scores, idxs = scores[0], idxs[0]

    if float(scores[0]) < UMBRAL_CONFIANZA:
        return (
            "No estoy seguro de haber entendido bien tu solicitud. "
            "La voy a escalar a un agente de la Mesa Estratégica de Servicios para que te ayude directamente."
        )

    vecinos = df_bot.iloc[idxs].copy()
    vecinos["score"] = scores
    vecinos_validos = vecinos[vecinos["score"] >= UMBRAL_CONFIANZA]
    ranking = vecinos_validos.groupby("categoria_nombre")["score"].sum().sort_values(ascending=False)

    if len(ranking) >= 2 and (ranking.iloc[0] - ranking.iloc[1]) < MARGEN_MINIMO:
        return (
            "Tu solicitud podría estar relacionada con más de un trámite "
            f"(**{ranking.index[0]}** o **{ranking.index[1]}**). "
            "¿Puedes darme un poco más de detalle para orientarte mejor?"
        )

    categoria_ganadora = ranking.index[0]
    respuesta = vecinos_validos[vecinos_validos["categoria_nombre"] == categoria_ganadora]["respuesta_generalizada"].iloc[0]
    return f"**{categoria_ganadora}**\n\n{respuesta}"

# ------------------------------------------------------------
# Interfaz de chat — con identidad visual Universidad Externado de Colombia
# ------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Poppins:wght@400;500;600&display=swap');

    :root {
        --verde-externado: #0b5e3c;
        --verde-oscuro: #08462c;
        --dorado: #c9a227;
    }
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
        background: linear-gradient(180deg, #eef2ee 0%, #f7f8f6 60%, #eef2ee 100%) !important;
    }
    /* Centrar y limitar el contenido para que no se vea perdido en pantallas anchas */
    .block-container {
        max-width: 1100px;
        margin: 0 auto;
        padding-top: 2rem;
    }
    /* El cuadro de texto de abajo vive fuera del block-container por defecto: lo igualamos al mismo ancho */
    [data-testid="stBottomBlockContainer"],
    [data-testid="stChatInput"],
    .stChatFloatingInputContainer {
        max-width: 1100px !important;
        margin: 0 auto !important;
    }
    /* Tamaño de letra base un poco más grande */
    html, body, p, span, label, li, .stMarkdown, [data-testid="stChatMessageContent"] {
        font-size: 1.05rem;
    }
    /* Ocultar el título y ancho por defecto de Streamlit */
    #MainMenu, footer {visibility: hidden;}

    .encabezado-uec {
        background: linear-gradient(135deg, var(--verde-externado) 0%, var(--verde-oscuro) 100%);
        padding: 1.4rem 1.8rem;
        border-radius: 14px;
        margin-bottom: 1.2rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
        box-shadow: 0 4px 14px rgba(0,0,0,0.15);
    }
    .encabezado-uec h1 {
        font-family: 'Playfair Display', serif;
        color: white;
        font-size: 1.6rem;
        margin: 0;
        line-height: 1.3;
    }
    .encabezado-uec p {
        font-family: 'Poppins', sans-serif;
        color: #e3ecE7;
        margin: 0.2rem 0 0 0;
        font-size: 0.9rem;
    }

    /* Tipografía general de la app (excluye íconos para no romperlos) */
    html, body, p, span, label, li, .stMarkdown, [data-testid="stChatMessageContent"] {
        font-family: 'Poppins', sans-serif;
    }

    /* Contenedor tipo tarjeta para el chat */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #ffffff;
        border: 1.5px solid #d7dbd8 !important;
        border-radius: 16px;
        padding: 1rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }

    /* Burbujas de chat: fondo blanco y letra oscura forzados, siempre legible */
    div[data-testid="stChatMessage"] {
        background-color: #ffffff !important;
        border: 1px solid #e0e3e1;
        border-radius: 14px;
        padding: 0.75rem 1rem;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    }
    div[data-testid="stChatMessage"] * {
        color: #1a1a1a !important;
    }
    div[data-testid="stChatMessageAvatarUser"], div[data-testid="stChatMessageAvatarAssistant"] {
        background-color: transparent !important;
        width: 44px !important;
        height: 44px !important;
        font-size: 1.6rem !important;
    }

    /* Caja de texto de entrada */
    div[data-testid="stChatInput"] textarea {
        border: 2px solid var(--verde-externado) !important;
        border-radius: 10px !important;
    }

    /* Botón de enviar */
    button[data-testid="stChatInputSubmitButton"] {
        background-color: var(--verde-externado) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

col_logo1, col_titulo, col_logo2 = st.columns([1, 4, 1])
with col_logo1:
    st.image("logo_externado.png", width=130)
with col_titulo:
    st.markdown(
        """
        <div class="encabezado-uec" style="background:none; box-shadow:none; padding:0.2rem 0;">
            <div>
                <h1 style="color:#0b5e3c;">Universidad Externado de Colombia</h1>
                <p style="color:#555;">Asistente Virtual · Mesa Estratégica de Servicios</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col_logo2:
    st.image("logo_mes.png", width=170)

st.markdown(
    """
    <div class="encabezado-uec">
        <div>
            <h1>👋 ¿En qué te podemos ayudar hoy?</h1>
            <p>Escribe tu solicitud: correo, matrícula, pagos, certificados, carné digital, entre otros.</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


if "historial" not in st.session_state:
    st.session_state.historial = []

chat_box = st.container(border=True)

pregunta = st.chat_input("Escribe tu solicitud aquí...")
if pregunta:
    st.session_state.historial.append(("user", pregunta))
    respuesta = responder(pregunta)
    respuesta = corregir_tildes(respuesta)
    st.session_state.historial.append(("assistant", respuesta))

with chat_box:
    for rol, mensaje in st.session_state.historial:
        avatar = "🧑" if rol == "user" else "🎓"
        with st.chat_message(rol, avatar=avatar):
            st.markdown(mensaje)
