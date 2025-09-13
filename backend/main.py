import os
from io import BytesIO
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PyPDF2 import PdfReader
import google.generativeai as genai
from dotenv import load_dotenv
import re
import logging
import json

load_dotenv()

app = FastAPI()

# Autoriser CORS pour toutes origines (adapter en prod)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration API Google Gemini
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel("gemini-2.0-flash")

def extract_text_from_pdf(file_bytes: bytes) -> str:
    pdf_file = BytesIO(file_bytes)
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

def parse_quiz_text(quiz_text: str) -> list:
    # Sépare les questions par le pattern "**Question X:**"
    questions_raw = re.split(r"\*\*Question\s*\d+\s*:\*\*", quiz_text)
    questions_raw = questions_raw[1:]  # Ignorer tout avant 1ère question

    quiz = []
    for raw in questions_raw:
        try:
            pattern = re.compile(
                r"^(.*?)\n"                 # question
                r"a\)\s*(.*?)\n"
                r"b\)\s*(.*?)\n"
                r"c\)\s*(.*?)\n"
                r"d\)\s*(.*?)\n"
                r"\*\*Réponse:\*\*\s*([a-dA-D])\)?\n"
                r"\*\*Explication:\*\*\s*([\s\S]*)$",  # explication multiline
                re.MULTILINE
            )
            match = pattern.match(raw.strip())
            if not match:
                logging.warning("Question ignorée format incorrect:\n" + raw)
                continue

            question, a, b, c, d, answer_letter, explanation = match.groups()
            options = [a.strip(), b.strip(), c.strip(), d.strip()]
            index = "abcd".index(answer_letter.lower())

            quiz.append({
                "question": question.strip(),
                "options": options,
                "answer": options[index],
                "explanation": explanation.strip() or "Aucune explication fournie."
            })
        except Exception as e:
            logging.warning(f"Erreur parsing question : {e}")
            continue

    return quiz

# --- Variante JSON (plus robuste parsing) ---
import re
import json

def parse_quiz_json(quiz_text: str):
    try:
        # Supprime les balises ```json ... ```
        quiz_text = re.sub(r"^```json|```$", "", quiz_text.strip(), flags=re.MULTILINE).strip()

        # Essaye de trouver un tableau JSON même s'il y a du texte autour
        match = re.search(r"\[.*\]", quiz_text, re.DOTALL)
        if match:
            quiz_text = match.group(0)

        return json.loads(quiz_text)
    except Exception as e:
        logging.error(f"Erreur parsing JSON quiz : {e}")
        return []

@app.post("/generate_quiz")
async def generate_quiz(file: UploadFile = File(...), num_questions: int = Form(...)):
    try:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Fichier vide.")

        text = extract_text_from_pdf(content)
        if not text.strip():
            raise HTTPException(status_code=400, detail="Le PDF ne contient aucun texte exploitable.")

        # --- Prompt JSON ---
        prompt = (
            f"Tu es un assistant pédagogique. Génère un quiz JSON avec EXACTEMENT {num_questions} questions à choix multiple "
            f"(4 options par question). Utilise CE FORMAT JSON STRICT :\n"
            f"[\n"
            f"  {{\n"
            f"    \"question\": \"...\",\n"
            f"    \"options\": [\"option a\", \"option b\", \"option c\", \"option d\"],\n"
            f"    \"answer\": \"option correcte (copiée de la liste options)\",\n"
            f"    \"explanation\": \"explication courte\"\n"
            f"  }}\n"
            f"]\n\n"
            f"Ne réponds que par ce JSON valide, sans introduction, ni texte supplémentaire.\n\n"
            f"Voici le texte source :\n{text}"
        )

        response = model.generate_content(prompt)

        if not response.candidates or not response.candidates[0].content.parts:
            raise HTTPException(status_code=500, detail="Réponse vide ou invalide de Gemini.")

        quiz_text = response.candidates[0].content.parts[0].text.strip()
        logging.info(f"Quiz généré brut:\n{quiz_text}")

        # --- Parsing JSON robuste ---
        quiz = parse_quiz_json(quiz_text)
        if not quiz:
            raise HTTPException(status_code=500, detail="Erreur parsing JSON du quiz généré.")

        return {"quiz": quiz[:num_questions]}

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Erreur dans generate_quiz: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Erreur interne du serveur lors de la génération du quiz.")
