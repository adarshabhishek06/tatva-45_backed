import os
import shutil
import time

from typing import List

from fastapi import (
    FastAPI,
    File,
    UploadFile,
    HTTPException
)

from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel, validator

from google import genai
from google.genai import types

from rag import (
    build_index,
    search,
    UPLOAD_DIR,
    VECTOR_DIR
)


app = FastAPI(
    title="TATVA RAG API",
    version="1.0"
)

# Gemini client. Set GEMINI_API_KEY in the environment; never hard-code the key.
# Read the Gemini key from the environment. Never hard-code it in source code.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY environment variable is not set.")

gemini_client = genai.Client(api_key=GEMINI_API_KEY)
GEMINI_MODELS = [
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash-lite",
]

# How many explicit retries to make for temporary service errors.
MAX_GEMINI_RETRIES = 2
SUPPORTED_LANGUAGES = (
    "English",
    "Assamese",
    "Bengali",
    "Bodo",
    "Dogri",
    "Gujarati",
    "Hindi",
    "Kannada",
    "Kashmiri",
    "Konkani",
    "Maithili",
    "Malayalam",
    "Manipuri",
    "Marathi",
    "Nepali",
    "Odia",
    "Punjabi",
    "Sanskrit",
    "Santali",
    "Sindhi",
    "Tamil",
    "Telugu",
    "Urdu",
)
LANGUAGE_ALIASES = {
    language.casefold(): language for language in SUPPORTED_LANGUAGES
}


def normalize_language(value: str) -> str:
    if value is None:
        raise ValueError("Language cannot be empty.")

    normalized = str(value).strip()

    if not normalized:
        raise ValueError("Language cannot be empty.")

    lookup = normalized.casefold()

    if lookup in LANGUAGE_ALIASES:
        return LANGUAGE_ALIASES[lookup]

    if normalized in SUPPORTED_LANGUAGES:
        return normalized

    raise ValueError(
        "Unsupported language. Select one of the supported languages."
    )


app.add_middleware(
    CORSMiddleware,

    allow_origins=[
        "*"
    ],

    allow_credentials=True,

    allow_methods=[
        "*"
    ],

    allow_headers=[
        "*"
    ]
)


class ChatRequest(BaseModel):

    question: str

    jurisdiction: str = "india"

    language: str = "English"

    @validator("language", pre=True)
    def validate_language(cls, value):
        return normalize_language(value)


class ClassificationRequest(BaseModel):

    formulation_name: str

    ingredients: str

    intended_use: str


class IPRequest(BaseModel):

    ip_type: str

    jurisdiction: str = "india"


class ABSRequest(BaseModel):

    resource: str


class TKDLRequest(BaseModel):

    keyword: str


@app.get("/")
def root():

    return {
        "system":
            "TATVA",

        "status":
            "running",

        "embedding":
            "BAAI/bge-m3",

        "vector_database":
            "FAISS"
    }


@app.get("/api/status")
def status():

    index_exists = os.path.exists(
        os.path.join(
            VECTOR_DIR,
            "index.faiss"
        )
    )

    return {

        "status":
            "ready"
            if index_exists
            else "waiting",

        "embedding":
            "BGE-M3",

        "vector_database":
            "FAISS",

        "index_exists":
            index_exists

    }


@app.post(
    "/api/documents/process"
)
async def process_documents(
    files: List[UploadFile] = File(...)
):

    pdf_paths = []

    try:

        for file in files:

            filename = file.filename

            if not filename.lower().endswith(
                ".pdf"
            ):

                raise HTTPException(
                    status_code=400,
                    detail=
                        f"{filename} is not a PDF."
                )

            safe_filename = os.path.basename(
                filename
            )

            destination = os.path.join(
                UPLOAD_DIR,
                safe_filename
            )

            with open(
                destination,
                "wb"
            ) as buffer:

                shutil.copyfileobj(
                    file.file,
                    buffer
                )

            pdf_paths.append(
                destination
            )

        result = build_index(
            pdf_paths
        )

        return {
            "success":
                True,

            "documents":
                result["documents"],

            "total_chunks":
                result["total_chunks"],

            "embedding":
                "BAAI/bge-m3",

            "vector_database":
                "FAISS",

            "dimension":
                result["dimension"]
        }

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )


@app.post("/api/chat")
def chat(request: ChatRequest):

    question = request.question.strip()

    if not question:

        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    try:

        results = search(
            question,
            top_k=3
        )

    except Exception as error:

        raise HTTPException(
            status_code=400,
            detail=str(error)
        )

    if not results:

        return {

            "answer":
                "I could not find relevant evidence "
                "in the uploaded knowledge base.",

            "sources": [],

            "confidence": 0.0

        }

    context = "\n\n".join(
        [
            (
                f"Source: {item['document']}, "
                f"Page: {item['page']}\n"
                f"{item['text']}"
            )

            for item in results
        ]
    )

    # Generate a complete, concise, grounded answer with Gemini.
    prompt = f"""You are TATVA, an AI assistant for intellectual-property and regulatory documents.

Your job is to answer the user's question directly from the retrieved evidence.

STRICT RULES:
1. Use ONLY the retrieved evidence. Do not add outside legal or regulatory knowledge.
2. Give the actual answer, not a description of the evidence.
3. If the user asks "what is required", "what are the requirements", or similar, state ALL relevant requirements found in the evidence.
4. Prefer 2-5 short bullet points when the answer contains multiple requirements.
5. Start directly with the answer. Do NOT write phrases such as "The following are..." unless the actual items immediately follow.
6. Every sentence must be complete. NEVER stop after an introductory phrase.
7. Do not reproduce the PDF passage word-for-word.
8. Do not mention FAISS, BGE-M3, retrieval, chunks, context, prompts, or the AI process.
9. If the evidence does not answer the question, reply exactly:
"The uploaded documents do not provide enough information to answer this question."
10. Respond in {request.language}.

User question:
{question}

Retrieved evidence:
{context}

Now provide ONLY the final answer to the user.
"""

    def is_transient_error(error):
        message = str(error).lower()
        return any(
            marker in message
            for marker in (
                "503",
                "unavailable",
                "service unavailable",
                "429",
                "resource_exhausted",
                "500",
                "internal server error",
                "504",
                "deadline exceeded",
            )
        )

    answer = ""
    last_error = None

    # Gemini's SDK already retries transient failures. These explicit retries
    # add a small second layer and then move to another current Gemini model
    # if one model is temporarily overloaded.
    for model_name in GEMINI_MODELS:
        for attempt in range(MAX_GEMINI_RETRIES):
            try:
                response = gemini_client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        max_output_tokens=800,
                        thinking_config=types.ThinkingConfig(
                            thinking_level="low"
                        ),
                    ),
                )

                answer = (response.text or "").strip()

                if answer:
                    break

                last_error = RuntimeError(
                    f"{model_name} returned an empty response."
                )

            except Exception as error:
                last_error = error

                # Do not retry authentication, permission, malformed-request,
                # or other non-transient errors.
                if not is_transient_error(error):
                    raise HTTPException(
                        status_code=502,
                        detail=(
                            "Gemini answer generation failed: "
                            f"{error}"
                        ),
                    )

                # Exponential backoff for temporary 5xx/429 errors.
                if attempt < MAX_GEMINI_RETRIES - 1:
                    time.sleep(2 ** attempt)

        if answer:
            break

    if not answer:
        raise HTTPException(
            status_code=503,
            detail=(
                "Gemini is temporarily unavailable after retries. "
                "Please try the question again in a moment. "
                f"Last error: {last_error}"
            ),
        )

    confidence = calculate_confidence(
        results
    )

    sources = [

        {
            "document":
                item["document"],

            "page":
                item["page"],

            "score":
                item["score"]

        }

        for item in results
    ]

    return {

        "answer":
            answer,

        "sources":
            sources,

        "confidence":
            confidence

    }


@app.post("/api/classify")
def classify(
    request: ClassificationRequest
):

    query = f"""
Formulation:
{request.formulation_name}

Ingredients:
{request.ingredients}

Intended use:
{request.intended_use}

Classify this Ayurvedic formulation using
the available authoritative knowledge base.
Consider classical/proprietary/non-classical
classification and relevant regulatory categories.
"""

    results = search(
        query,
        top_k=5
    )

    context = "\n\n".join(
        item["text"]
        for item in results
    )

    answer = (
        "Preliminary RAG-assisted classification "
        "based on retrieved evidence:\n\n"
        + context
        + "\n\nHuman/legal/regulatory verification "
        "is recommended."
    )

    return {

        "answer":
            answer,

        "sources":
            [
                {
                    "document":
                        item["document"],

                    "page":
                        item["page"],

                    "score":
                        item["score"]
                }

                for item in results
            ]

    }


@app.post("/api/ip-navigator")
def ip_navigator(
    request: IPRequest
):

    query = f"""
Intellectual property type:
{request.ip_type}

Jurisdiction:
{request.jurisdiction}

Explain the applicable intellectual property
protection, requirements, exclusions and
relevant considerations using the authoritative
knowledge base.
"""

    results = search(
        query,
        top_k=5
    )

    context = "\n\n".join(
        item["text"]
        for item in results
    )

    return {

        "answer":
            context,

        "sources":
            [
                {
                    "document":
                        item["document"],

                    "page":
                        item["page"],

                    "score":
                        item["score"]
                }

                for item in results
            ]

    }


@app.post("/api/abs/screen")
def abs_screen(
    request: ABSRequest
):

    results = search(
        request.resource,
        top_k=5
    )

    context = "\n\n".join(
        item["text"]
        for item in results
    )

    return {

        "answer":
            "Preliminary ABS/TK evidence:\n\n"
            + context
            + "\n\nThis is preliminary screening "
              "and not legal advice.",

        "sources":
            [
                {
                    "document":
                        item["document"],

                    "page":
                        item["page"],

                    "score":
                        item["score"]
                }

                for item in results
            ]

    }


@app.post("/api/tkdl/search")
def tkdl_search(
    request: TKDLRequest
):

    results = search(
        request.keyword,
        top_k=8
    )

    return {

        "results":
            results

    }


def calculate_confidence(
    results
):

    if not results:

        return 0.0

    scores = [
        max(
            0,
            min(
                1,
                item["score"]
            )
        )

        for item in results
    ]

    return round(
        sum(scores) / len(scores),
        3
    )