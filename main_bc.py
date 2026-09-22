import os
import shutil

from typing import List

from fastapi import (
    FastAPI,
    File,
    UploadFile,
    HTTPException
)

from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel

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
            top_k=5
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

    # ------------------------------------------------
    # CURRENT VERSION:
    # Retrieval is real.
    # LLM generation is deliberately isolated here.
    # ------------------------------------------------

    answer = (
        "The FAISS retrieval engine found the following "
        "relevant evidence in the uploaded knowledge base.\n\n"
        + context
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