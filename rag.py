import os
import pickle
import re

import faiss
import fitz
import numpy as np

from sentence_transformers import SentenceTransformer


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
VECTOR_DIR = os.path.join(BASE_DIR, "vectorstore")

INDEX_FILE = os.path.join(
    VECTOR_DIR,
    "index.faiss"
)

METADATA_FILE = os.path.join(
    VECTOR_DIR,
    "metadata.pkl"
)

MODEL_NAME = "BAAI/bge-m3"


os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(VECTOR_DIR, exist_ok=True)


print("Loading BGE-M3 embedding model...")

embedding_model = SentenceTransformer(
    MODEL_NAME
)

print("BGE-M3 loaded.")


def clean_text(text):
    # Normalize whitespace while preserving readable paragraph boundaries.
    text = re.sub(r"[\t\r]+", " ", text)
    text = re.sub(r" +", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_text(text, chunk_size=1200, overlap=200):
    """
    Split PDF text while keeping named sections together when possible.
    This is better for regulatory documents than cutting every N characters.
    """
    text = clean_text(text)
    if not text:
        return []

    # Detect headings such as A., B., C. ... and keep each section together.
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    sections = []
    current = []

    for line in lines:
        if re.match(r"^[A-Z]\.\s+", line) and current:
            sections.append("\n".join(current))
            current = [line]
        else:
            current.append(line)

    if current:
        sections.append("\n".join(current))

    # If no meaningful headings were found, use the original sliding-window strategy.
    if len(sections) <= 1 and len(text) > chunk_size:
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end >= len(text):
                break
            start = end - overlap
        return chunks

    chunks = []
    for section in sections:
        if len(section) <= chunk_size:
            chunks.append(section.strip())
            continue

        # Large sections are split into smaller overlapping pieces.
        start = 0
        while start < len(section):
            end = start + chunk_size
            chunk = section[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end >= len(section):
                break
            start = end - overlap

    return chunks


def extract_pdf(pdf_path):

    document = fitz.open(pdf_path)

    chunks = []

    for page_number in range(
        len(document)
    ):

        page = document[page_number]

        text = page.get_text(
            "text"
        )

        if not text.strip():
            continue

        page_chunks = split_text(text)

        for chunk in page_chunks:

            chunks.append({

                "text": chunk,

                "document":
                    os.path.basename(
                        pdf_path
                    ),

                "page":
                    page_number + 1

            })

    document.close()

    return chunks


def build_index(pdf_paths):

    all_chunks = []

    for pdf_path in pdf_paths:

        print(
            f"Processing: {pdf_path}"
        )

        chunks = extract_pdf(
            pdf_path
        )

        all_chunks.extend(
            chunks
        )

    if not all_chunks:

        raise ValueError(
            "No text could be extracted from PDFs."
        )

    texts = [
        item["text"]
        for item in all_chunks
    ]

    print(
        f"Creating embeddings for {len(texts)} chunks..."
    )

    embeddings = embedding_model.encode(
        texts,
        batch_size=16,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    embeddings = np.asarray(
        embeddings,
        dtype="float32"
    )

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(
        dimension
    )

    index.add(
        embeddings
    )

    faiss.write_index(
        index,
        INDEX_FILE
    )

    with open(
        METADATA_FILE,
        "wb"
    ) as file:

        pickle.dump(
            all_chunks,
            file
        )

    return {
        "documents":
            list(
                set(
                    item["document"]
                    for item in all_chunks
                )
            ),

        "total_chunks":
            len(all_chunks),

        "dimension":
            dimension
    }


def load_index():

    if not os.path.exists(
        INDEX_FILE
    ):

        raise FileNotFoundError(
            "FAISS index does not exist. "
            "Please process PDFs first."
        )

    if not os.path.exists(
        METADATA_FILE
    ):

        raise FileNotFoundError(
            "Metadata file does not exist."
        )

    index = faiss.read_index(
        INDEX_FILE
    )

    with open(
        METADATA_FILE,
        "rb"
    ) as file:

        metadata = pickle.load(
            file
        )

    return index, metadata


def search(
    query,
    top_k=5
):

    index, metadata = load_index()

    query_embedding = embedding_model.encode(
        [query],
        normalize_embeddings=True
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype="float32"
    )

    scores, indices = index.search(
        query_embedding,
        top_k
    )

    results = []

    for score, idx in zip(
        scores[0],
        indices[0]
    ):

        if idx < 0:
            continue

        item = metadata[idx].copy()

        item["score"] = float(
            score
        )

        results.append(
            item
        )

    return results