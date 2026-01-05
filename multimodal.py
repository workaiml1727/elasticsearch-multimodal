import logging
import gradio as gr
import os
from datetime import datetime, timedelta, timezone
import torch
import whisper
from ultralytics import YOLO
import numpy as np
import cv2
import pytesseract
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from sentence_transformers import SentenceTransformer
import boto3
from minio import Minio
from elasticsearch import Elasticsearch
from html import escape

# LOGGING
logging.basicConfig(filename="rag.log", level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# MinIO Config
MINIO_ENDPOINT = "127.0.0.1:9000"
MINIO_ACCESS_KEY = "minioadmin"
MINIO_SECRET_KEY = "minioadmin"
s3 = boto3.client("s3", endpoint_url=f"http://{MINIO_ENDPOINT}", aws_access_key_id=MINIO_ACCESS_KEY, aws_secret_access_key=MINIO_SECRET_KEY)
minio_client = Minio(MINIO_ENDPOINT, access_key=MINIO_ACCESS_KEY, secret_key=MINIO_SECRET_KEY, secure=False)

def get_presigned_url(bucket_name, object_name, expires_days=7):
    try:
        return minio_client.presigned_get_object(bucket_name, object_name, expires=timedelta(days=expires_days))
    except Exception as e:
        logger.error(f"Presigned URL failed: {e}")
        return None

# Elasticsearch Config
ES_HOST = "http://127.0.0.1:9200"
es = Elasticsearch(
    hosts=[ES_HOST],
    timeout=300,
    max_retries=30,
    retry_on_timeout=True,
    verify_certs=False
)

# Test connection
try:
    info = es.info()
    logger.info("Elasticsearch connected successfully")
    print("Elasticsearch connected:", info["cluster_name"], info["version"]["number"])
except Exception as e:
    logger.error(f"Elasticsearch connection failed: {e}")
    print(f"Elasticsearch connection failed: {e}")

# Multimodal RAG Index
if not es.indices.exists(index="multimodal_rag"):
    es.indices.create(index="multimodal_rag", mappings={
        "properties": {
            "file_name": {"type": "keyword"},
            "file_type": {"type": "keyword"},  # video, image
            "chunk_id": {"type": "keyword"},
            "chunk_type": {"type": "keyword"},  # audio, caption, ocr, object
            "chunk_text": {"type": "text"},
            "timestamp": {"type": "float"},
            "frame_idx": {"type": "integer"},
            "embedding": {
                "type": "dense_vector",
                "dims": 384,
                "index": True,
                "similarity": "cosine"
            }
        }
    })
    logger.info("Created multimodal_rag index")

# Load Models
device = "cuda" if torch.cuda.is_available() else "cpu"
blip_processor = BlipProcessor.from_pretrained(r"C:\Users\Nikita\Documents\Elastic\Models\BLIP")
blip_model = BlipForConditionalGeneration.from_pretrained(r"C:\Users\Nikita\Documents\Elastic\Models\BLIP").to(device)
yolo_model = YOLO(r"C:\Users\Nikita\Documents\Elastic\Models\Yolo\yolov8n.pt")
whisper_model = whisper.load_model("base", device=device)
embedder = SentenceTransformer('all-MiniLM-L6-v2', device=device)

TEMP_FOLDER = "temp_media"
os.makedirs(TEMP_FOLDER, exist_ok=True)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Simple chunking
def chunk_text(text, max_length=512, overlap=128):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + max_length, len(text))
        chunks.append(text[start:end])
        start += max_length - overlap
    return chunks

# Process & Index File (VIDEO + IMAGE)
def process_and_index_file(bucket_name, file_name):
    temp_path = os.path.join(TEMP_FOLDER, os.path.basename(file_name))
    try:
        minio_client.fget_object(bucket_name, file_name, temp_path)
        file_id = f"file_{hash(file_name)}"
        file_type = "video" if file_name.lower().endswith((".mp4", ".avi", ".mov")) else "image"

        # Metadata
        meta_doc = {
            "file_id": file_id,
            "file_name": file_name,
            "file_type": file_type,
            "upload_date": datetime.now(timezone.utc).isoformat()
        }
        es.index(index="files_meta", id=file_id, document=meta_doc)

        chunks = []

        if file_type == "video":
            # ────────────────────────────────────────────────
            # VIDEO PROCESSING (unchanged - as it was working)
            # ────────────────────────────────────────────────
            cap = cv2.VideoCapture(temp_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            duration_sec = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps)
            cap.release()

            transcript = ""
            try:
                result = whisper_model.transcribe(temp_path, fp16=False)
                transcript = result["text"]
            except Exception as e:
                logger.error(f"Whisper failed: {e}")

            for i, chunk in enumerate(chunk_text(transcript)):
                chunk_id = f"{file_name}#audio#{i}"
                doc = {
                    "file_name": file_name,
                    "file_type": "video",
                    "chunk_id": chunk_id,
                    "chunk_type": "audio",
                    "chunk_text": chunk,
                    "embedding": embedder.encode(chunk).tolist(),
                    "timestamp": i * 20
                }
                chunks.append(doc)

            # Add your video frame/event logic here if needed...

        elif file_type == "image":
            # ────────────────────────────────────────────────
            # IMAGE PROCESSING (new + BLIP captioning included)
            # ────────────────────────────────────────────────
            img = Image.open(temp_path)

            # 1. OCR: text from image
            ocr_text = pytesseract.image_to_string(img).strip()

            # 2. BLIP Captioning
            inputs = blip_processor(img, return_tensors="pt").to(device)
            caption_ids = blip_model.generate(**inputs)[0]
            caption = blip_processor.decode(caption_ids, skip_special_tokens=True)

            # 3. Vision AI: YOLO objects
            results = yolo_model(temp_path)
            objects = list(set(results[0].names[int(cls)] for cls in results[0].boxes.cls))
            objects_str = ", ".join(objects)

            # 4. Combine all text for chunking
            combined_text = f"Caption: {caption}\nOCR Text: {ocr_text}\nDetected Objects: {objects_str}"

            # 5. Clean + Chunk text
            text_chunks = chunk_text(combined_text)

            # 6. Index each chunk
            for i, chunk in enumerate(text_chunks):
                chunk_id = f"{file_name}#image#{i}"
                doc = {
                    "file_name": file_name,
                    "file_type": "image",
                    "chunk_id": chunk_id,
                    "chunk_type": "image",
                    "chunk_text": chunk,
                    "embedding": embedder.encode(chunk).tolist(),
                    "frame_idx": 1  # single image
                }
                es.index(index="multimodal_rag", document=doc)

        return f"{file_type.capitalize()} {file_name} indexed successfully ({len(chunks)} chunks created)."

    except Exception as e:
        logger.error(f"File processing failed: {e}", exc_info=True)
        return f"Error: {str(e)}"

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

# Search (same as before - supports both video & image)
def search_media(query):
    try:
        body = {
            "size": 10,
            "query": {
                "bool": {
                    "should": [
                        {"match": {"chunk_text": {"query": query, "boost": 2.0}}},
                        {"script_score": {
                            "query": {"match_all": {}},
                            "script": {
                                "source": "cosineSimilarity(params.qvec, 'embedding') + 1.0",
                                "params": {"qvec": embedder.encode(query).tolist()}
                            },
                            "boost": 1.5
                        }}
                    ]
                }
            },
            "sort": [{"_score": {"order": "desc"}}]
        }
        res = es.search(index="multimodal_rag", body=body)
        hits = res["hits"]["hits"]
        output = []
        for hit in hits:
            src = hit["_source"]
            score = hit["_score"]
            file_link = get_presigned_url("media", src["file_name"])
            location = f"at ~{src.get('timestamp', 'N/A')}s" if src["file_type"] == "video" else "image"
            line = f"""
            <div style="border:1px solid #ddd; padding:10px; margin:10px 0;">
                <strong>Score: {score:.2f}</strong> | Type: {src['chunk_type']} | {location}<br>
                {escape(src['chunk_text'][:300])}...<br>
                <a href="{file_link}" target="_blank" style="color:blue; font-weight:bold;">Open {src['file_type'].capitalize()} →</a>
            </div>
            """
            output.append(line)
        return "".join(output) if output else "No results found."
    except Exception as e:
        logger.error(f"Search error: {e}")
        return f"Search error: {str(e)}"

# Gradio UI – supports video and image
with gr.Blocks(title="Multimodal Intelligence Pipeline") as demo:
    gr.Markdown("# Multimodal RAG\nVideo + Image Indexing & Search")
    with gr.Tab("Index File"):
        bucket_dd = gr.Dropdown(
            label="Bucket",
            choices=[],
            value=None,
            interactive=True,
            allow_custom_value=False
        )
        file_dd = gr.Dropdown(
            label="File (Video .mp4 or Image .jpg/.png)",
            choices=[],
            value=None,
            interactive=True,
            allow_custom_value=False
        )
        index_btn = gr.Button("Index File", variant="primary")
        status = gr.Textbox(label="Status", lines=5)

        def load_buckets():
            try:
                buckets = [b['Name'] for b in s3.list_buckets()['Buckets']]
                print(f"DEBUG: Loaded buckets: {buckets}")
                return gr.update(choices=buckets, value=None)
            except Exception as e:
                print(f"Bucket load error: {e}")
                return gr.update(choices=[], value=None)

        def load_files(bucket):
            try:
                if not bucket:
                    return gr.update(choices=[], value=None)
                objs = s3.list_objects_v2(Bucket=bucket).get('Contents', [])
                files = [o['Key'] for o in objs if o['Key'].lower().endswith(('.mp4', '.jpg', '.png', '.jpeg'))]
                print(f"DEBUG: Loaded {len(files)} files from {bucket}")
                return gr.update(choices=files, value=None)
            except Exception as e:
                print(f"File load error for {bucket}: {e}")
                return gr.update(choices=[], value=None)

        demo.load(load_buckets, outputs=bucket_dd)
        bucket_dd.change(load_files, bucket_dd, file_dd)
        index_btn.click(process_and_index_file, [bucket_dd, file_dd], status)

    with gr.Tab("Search"):
        q = gr.Textbox(label="Search anything", lines=2, placeholder="e.g. black tshirt man, ship radar, text in image")
        btn = gr.Button("Search")
        res = gr.HTML()
        btn.click(search_media, q, res)
        q.submit(search_media, q, res)

    demo.launch(server_name="127.0.0.1", server_port=7862, share=False)