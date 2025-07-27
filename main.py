from fastapi import FastAPI, HTTPException
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, SearchRequest
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# CORS ayarları
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Sağlık kontrol endpoint'i
@app.get("/")
def root():
    return {"status": "API is running"}

# Qdrant bağlantısı
client = QdrantClient(
    url="https://ae6673f5-67e3-4089-8007-9352def14318.eu-central-1-0.aws.cloud.qdrant.io:6333",
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.DL5R4Jn0WQ4fqAgoSS-mJcziw1VBblz5H56gNLX8j-Q"
)

# Kategori öneri endpoint'i
@app.get("/api/category_recommend")
def category_recommend(platform: str, category_id: int):
    # İstenen platform ve category_id'den vektör çek
    scroll_result = client.scroll(
        collection_name="category_embeddings",
        scroll_filter=Filter(
            must=[
                FieldCondition(key="platform", match=MatchValue(value=platform)),
                FieldCondition(key="category_id", match=MatchValue(value=category_id))
            ]
        ),
        limit=1,
        with_vectors=True,
        with_payload=True
    )

    if not scroll_result[0]:
        raise HTTPException(status_code=404, detail="Kategori bulunamadı.")

    query_vector = scroll_result[0][0].vector

    # Top 20 benzer sonucu getir
    responses = client.search_batch(
        collection_name="category_embeddings",
        requests=[
            SearchRequest(
                vector=query_vector,
                limit=20,
                with_payload=True
            )
        ]
    )

    # En iyi eşleşmeleri tutmak için
    best_matches = {}

    for r in responses[0]:
        other_platform = r.payload.get('platform')
        #similarity = round(1 - r.score, 4)
        similarity = round(r.score*100, 2)

        # Aynı platformdan gelenleri atla
        if other_platform == platform:
            continue

        # Her diğer platformdan sadece en iyi eşleşme
        if (other_platform not in best_matches) or (similarity > best_matches[other_platform]['score']):
            best_matches[other_platform] = {
                "platform": other_platform,
                "id": r.payload.get("category_id"),
                "hierarchy": r.payload.get("hierarchy"),
                "score": similarity
            }

    # Sonuçları liste halinde döndür
    return list(best_matches.values())
