from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, SearchRequest
from collections import defaultdict

app = FastAPI()

@app.get("/")
def root():
    return {"status": "API is running"}

# Qdrant bağlantısı (gerekirse host ve port değiştir)

client = QdrantClient(
    url="https://ae6673f5-67e3-4089-8007-9352def14318.eu-central-1-0.aws.cloud.qdrant.io:6333",
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.DL5R4Jn0WQ4fqAgoSS-mJcziw1VBblz5H56gNLX8j-Q"
)


# İstek JSON modeli
class CategoryRequest(BaseModel):
    platform: str
    category_id: int

@app.post("/api/category_recommend")
def category_recommend(request: CategoryRequest):
    scroll_result = client.scroll(
        collection_name="category_embeddings",
        scroll_filter=Filter(
            must=[
                FieldCondition(key="platform", match=MatchValue(value=request.platform)),
                FieldCondition(key="category_id", match=MatchValue(value=request.category_id))
            ]
        ),
        limit=1,
        with_vectors=True,
        with_payload=True
    )

    if not scroll_result[0]:
        raise HTTPException(status_code=404, detail="Kategori bulunamadı.")

    query_vector = scroll_result[0][0].vector

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

    grouped_results = defaultdict(list)

    for r in responses[0]:
        
        grouped_results[r.payload['platform']].append({
            "platform": r.payload['platform'],
            "id": r.payload.get('category_id'),
            "hierarchy": r.payload.get('hierarchy')
        })

    
    output = []
    for plat in ["Hepsiburada", "Trendyol", "N11"]:
        output.extend(grouped_results.get(plat, []))

    return output
