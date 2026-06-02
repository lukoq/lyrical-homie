from pathlib import Path

import chromadb
import umap
import pandas as pd
import plotly.express as px
import numpy as np


BASE_DIR = Path(__file__).parent
CHROMA_DB_PATH = BASE_DIR.parent / "vector_db"
client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
collection = client.get_collection(name="polish_rap_lyrics")


MAX_POINTS = 20000

results = collection.peek(limit=MAX_POINTS)

embeddings = np.array(results["embeddings"])
documents = results["documents"]
metadatas = results["metadatas"]

if len(embeddings) > MAX_POINTS:
    indices = np.random.choice(len(embeddings), MAX_POINTS, replace=False)
    embeddings = embeddings[indices]
    documents = [documents[i] for i in indices] if documents else None
    metadatas = [metadatas[i] for i in indices] if metadatas else None

embeddings = embeddings + np.random.normal(0, 1e-5, embeddings.shape)
reducer = umap.UMAP(
    n_neighbors=15,
    min_dist=0.1,
    metric='cosine',
    random_state=42)
embeddings_2d = reducer.fit_transform(embeddings)

df = pd.DataFrame(embeddings_2d, columns=['x', 'y'])

if metadatas:
    df['artist'] = [m.get('artist', 'Nieznany') for m in metadatas]
    df['title'] = [m.get('title', 'Brak tytułu') for m in metadatas]
    df['tags'] = [m.get('tags', '') for m in metadatas]

    df['text_preview'] = [doc[:150] + "..." if len(doc) > 150 else doc for doc in documents]

    df['hover_info'] = "<b>Artist:</b> " + df['artist'] + "<br>" + \
                       "<b>Title:</b> " + df['title'] + "<br>" + \
                       "<b>Tags:</b> " + df['tags'] + "<br>" + \
                       "<b>Lyrics:</b> " + df['text_preview']


    df['category'] = df['artist']
else:
    df['hover_info'] = "Brak metadanych"
    df['category'] = 'Wszystkie dane'

fig = px.scatter(
    df, x='x', y='y',
    color='category',
    hover_name='title',
    hover_data={'x': False, 'y': False, 'hover_info': True, 'category': False},
    title="Wizualizacja przestrzeni semantycznej UMAP",
    labels={'x': 'UMAP 1', 'y': 'UMAP 2'},
    opacity=0.6
)
fig.update_traces(marker=dict(size=4))
fig.update_layout(template="plotly_white")

fig.write_html("visualization_chromadb_rag.html")
fig.show()