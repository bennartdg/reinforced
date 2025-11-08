from neo4j import GraphDatabase
import pandas as pd
import streamlit as st

# Ganti sesuai dengan konfigurasi Neo4j Anda
uri = st.secrets["NEO4J_URI"]
user = st.secrets["NEO4J_USER"]
password = st.secrets["NEO4J_PASSWORD"]

driver = GraphDatabase.driver(uri, auth=(user, password))

# Fungsi ambil data peneliti
def get_all_person():
    query = """
    MATCH (p:ns0__Person)
    RETURN p.ns0__hasName AS Nama, p.ns0__hasSintaID AS SintaID, p.ns0__hasDepartment AS Departemen
    ORDER BY Nama
    """
    with driver.session() as session:
        result = session.run(query)
        data = [record.data() for record in result]
        return pd.DataFrame(data)
    

def run_query(query):
    with driver.session() as session:
        result = session.run(query)
        return pd.DataFrame([r.data() for r in result])



def ane():
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics.pairwise import cosine_similarity
    import networkx as nx
    import random
    from collections import Counter, defaultdict
    
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.optim as optim

    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Parameter random walk
    parameter_random_walk = 10
    num_walks = parameter_random_walk
    walk_length = parameter_random_walk
    non_local_limit = parameter_random_walk
    
    set_top_k_rekomendasi = 5
    set_hidden_dim = 64
    set_embedding_dim = 40

    query_full_attr_with_name = """
		MATCH (p:ns0__Person)
		RETURN 
			p.ns0__hasSintaID AS sinta_id,
			p.ns0__hasName AS name,
			toInteger(p.ns0__hasAcademicAge) AS AA,
			toInteger(p.ns0__hasCollaborator) AS CO,
			toFloat(p.ns0__hasAverageCitationScholar) AS AvgCiteScholar,
			toFloat(p.ns0__hasAverageCitationScopus) AS AvgCiteScopus,
			toFloat(p.ns0__hasAverageCitationWos) AS AvgCiteWos,
			toInteger(p.ns0__hasHIndexScholar) AS HIndexScholar,
			toInteger(p.ns0__hasHIndexScopus) AS HIndexScopus,
			toInteger(p.ns0__hasHIndexWos) AS HIndexWos,
			toInteger(p.ns0__hasPublicationScholar) AS PubScholar,
			toInteger(p.ns0__hasPublicationScopus) AS PubScopus,
			toInteger(p.ns0__hasPublicationWos) AS PubWos
		"""

    df_attr_named = run_query(query_full_attr_with_name).dropna()

    columns_to_scale = df_attr_named.columns.difference(['sinta_id', 'name'])
    scaler = MinMaxScaler()
    df_scaled = df_attr_named.copy()
    df_scaled[columns_to_scale] = scaler.fit_transform(df_attr_named[columns_to_scale])

    df_scaled.set_index('sinta_id', inplace=True)
    
    # ----------------------------------------------------------------------------------
    # STEP 1: Hitung Kemiripan Atribut (Cosine Similarity)
    # ----------------------------------------------------------------------------------
    
    # Hanya gunakan kolom numerik (tanpa name)
    X = df_scaled.drop(columns=['name'])
    similarity_matrix = cosine_similarity(X)
    
    # Buat DataFrame untuk kemudahan interpretasi
    similarity_df = pd.DataFrame(similarity_matrix, index=df_scaled['name'], columns=df_scaled['name'])
    
    # -------------------------------
    # Ambil Top-2 Tetangga (attr_sim)
    # -------------------------------
    top_k_neighbors = {}
    names = df_scaled['name'].tolist()

    for i, name in enumerate(names):
        scores = similarity_matrix[i].copy()
        scores[i] = -1  # Hindari self
        top_indices = scores.argsort()[::-1][:2]  # Top-2

        # Simpan nama dan nilai cosine similarity-nya
        top_k = [(names[j], round(scores[j], 4)) for j in top_indices]
        top_k_neighbors[name] = top_k
    
    # ----------------------------------------------------------------------------------
    # STEP 2: Menentukan Non-Local Neighbor
    # ----------------------------------------------------------------------------------
    
    # Ambil struktur jaringan (relasi collaborateWith)
    query_edges = """
    MATCH (p1:ns0__Person)-[:collaborateWith]-(p2:ns0__Person)
    RETURN p1.ns0__hasSintaID AS source, p2.ns0__hasSintaID AS target
    """
    df_edges = run_query(query_edges)

    # Bangun graph undirected
    G = nx.Graph()
    G.add_edges_from(df_edges[['source', 'target']].values)
    
    # Simpan hasil random walk untuk setiap node
    non_local_neighbors = defaultdict(list)

    for node in G.nodes:
        walks = []
        for _ in range(num_walks):
            walk = [node]
            current = node
            for _ in range(walk_length):
                neighbors = list(G.neighbors(current))
                if not neighbors:
                    break
                current = random.choice(neighbors)
                walk.append(current)
            walks.extend(walk)

        # Hitung frekuensi kunjungan (exclude self)
        counter = Counter([n for n in walks if n != node])
        top_k = [sid for sid, _ in counter.most_common(non_local_limit)]
        non_local_neighbors[node] = top_k
    
    # Ambil semua person ID dan nama
    query_names = """
    MATCH (p:ns0__Person)
    RETURN p.ns0__hasSintaID AS sinta_id, p.ns0__hasName AS name
    """
    df_names = run_query(query_names).dropna()
    sinta_id_to_name = df_names.set_index('sinta_id')['name'].to_dict()

    # Gabungkan ID SINTA dan nama pada hasil random walk
    non_local_named = {}
    for sid, neighbors in non_local_neighbors.items():
        nama_asal = sinta_id_to_name.get(sid, "❓Unknown")
        pasangan = [f"{nid} ({sinta_id_to_name.get(nid, '❓Unknown')})" for nid in neighbors]
        non_local_named[f"{sid} ({nama_asal})"] = pasangan

    # ----------------------------------------------------------------------------------
    # STEP 3: Menggabungkan Attribute Similarity dan Non-local Neighbors (FUSION)
    # ----------------------------------------------------------------------------------
    # Ambil nama-nama peneliti dari df_scaled
    names = df_scaled['name'].tolist()
    sinta_ids = df_scaled.index.tolist()  # index adalah sinta_id

    # Bangun kembali similarity_matrix
    from sklearn.metrics.pairwise import cosine_similarity
    similarity_matrix = cosine_similarity(df_scaled.drop(columns=['name']))

    # Hitung Top-K attr_sim (misal Top-2)
    attr_sim_neighbors = {}
    for i, sid in enumerate(sinta_ids):
        scores = similarity_matrix[i].copy()
        scores[i] = -1  # hindari diri sendiri
        top_indices = scores.argsort()[::-1][:2]
        top_sinta_ids = [sinta_ids[j] for j in top_indices]
        attr_sim_neighbors[sid] = top_sinta_ids


    multi_graph = nx.Graph()

    # Tambahkan edge attr_sim
    for source, targets in attr_sim_neighbors.items():
        for target in targets:
            if source != target:
                multi_graph.add_edge(source, target, rel='attr_sim')

    # Tambahkan edge non_local_sim
    for source, targets in non_local_neighbors.items():
        for target in targets:
            if source != target:
                multi_graph.add_edge(source, target, rel='non_local_sim')

    # print(f"📦 Jumlah node: {multi_graph.number_of_nodes()}")
    # print(f"🔗 Jumlah edge (multi-relasi): {multi_graph.number_of_edges()}")
    
    # Contoh beberapa edge
    nama_dict = df_scaled['name'].to_dict()

    # ----------------------------------------------------------------------------------
    # AUTOENCODER
    # A. Adjacecy Matrix
    # ----------------------------------------------------------------------------------

    # Buat daftar node (urut berdasarkan sinta_id)
    node_list = list(multi_graph.nodes)
    node_index = {sid: i for i, sid in enumerate(node_list)}

    # Buat adjacency matrix (binary)
    import numpy as np

    A = np.zeros((len(node_list), len(node_list)))
    for u, v in multi_graph.edges:
        i, j = node_index[u], node_index[v]
        A[i][j] = 1
        A[j][i] = 1  # undirected

    # print("📦 Bentuk Adjacency Matrix: ", A.shape)

    # ----------------------------------------------------------------------------------
    # AUTOENCODER
    # B. Attribute Matrix
    # ----------------------------------------------------------------------------------

    # Ambil baris sesuai urutan node_list
    X_attr = df_scaled.loc[node_list].drop(columns=["name"]).values
    # print("📌 Bentuk Attribute Matrix:", X_attr.shape)
    
    # Gabungkan atribut dan struktur jaringan
    # Gabungkan atribut dan struktur jaringan
    alpha = 0.5
    X_input = np.hstack([X_attr, A])  # dimensi jadi (245, 256)

    # Definisikan AutoEncoder
    class ImprovedAutoEncoder(nn.Module):
        def __init__(self, input_dim, hidden_dim=set_hidden_dim, embedding_dim=set_embedding_dim):
            super(ImprovedAutoEncoder, self).__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim, embedding_dim)
            )
            self.decoder = nn.Sequential(
                nn.Linear(embedding_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, input_dim),
                nn.Sigmoid()
            )

        def forward(self, x):
            z = self.encoder(x)
            x_recon = self.decoder(z)
            return x_recon, z

    # Training
    X_tensor = torch.tensor(X_input, dtype=torch.float32)
    model = ImprovedAutoEncoder(input_dim=X_tensor.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    for epoch in range(300):
        model.train()
        optimizer.zero_grad()
        x_recon, z = model(X_tensor)
        loss = criterion(x_recon, X_tensor)
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.6f}")

    # Simpan hasil embedding
    model.eval()
    with torch.no_grad():
        _, embeddings = model(X_tensor)

    # Normalisasi sebelum cosine similarity
    normalized_embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)

    # Buat dict hasil embedding dengan Sinta ID sebagai key
    index_to_sinta = {v: k for k, v in node_index.items()}
    embedding_result = {
        f"{sinta_id}": embeddings[idx].numpy()
        for idx, sinta_id in index_to_sinta.items()
    }

    embedding_result_named = {}
    for idx, sinta_id in index_to_sinta.items():
        nama = nama_dict.get(sinta_id, "Unknown")
        label = f"{nama} (SINTA: {sinta_id})"
        embedding_result_named[label] = embeddings[idx].numpy()
    
    # ---------------------------------------------------------------------
    # Embedding Result
    # ---------------------------------------------------------------------
    
    query_attr = """
    MATCH (p:ns0__Person)
    RETURN 
        p.ns0__hasSintaID AS sinta_id,
        p.ns0__hasName AS name
    """
    # Fungsi untuk menjalankan Cypher dan ambil hasil ke DataFrame

    df_attr_sinta_name = run_query(query_attr)

    index_to_sinta = {v: k for k, v in node_index.items()}
    embedding_result = {}
    # Misalnya df_attr berisi kolom: "sinta_id" dan "name"
    nama_dict = dict(zip(df_attr_sinta_name["sinta_id"].astype(str), df_attr_sinta_name["name"]))

    for i in range(len(embeddings)):
        sinta_id = index_to_sinta[i]
        nama = nama_dict.get(sinta_id, "Unknown")
        label = f"{nama} (SINTA: {sinta_id})"
        embedding_result[label] = embeddings[i].numpy()

    embedding_result
    
    # ---------------------------------------------------------------------
    # Recommendation
    # ---------------------------------------------------------------------
    
    labels = list(embedding_result.keys())
    embedding_matrix = np.array([embedding_result[label] for label in labels])
    cos_sim = cosine_similarity(embedding_matrix)

    top_k = set_top_k_rekomendasi  # Jumlah rekomendasi
    data_rows = []

    for idx, label in enumerate(labels):
        sim_scores = cos_sim[idx].copy()
        sim_scores[idx] = -1  # Hindari self-matching

        top_indices = np.argsort(sim_scores)[::-1][:top_k]
        for j in top_indices:
            data_rows.append({
                "Peneliti": label,
                "Rekomendasi": labels[j],
                "Skor Kemiripan": sim_scores[j]
            })

    df_rekomendasi = pd.DataFrame(data_rows)
    
    return df_rekomendasi

try:
    df_all_recommendation = ane()
    st.success("✅ Data rekomendasi berhasil dimuat.")
except Exception as e:
    df_all_recommendation = None
    st.error(f"❌ Gagal mengambil data rekomendasi: {e}")

def recommender(name):
    df_recommendation = df_all_recommendation.copy()

    # Normalisasi nama
    name = name.strip().lower()

    # Ekstrak kolom nama dan SINTA
    df_recommendation["Nama"] = df_recommendation["Peneliti"].str.extract(r"^(.*?)\s+\(SINTA:", expand=False).str.strip()
    df_recommendation["SINTA_ID"] = df_recommendation["Peneliti"].str.extract(r"SINTA:\s*(\d+)", expand=False)
    df_recommendation["Rekomendasi_Nama"] = df_recommendation["Rekomendasi"].str.extract(r"^(.*?)\s+\(SINTA:", expand=False).str.strip()
    df_recommendation["Rekomendasi_SINTA_ID"] = df_recommendation["Rekomendasi"].str.extract(r"SINTA:\s*(\d+)", expand=False)

    # Filter berdasarkan nama
    filtered_df = df_recommendation[df_recommendation["Nama"].str.lower() == name]

    if filtered_df.empty:
        return pd.DataFrame(columns=[
            "Nama", "SINTA_ID", "Rekomendasi_Nama", "Rekomendasi_SINTA_ID", "Skor Kemiripan"
        ])


    return filtered_df[[
        "Nama", "SINTA_ID", "Rekomendasi_Nama", "Rekomendasi_SINTA_ID", "Skor Kemiripan"
    ]]

import re
from collections import defaultdict

def get_rekomendasi_sinta_id(df):
    rekomendasi_sinta_id = defaultdict(list)

    for _, row in df.iterrows():
        peneliti_str = row['Peneliti']
        rekom_str = row['Rekomendasi']

        # Ekstrak ID dari format (SINTA: 6679316)
        sid_author = re.search(r"SINTA:\s*(\d+)", peneliti_str)
        sid_rekom = re.search(r"SINTA:\s*(\d+)", rekom_str)

        if sid_author and sid_rekom:
            sid1 = sid_author.group(1)
            sid2 = sid_rekom.group(1)
            rekomendasi_sinta_id[sid1].append(sid2)

    return dict(rekomendasi_sinta_id)

def evaluation():
    # ----------------------------------------------------------------------------------
    # EVALUASI
    # ----------------------------------------------------------------------------------

    # Ambil ground truth dari Neo4j
    query = """
    MATCH (a:ns0__Person)-[:collaborateWith]-(b:ns0__Person)
    RETURN DISTINCT a.ns0__hasSintaID AS sid1, b.ns0__hasSintaID AS sid2
    """
    df_ground_truth = run_query(query)
    ground_truth_pairs = set(tuple(sorted([row['sid1'], row['sid2']])) for _, row in df_ground_truth.iterrows())

    # Buat pasangan prediksi dari global df_all_recommendation
    rekomendasi_sinta_id = get_rekomendasi_sinta_id(df_all_recommendation)

    rekomendasi_pairs = set()
    for sid, recs in rekomendasi_sinta_id.items():
        for sid2 in recs:
            pair = tuple(sorted([sid, sid2]))
            rekomendasi_pairs.add(pair)

    # Hitung metrik evaluasi
    tp = len(ground_truth_pairs & rekomendasi_pairs)
    fp = len(rekomendasi_pairs - ground_truth_pairs)
    fn = len(ground_truth_pairs - rekomendasi_pairs)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # # Tampilkan hasil
    # print(f"TP        : {tp}")
    # print(f"FP        : {fp}")
    # print(f"FN        : {fn}")
    # print(f"🎯 Precision : {precision:.4f}")
    # print(f"🎯 Recall    : {recall:.4f}")
    # print(f"🎯 F1 Score  : {f1:.4f}")

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }
