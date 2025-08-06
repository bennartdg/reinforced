import streamlit.components.v1 as components
from pyvis.network import Network
from ane import run_query
import streamlit as st
from collections import defaultdict
import pandas as pd


def show_collaboration_graph():
    try:
        query = """
        MATCH (p1:ns0__Person)-[:collaborateWith]->(p2:ns0__Person)
        RETURN p1, p2
        """
        df_result = run_query(query)

        if df_result.empty:
            st.warning("Tidak ada data kolaborasi ditemukan.")
            return

        net = Network(height="600px", width="100%", bgcolor="#0f1117", font_color="white")
        net.force_atlas_2based()

        added_nodes = set()

        for _, row in df_result.iterrows():
            p1 = row["p1"]
            p2 = row["p2"]

            name1 = p1.get("ns0__hasName", "Unknown 1")
            id1 = p1.get("ns0__hasSintaID", "id1")

            name2 = p2.get("ns0__hasName", "Unknown 2")
            id2 = p2.get("ns0__hasSintaID", "id2")

            label1 = f"{name1}\n(SINTA: {id1})"
            label2 = f"{name2}\n(SINTA: {id2})"

            if id1 not in added_nodes:
                net.add_node(id1, label=label1, title=label1, color="#3794ff")
                added_nodes.add(id1)

            if id2 not in added_nodes:
                net.add_node(id2, label=label2, title=label2, color="#3794ff")
                added_nodes.add(id2)

            net.add_edge(id1, id2, title="collaborateWith", color="gray")

        net.save_graph("graph_collaboration.html")
        with open("graph_collaboration.html", "r", encoding="utf-8") as f:
            html = f.read()
            components.html(html, height=650, scrolling=True)

    except Exception as e:
        st.error(f"Gagal menampilkan graf kolaborasi: {e}")
        


def visualize_recommendation_paths(target_name, rekom_names):
    # Bangun query UNION dari semua rekomendasi
    print(rekom_names)
    query_blocks = []
    for nama_rekom in rekom_names:
        block = f"""
  MATCH path = shortestPath(
    (p1:ns0__Person {{ns0__hasName: '{target_name}'}})-[:collaborateWith*1..10]-
    (p2:ns0__Person {{ns0__hasName: '{nama_rekom}'}})
  )
  RETURN nodes(path) AS nodes, relationships(path) AS rels
  """
        query_blocks.append(block)

    full_query = "\nUNION\n".join(query_blocks)

    result = run_query(full_query)
    
    if result.empty:
        fallback_query = f"""
        MATCH (target:ns0__Person {{ns0__hasName: '{target_name}'}})
        WITH target
        MATCH (rekom:ns0__Person)
        WHERE rekom.ns0__hasName IN {rekom_names}
        RETURN collect(DISTINCT target) AS target_nodes, collect(DISTINCT rekom) AS rekom_nodes
        """
        result = run_query(fallback_query)

        if not result.empty:
            target_nodes = result.iloc[0]["target_nodes"]  # list of dicts
            rekom_nodes = result.iloc[0]["rekom_nodes"]    # list of dicts

            nodes = target_nodes + rekom_nodes

            # Buat relasi programatikal: target → setiap rekomendasi
            rels = []
            for rekom_node in rekom_nodes:
                rels.append((target_nodes[0], "recommended", rekom_node))

            # Ganti struktur result menjadi DataFrame dengan kolom 'nodes' dan 'rels'
            result = pd.DataFrame([{
                "nodes": nodes,
                "rels": rels
            }])

    net = Network(height="650px", bgcolor="#0d1117", font_color="white")
    net.barnes_hut()
    added_nodes = set()
    
    for idx, row in result.iterrows():
        nodes = row["nodes"]
        rels = row["rels"]

        # Tambahkan node
        for node in nodes:
            label = node.get("ns0__hasName", "Unknown")
            node_id = label  # Gunakan nama sebagai ID
            # 🔷 Tentukan warna berdasarkan jenis node
            if label == target_name:
                color = "#1f77b4"  # Biru → target utama
            elif label in rekom_names:
                color = "#ff9800"  # Oranye → hasil rekomendasi
            else:
                color = "#4caf50"  # Hijau → penghubung biasa
            
            if node_id not in added_nodes:
                net.add_node(node_id, label=label, title=label, color=color)
                added_nodes.add(node_id)
                
        edge_map = defaultdict(set)
        # Tambahkan edge dari relasi
        for rel in rels:
            if isinstance(rel, tuple) and len(rel) == 3:
                source_node, rel_type, target_node = rel
                source_label = source_node.get("ns0__hasName", "Unknown")
                target_label = target_node.get("ns0__hasName", "Unknown")
                edge_map[(source_label, target_label)].add(rel_type)

        # Tambah relasi tambahan dari rekomendasi
        for rekom_name in rekom_names:
            if rekom_name != target_name and rekom_name in added_nodes:
                edge_map[(target_name, rekom_name)].add("recommended")

        # Tambahkan ke graf
        for (src, tgt), rel_types in edge_map.items():
            # Gabungkan label
            label = ", ".join(rel_types)
            color = "orange" if "recommended" in rel_types else "white"
            net.add_edge(src, tgt, label=label, title=label, color=color, width=2 if "recommended" in rel_types else 1)
              
    
    net.save_graph("graph_recommendation_path.html")
    with open("graph_recommendation_path.html", "r", encoding="utf-8") as f:
        html = f.read()
        st.components.v1.html(html, height=600, scrolling=False)
        
    st.markdown("""
    <ul style="list-style: none; padding-left: 0;">
      <li>
        <span style="display: inline-block; width: 12px; height: 12px; background-color: #1f77b4; margin-right: 10px;"></span>
        <strong>Peneliti Target</strong>
      </li>
      <li>
        <span style="display: inline-block; width: 12px; height: 12px; background-color: #ff7f0e; margin-right: 10px;"></span>
        <strong>Peneliti Rekomendasi</strong>
      </li>
      <li>
        <span style="display: inline-block; width: 12px; height: 12px; background-color: #2ca02c; margin-right: 10px;"></span>
        <strong>Peneliti Penghubung (bukan rekomendasi)</strong>
      </li>
    </ul>

    <ul style="list-style: none; padding-left: 0;">
      <li>
        <span style="display: inline-block; width: 12px; height: 2px; background-color: white; margin-right: 10px;"></span>
        <strong>Relasi collaborateWith (pernah berkolaborasi)</strong>
      </li>
      <li>
        <span style="display: inline-block; width: 12px; height: 2px; background-color: orange; margin-right: 10px;"></span>
        <strong>Relasi recommended (direkomendasikan)</strong>
      </li>
    </ul>
    <hr>
    """, unsafe_allow_html=True)
    