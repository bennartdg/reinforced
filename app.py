import streamlit as st
import pandas as pd
import os
import glob
from datetime import datetime
from ane import get_all_person
from ane import recommender
from ane import evaluation
from ane import run_query
from graph import show_collaboration_graph
from graph import visualize_recommendation_paths


st.set_page_config(page_icon="🚀")

def publikasi():
    return 0

# Inisialisasi session_state
if "page" not in st.session_state:
    st.session_state.page = "home"

# Fungsi navigasi
def go_to_recommendation():
    st.session_state.page = "recommendation"
    st.rerun()

def go_to_daftar_peneliti():
    st.session_state.page = "daftar_peneliti"
    st.rerun()

def go_to_hasil_evaluasi():
    st.session_state.page = "hasil_evaluasi"
    st.rerun()

def go_to_home():
    st.session_state.page = "home"
    st.rerun()

try :
    df_person = get_all_person() 
    list_nama_peneliti = df_person["Nama"].sort_values().tolist()
    list_nama_peneliti.insert(0, "")
except Exception as e:
    st.error(f"Gagal mengambil data dari Neo4j: {e}")

@st.dialog("📚 Daftar Publikasi", width='large')
def tampilkan_publikasi(sinta_id, nama):
    st.markdown(f"### {nama}")

    try:
        query = f"""
        MATCH (p:ns0__Person {{ns0__hasSintaID: '{sinta_id}'}})-[:hasPublished]->(pub:ns0__Publication)
        RETURN 
            pub.ns0__hasTitle AS Judul,
            pub.ns0__hasYear AS Tahun,
            pub.ns0__hasDoi AS DOI,
            pub.ns0__hasSourceType AS Sumber
        ORDER BY Tahun DESC
        """
        df_pub = run_query(query)

        if not df_pub.empty:
            st.dataframe(df_pub, use_container_width=True, hide_index=True)
        else:
            st.info("📭 Tidak ada publikasi ditemukan.")

    except Exception as e:
        st.error(f"Gagal memuat publikasi: {e}")

# Halaman 1: Home
if st.session_state.page == "home":
    st.set_page_config(page_title="REINFORCED", layout="centered")    
    
    # ? Sidebar
    with st.sidebar:
        name = st.selectbox("🧑🏻‍💼 Cari Nama Peneliti", list_nama_peneliti, placeholder="Ketik atau Pilih")
        if st.button("🔍 Temukan Rekomendasi"):
            if name:
                try:
                    st.session_state.selected_name = name
                    st.session_state.recommendation_result = recommender(name)
                    st.session_state.page = "recommendation"
                    st.rerun()
                except Exception as e:
                    st.error(f"Gagal menjalankan rekomendasi: {e}")
            else:
                st.warning("⚠️ Silakan masukkan nama terlebih dahulu.")
        if st.button("📊 Hasil Evaluasi"):
            go_to_hasil_evaluasi()
    # ? Sidebar End
    
    st.markdown("""
        <div style="color:#00aaff; font-weight:bold; font-size: 120px; line-height: 0.8; margin-bottom: 10px;">
            REINFORCED
        </div>
        <div style="padding: 0; margin: 0;">
            <div style="font-size:20px; line-height: 1.4;">
                ATT<span style="color:#00aaff; font-weight:bold;">R</span>IBUTED N<span style="color:#00aaff; font-weight:bold;">E</span>TWORK EMBEDD<span style="color:#00aaff; font-weight:bold;">I</span>NG OF KNOWLEDGE GRAPH AND ONTOLOGY BASED
                <span style="color:#00aaff; font-weight:bold;">FO</span>R <span style="color:#00aaff; font-weight:bold;">R</span>ESEARCH <span style="color:#00aaff; font-weight:bold;">C</span>OLLABORATOR R<span style="color:#00aaff; font-weight:bold;">E</span>COMEN<span style="color:#00aaff; font-weight:bold;">D</span>ATION
            </div>
        </div>
""", unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: justify'>
    <b>REINFORCED</b> merupakan penelitian yang bertujuan menemukan rekomendasi kolaborasi untuk melakukan penelitian. 
    Penelitian ini mengimplementasi <b>Attributed Network Embedding</b> sebagai model rekomendasi (Du & Li, 2022). 
    Model rekomendasi dibangun menggunakan data berbasis graf <b>(Graph Base)</b> yang dibangun menggunakan <b>Ontologi</b>.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📘 Langkah Penilaian")
    st.markdown("""
1. Ketik **Nama Lengkap** Anda pada sisi kiri tampilan (Sidebar) hingga saran nama Anda muncul pada kolom **Cari Nama Peneliti**.  
(Apabila nama anda tidak ditemukan, harap diabaikan dan tidak perlu melanjutkan penilaian).
2. Pilih nama Anda yang muncul pada saran yang diberikan.
3. Tekan tombol **Temukan Rekomendasi**, lalu tunggu hingga tampilan selesai memuat.
4. Gulir tampilan ke **Bawah** hingga menunjukkan daftar Rekomendasi Peneliti untuk Anda.
5. Berikan penilaian dengan memilih nilai 5-1 (Sangat setuju - Sangat tidak setuju) untuk setiap Rekomendasi yang diberikan.
6. Isi komentar anda terhadap Rekomendasi yang diberikan dapat mengikuti contoh yang ditampilkan.
7. Tekan tombol **Simpan Penilaian**, sistem akan memberikan tanda apabila anda sudah selesai memberikan penilaian.
8. Proses pemberian penilaian selesai dilakukan, **Terima kasih banyak** sudah mengikuti langkah ini dengan baik.

***Steadfast with the REINFORCED Mind!***
""")
    
    # Tombol untuk pindah halaman
    
    if st.button("🧑🏻‍💼 Daftar Peneliti"):
        go_to_daftar_peneliti()
    
    st.markdown("""
    Tim:<br>
    Bennart Dem Gunawan<sup>1</sup> 🔗 [@bennartdg](https://www.instagram.com/bennartdg/)  
    Kurnia Ramadhan Putra<sup>2</sup> 🔗 [@kramadhanputra](https://www.instagram.com/kramadhanputra/)
    """,unsafe_allow_html=True)
# Halaman 2: Daftar Peneliti
elif st.session_state.page == "daftar_peneliti":
    st.set_page_config(page_title="Daftar Peneliti | REINFORCED", layout="wide")
    # ? Sidebar
    with st.sidebar:
        name = st.selectbox("🧑🏻‍💼 Cari Nama Peneliti", list_nama_peneliti, placeholder="Ketik atau Pilih")
        if st.button("🔍 Temukan Rekomendasi"):
            if name:
                try:
                    st.session_state.selected_name = name
                    st.session_state.recommendation_result = recommender(name)
                    st.session_state.page = "recommendation"
                    st.rerun()
                except Exception as e:
                    st.error(f"Gagal menjalankan rekomendasi: {e}")
            else:
                st.warning("⚠️ Silakan masukkan nama terlebih dahulu.")
        if st.button("📊 Hasil Evaluasi"):
            go_to_hasil_evaluasi()
        
        if st.button("🔙 Kembali"):
            go_to_home()
    # ? Sidebar End
        
    # !!Main
    try:
        st.markdown("### 🕸️ Visualisasi Kolaborasi Peneliti")
        show_collaboration_graph()
        
        st.divider()
        
        st.markdown(f"""### 📋 Daftar Peneliti Tersedia ({df_person.shape[0]} peneliti)""")
        st.dataframe(df_person, use_container_width=True, hide_index=True)
    except Exception as e:
        st.error(f"Gagal mengambil data dari Neo4j: {e}")
    # !!Main End
# Halaman 3: Rekomendasi
elif st.session_state.page == "recommendation":
    st.set_page_config(page_title=f"""{st.session_state.get("selected_name", "").strip().upper()} | REINFORCED""", layout="wide")
    # Sidebar
    with st.sidebar:
        name = st.selectbox("🧑🏻‍💼 Cari Nama Peneliti", list_nama_peneliti, placeholder="Ketik atau Pilih")
        if st.button("🔍 Temukan Rekomendasi"):
            if name:
                try:
                    st.session_state.selected_name = name
                    st.session_state.recommendation_result = recommender(name)
                    st.rerun()
                except Exception as e:
                    st.error(f"Gagal menjalankan rekomendasi: {e}")
            else:
                st.warning("⚠️ Silakan masukkan nama terlebih dahulu.")
        
        if st.button("📊 Hasil Evaluasi"):
            go_to_hasil_evaluasi()
        
        if st.button("🔙 Kembali"):
            go_to_home()

    hasil = st.session_state.get('recommendation_result', None)
    
    if isinstance(hasil, pd.DataFrame) and not hasil.empty:
        # Ambil top-k teratas dari hasil rekomendasi
        # !! parameter
        top_k = 5
        top_k_rekom = hasil.head(top_k) 
        
        rekom_names = top_k_rekom["Rekomendasi_Nama"].tolist()
        target_name = st.session_state.get("selected_name", "").strip().upper()

        # Tampilkan visualisasi
        st.markdown(f"### 🕸️ Visualisasi Jaringan Rekomendasi untuk '{st.session_state.get('selected_name', '...').strip().upper()}'")
        visualize_recommendation_paths(target_name, rekom_names)
        
        st.markdown(f"### 🎯 Daftar Rekomendasi untuk '{st.session_state.get('selected_name', '...').strip().upper()}'")
        st.dataframe(hasil, hide_index=True)
        
        # Simpan penilaian pengguna
        penilaian_user = {}
        
        st.divider()
        
        st.markdown(f"### 📝 Penilaian Rekomendasi untuk '{name.strip().upper()}'")

        # Cek apakah nama peneliti baru berbeda dari sebelumnya
        current_name = name.strip().lower()
        prev_name = st.session_state.get("prev_searched_name", "").lower()

        # Jalankan query hanya jika nama berubah
        if current_name != prev_name:
            try:
                target_query = f"""
                MATCH (p:ns0__Person)
                WHERE toLower(p.ns0__hasName) CONTAINS '{name.lower()}'
                RETURN p LIMIT 1
                """
                df_target = run_query(target_query)

                if not df_target.empty:
                    st.session_state.atribut_target = df_target.iloc[0]["p"]
                else:
                    st.session_state.atribut_target = {}
            except Exception as e:
                st.error(f"Gagal mengambil data peneliti target: {e}")
                st.stop()

        penilaian_keys = []
        # Loop tiap baris dan tampilkan data + penilaian
        for i, (_, row) in enumerate(top_k_rekom.iterrows(), start=1):
            rekom_nama = row['Rekomendasi_Nama']
            rekom_sinta = row['Rekomendasi_SINTA_ID']
            target_nama = st.session_state.atribut_target.get("ns0__hasName", "Peneliti Target")

            st.markdown(f"**{i}. {rekom_nama}** (SINTA: {rekom_sinta})")

            # Ambil atribut peneliti rekomendasi
            try:
                query = f"""
                MATCH (p:ns0__Person {{ns0__hasSintaID: '{rekom_sinta}'}})
                RETURN p
                """
                df_rekom = run_query(query)

                if not df_rekom.empty:
                    data_target = st.session_state.atribut_target
                    data_rekom = df_rekom.iloc[0]["p"]

                    # Gabungkan dan transposisi
                    df_compare = pd.DataFrame({
                        target_nama: {k: str(v) for k, v in data_target.items()},
                        rekom_nama: {k: str(v) for k, v in data_rekom.items()}
                    })
                    
                    df_compare.index = (
                        df_compare.index
                        .str.replace("ns0__has", "", regex=False)
                        .str.replace("_", " ", regex=False)
                        .str.title()
                    )
                    
                    st.dataframe(df_compare, use_container_width=True, hide_index=False)
                else:
                    st.info("Atribut tidak ditemukan.")
            except Exception as e:
                st.error(f"Gagal mengambil data untuk {rekom_nama}: {e}")

            key_radio = f"penilaian_{i}"
            penilaian_keys.append((rekom_nama, key_radio))

            if st.button(f"📚 Daftar Publikasi {rekom_nama}"):
                tampilkan_publikasi(rekom_sinta, rekom_nama)
            
            # Penilaian
            st.radio(
                f"Nilai rekomendasi untuk {rekom_nama}",
                options=[5, 4, 3, 2, 1],
                format_func=lambda x: {
                    5: "5 - Sangat Setuju",
                    4: "4 - Setuju",
                    3: "3 - Netral",
                    2: "2 - Tidak Setuju",
                    1: "1 - Sangat Tidak Setuju"
                }[x],
                key=key_radio
            )
            
            st.markdown("<br>", unsafe_allow_html=True)

        # Text area untuk komentar umum
        komentar = st.text_area("📝 Komentar Anda terhadap Rekomendasi", placeholder="Beri komentar jika ada hal yang menurut Anda kurang sesuai atau sudah sesuai dari hasil rekomendasi ini.")

        # Tombol simpan
        if st.button("💾 Simpan Penilaian"):
            penilaian_user = {}

            for nama, key in penilaian_keys:
                nilai = st.session_state.get(key)
                if nilai is not None:
                    penilaian_user[nama] = nilai

            if not penilaian_user:
                st.warning("⚠️ Tidak ada penilaian yang diberikan.")
            else:
                st.success("📊 Penilaian telah disimpan.")
                st.write("Hasil penilaian Anda:")
                st.json(penilaian_user)

                # Simpan ke CSV
                df_penilaian = pd.DataFrame([
                    {
                        "Nama Rekomendasi": nama,
                        "Nilai": nilai,
                        "Komentar": komentar.strip() if komentar else ""  # satu komentar yang berlaku untuk semua
                    }
                    for nama, nilai in penilaian_user.items()
                ])

                from datetime import datetime
                import os

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                nama_file = f"penilaian_{st.session_state.get('selected_name', 'unknown')}_{timestamp}.csv"

                folder = "penilaian"
                os.makedirs(folder, exist_ok=True)
                df_penilaian.to_csv(os.path.join(folder, nama_file), index=False)

                st.success(f"📁 Penilaian disimpan sebagai `{nama_file}` di folder `{folder}/`")
    else:
        st.info("📭 Tidak ditemukan rekomendasi atau data kosong.")
# Halaman 4: Hasil Evaluasi
elif st.session_state.page == "hasil_evaluasi":
    st.set_page_config(page_title="Hasil Evaluasi | REINFORCED", layout="wide")
    # ? Sidebar
    with st.sidebar:
        name = st.selectbox("🧑🏻‍💼 Cari Nama Peneliti", list_nama_peneliti, placeholder="Ketik atau Pilih")
        if st.button("🔍 Temukan Rekomendasi"):
            if name:
                try:
                    st.session_state.selected_name = name
                    st.session_state.recommendation_result = recommender(name)
                    st.session_state.page = "recommendation"
                    st.rerun()
                except Exception as e:
                    st.error(f"Gagal menjalankan rekomendasi: {e}")
            else:
                st.warning("⚠️ Silakan masukkan nama terlebih dahulu.")
        if st.button("🔙 Kembali"):
            go_to_home()
    # ? Sidebar End
    
    # ! MAIN
    # Evaluasi
    # eval_result = evaluation()
    # top_k = 5

    # st.markdown(f"""### 📊 Evaluasi Model Rekomendasi Top-{top_k}""")
    # st.markdown(
    #     f"""
    #     **True Positive:** {eval_result['tp']} (Direkomendasikan dan sudah pernah berkolaborasi)<br>
    #     **False Positive:** {eval_result['fp']} (Direkomendasikan tetapi belum pernah berkolaborasi)<br>
    #     **False Negative:** {eval_result['fn']} (Tidak direkomendasikan tetapi sudah pernah berkolaborasi)
    #     """,
    #     unsafe_allow_html=True
    # )
    # st.markdown(
    #     f"""
    #     **Precision@{top_k}:** {eval_result['precision']:.4f}<br>
    #     **Recall@{top_k}:** {eval_result['recall']:.4f}<br>
    #     **F1 Score@{top_k}:** {eval_result['f1_score']:.4f}
    #     """,
    #     unsafe_allow_html=True
    # )
    
    # st.divider()
    
    folder_path = "penilaian"
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))

    data_rows = []

    for file in csv_files:
        try:
            df = pd.read_csv(file)

            # Ekstrak nama peneliti target dari nama file
            nama_file = os.path.basename(file)
            nama_pemberi = nama_file.replace("penilaian_", "").split("_")[0]

            # Ambil hanya nilai-nilainya (pastikan urutan tetap)
            nilai_list = df["Nilai"].tolist()[:5]  # pastikan hanya 5 rekomendasi

            # Simpan sebagai dict
            row = {"Peneliti Target": nama_pemberi.strip().upper()}
            for i, nilai in enumerate(nilai_list):
                row[f"R{i+1}"] = nilai
            row["Rata-rata"] = round(sum(nilai_list) / len(nilai_list), 1)

            # Ambil komentar dari baris pertama (jika kolom 'Komentar' ada)
            if "Komentar" in df.columns:
                komentar = df.loc[0, "Komentar"]
                row["Komentar"] = komentar
            else:
                row["Komentar"] = "-"
            
            data_rows.append(row)

        except Exception as e:
            st.error(f"❌ Gagal membaca {file}: {e}")

    # Tampilkan jika ada data
    if data_rows:
        df_tabel = pd.DataFrame(data_rows)
        df_tabel = df_tabel.sort_values(by="Peneliti Target")  # opsional: urutkan nama
        st.markdown("### 📋 Rekap Penilaian Rekomendasi dari Peneliti")
        
        # Buat DataFrame dari semua baris
        df_tabel = pd.DataFrame(data_rows)

        # Hitung rata-rata dari kolom 'Rata-rata'
        rata_rata_total = df_tabel["Rata-rata"].mean()

        # Tambahkan baris Total
        row_total = {"Peneliti Target": "Total"}
        for i in range(1, 6):
            row_total[f"R{i}"] = ""
        row_total["Rata-rata"] = round(rata_rata_total, 2)
        row_total["Komentar"] = "-"

        df_tabel.loc[len(df_tabel)] = row_total
        
        st.dataframe(df_tabel, use_container_width=True, hide_index=True)
    else:
        st.markdown("### 📋 Rekap Penilaian Rekomendasi dari Peneliti")
        # Hardcode data langsung ke DataFrame
        data = [
            [1, "YUSUP MIFTAHUDDIN", "Informatika", 5, 4, 5, 5, 4],
            [2, "ALIF ULFA AFIFAH", "Teknik Industri", 2, 4, 4, 3, 4],
            [3,"CHOERUDIN", "Teknik Kimia", 2, 2, 2, 2, 3],
            [4, "MARISA PREMITASARI", "Informatika", 3, 3, 3, 3, 3],
            [5,"MOHAMMAD AZIS MAHARDIKA", "Teknik Mesin", 3, 2, 3, 2, 2],
            [6,"IRMA AMELIA DEWI", "Informatika", 4, 4, 4, 2, 5],
            [7,"IYUS KUSNAEDI", "Desain Interior", 4, 3, 4, 5, 3],
            [8,"ARSYAD RAMADHAN DARLIS", "Teknik Elektro", 1, 2, 4, 3, 2],
            [9,"KURNIA RAMADHAN PUTRA", "Sistem Informasi", 5, 5, 5, 5, 4],
            [10,"LIMAN HARTAWAN", "Teknik Mesin", 5, 5, 5, 5, 5],
            [11,"DEWI KANIA SARI", "Teknik Geodesi", 3, 5, 4, 5, 5],
            [12,"CORRY CAROMAWATI", "Sistem Informasi", 1, 1, 3, 2, 3],
            [13,"YOULLIA INDRAWATY NURHASANNAH", "Informatika", 5, 5, 5, 5, 5],
            [14,"NABILLA DINA ADHARINA", "Perencanaan Wilayah dan Kota", 4, 4, 3, 3, 5],
            [15,"HENDRO PRASSETIYO", "Teknik Industri", 5, 2, 3, 2, 3],
            [16,"MAHARANI DIAN PERMANASARI", "Desain Produk", 4, 4, 4, 4, 4],
            [17,"SOFIA UMAROH", "Sistem Informasi", 5, 4, 4, 2, 1],
            [18,"EUNEKE WIDYANINGSIH", "Teknik Sipil", 5, 5, 5, 5, 5],
            [19,"SRI SUCI YUNIAR", "Teknik Industri", 5, 1, 4, 1, 4],
            [20,"MUHAMMAD RIDWAN", "Teknik Mesin", 5, 5, 5, 5, 5],
            [21,"NUR FITRIANTI FAHRUDIN", "Sistem Informasi", 4, 5, 5, 5, 4],
            [22,"RADEN BUDIRAHARJO", "Sistem Informasi", 2, 2, 2, 2, 2],
            [23,"ASEP RIZAL NURJAMAN", "Sistem Informasi", 4, 4, 3, 4, 4],
            [24,"MIRA MUSRINI BARMAWI", "Sistem Informasi", 4, 4, 4, 4, 4],
            [25,"IWAN JUWANA", "Teknik Lingkungan", 5, 2, 4, 3, 5],
        ]

        df_hardcode = pd.DataFrame(data, columns=["No.", "Peneliti Target", "Department", "R1", "R2", "R3", "R4", "R5"])

        st.dataframe(df_hardcode, use_container_width=True, hide_index=True)
        # ----------------------
        # Hitung Precision@5 dan MAP@5
        # ----------------------
        precision_list = []
        ap_list = []

        for _, row in df_hardcode.iterrows():
            ratings = [row["R1"], row["R2"], row["R3"], row["R4"], row["R5"]]

            # relevan = nilai >= 4
            relevan = [1 if r >= 4 else 0 for r in ratings]

            # Precision@5 = jumlah relevan / 5
            precision = sum(relevan) / len(relevan)
            precision_list.append(precision)

            # AP@5 = (1/r) * Σ Precision@k * rel(k)
            r = sum(relevan)
            if r > 0:
                prec_at_k = []
                rel_count = 0
                for k in range(1, 6):
                    if relevan[k-1] == 1:
                        rel_count += 1
                        prec_at_k.append(rel_count / k)
                ap = sum(prec_at_k) / r
            else:
                ap = 0
            ap_list.append(ap)

        # MAP@5 = rata-rata AP@5
        map_at_5 = sum(ap_list) / len(ap_list)

        st.markdown("### 📊 Hasil Perhitungan Metrik")
        st.write(f"**Precision@5:** {round(sum(precision_list)/len(precision_list), 3)}")
        st.write(f"**MAP@5:** {round(map_at_5, 3)}")
    
    # ! DELETE
    # import os
    # import glob

    # folder_path = "penilaian"
    # csv_files = glob.glob(os.path.join(folder_path, "*.csv"))

    # st.markdown("### 🗑️ Hapus File Penilaian")

    # for file in csv_files:
    #     nama_file = os.path.basename(file)
    #     col1, col2 = st.columns([0.8, 0.2])
    #     with col1:
    #         st.write(nama_file)
    #     with col2:
    #         if st.button(f"Hapus", key=nama_file):
    #             os.remove(file)
    #             st.success(f"✅ {nama_file} telah dihapus.")
    #             st.rerun()
    # ! DELETE
    # ! MAIN
