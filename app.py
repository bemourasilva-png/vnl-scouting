import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go

st.set_page_config(page_title="VNL 2024 — Scouting de Atletas", page_icon="🏐", layout="wide")

# Mapa de nomes mediáticos
NOMES_MEDIA = {
    "Vargas": "Vargas (Melissa)",
    "Koga": "Koga Sarina",
    "Van Ryk": "Van Ryk",
    "Li Y.Y.": "Li Yingying",
    "Gray": "Gray Micha",
    "Egonu": "Egonu Paola",
    "Stysiak": "Stysiak Magdalena",
    "Gabi": "Gabi (Gabriela Guimarães)",
    "Ishikawa": "Ishikawa Saori",
    "Ana Cristina": "Ana Cristina",
    "Boskovic": "Bošković Tijana",
    "Kurtagić": "Kurtagić Jovana",
    "Korneluk": "Korneluk Agnieszka",
    "Carol": "Carol (Carolina Gattaz)",
    "Kästner": "Kästner Johanna",
    "Diao L.Y.": "Diao Linyu",
    "Iwasaki": "Iwasaki Akari",
    "Brie": "Brie Alison",
    "Szczygłowska": "Szczygłowska",
    "Wang M.J.": "Wang Mengjie",
    "Nyeme": "Nyeme (Ana Beatriz)",
    "Pogany": "Pogany Dorina",
    "Medved": "Medved Silvija",
    "Peña Isabel": "Peña Isabel",
    "Thompson": "Thompson Jordan",
    "Maglio": "Maglio Alexa",
    "Dain": "Dain Yeji",
}

# Mapa de posições
POSICOES_PT = {
    "OH": "Ponteira",
    "OP": "Oposta",
    "MB": "Central",
    "S": "Levantadora",
    "L": "Líbero",
}

FEATURES_PT = {
    "attack": "Ataque",
    "block": "Bloqueio",
    "serve": "Saque",
    "set": "Levantamento",
    "dig": "Defesa",
    "receive": "Recepção",
}

@st.cache_resource
def carregar_modelo():
    with open("modelo/kmeans_model.pkl", "rb") as f:
        modelo = pickle.load(f)
    with open("modelo/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return modelo, scaler

@st.cache_data
def carregar_dados():
    df = pd.read_csv("vnl.csv")
    nomes_cluster = {
        0: "🔴 Atacantes Completas",
        1: "🔵 Centrais Bloqueadoras",
        2: "🟢 Levantadoras",
        3: "🟡 Perfil Base",
        4: "🟣 Especialistas Defensivas"
    }
    features = ["attack", "block", "serve", "set", "dig", "receive"]
    modelo, scaler = carregar_modelo()
    X_scaled = scaler.transform(df[features])
    df["cluster"] = modelo.predict(X_scaled)
    df["perfil"] = df["cluster"].map(nomes_cluster)
    df["nome_media"] = df["player"].map(lambda x: NOMES_MEDIA.get(x, x))
    df["posicao_pt"] = df["position"].map(lambda x: POSICOES_PT.get(x, x))
    return df

modelo, scaler = carregar_modelo()
df = carregar_dados()
features = ["attack", "block", "serve", "set", "dig", "receive"]
features_pt = [FEATURES_PT[f] for f in features]

cores_cluster = {
    0: "#e74c3c",
    1: "#3498db",
    2: "#2ecc71",
    3: "#f39c12",
    4: "#9b59b6"
}

st.title("🏐 VNL 2024 — Segmentação de Perfis de Atletas")
st.markdown("Projeto de *Sports Analytics* com **K-Means Clustering** | Liga das Nações de Voleibol Feminino 2024")
st.divider()

tab1, tab2, tab3 = st.tabs(["🔍 Pesquisar Atleta", "🧪 Prever Perfil", "📊 Visão Geral"])

# ============================================================
# TAB 1
# ============================================================
with tab1:
    st.subheader("🔍 Pesquisar Atleta do Dataset")

    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        paises = ["Todos"] + sorted(df["country"].unique().tolist())
        pais_sel = st.selectbox("🌍 Filtrar por País", paises)
    with col_f2:
        posicoes = ["Todas"] + sorted(df["posicao_pt"].unique().tolist())
        pos_sel = st.selectbox("📌 Filtrar por Posição", posicoes)
    with col_f3:
        perfis = ["Todos"] + sorted(df["perfil"].unique().tolist())
        perfil_sel = st.selectbox("🎯 Filtrar por Perfil", perfis)

    df_filtrado = df.copy()
    if pais_sel != "Todos":
        df_filtrado = df_filtrado[df_filtrado["country"] == pais_sel]
    if pos_sel != "Todas":
        df_filtrado = df_filtrado[df_filtrado["posicao_pt"] == pos_sel]
    if perfil_sel != "Todos":
        df_filtrado = df_filtrado[df_filtrado["perfil"] == perfil_sel]

    if df_filtrado.empty:
        st.warning("Nenhuma atleta encontrada com esses filtros.")
    else:
        atleta_opcoes = sorted(df_filtrado["nome_media"].tolist())
        atleta_sel = st.selectbox("👤 Selecione uma Atleta", atleta_opcoes)

        if atleta_sel:
            atleta = df_filtrado[df_filtrado["nome_media"] == atleta_sel].iloc[0]
            cluster_id = atleta["cluster"]
            cor = cores_cluster[cluster_id]

            col1, col2 = st.columns([1, 2])
            with col1:
                st.markdown(f"### {atleta['nome_media']}")
                st.markdown(f"**País:** {atleta['country']}")
                st.markdown(f"**Posição:** {atleta['posicao_pt']}")
                st.markdown(f"**Idade:** {atleta['age']} anos")
                st.markdown(
                    f"<div style='background-color:{cor};padding:10px;border-radius:8px;"
                    f"color:white;font-weight:bold;font-size:16px;text-align:center'>"
                    f"{atleta['perfil']}</div>",
                    unsafe_allow_html=True
                )

            with col2:
                valores = [atleta[f] for f in features]
                max_val = max(valores) if max(valores) > 0 else 1
                fig = go.Figure()
                fig.add_trace(go.Scatterpolar(
                    r=valores + [valores[0]],
                    theta=features_pt + [features_pt[0]],
                    fill="toself",
                    fillcolor=cor,
                    opacity=0.4,
                    line=dict(color=cor, width=2),
                    name=atleta["nome_media"]
                ))
                fig.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, max_val * 1.2])),
                    title=f"Radar de Performance — {atleta['nome_media']}",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)

            st.markdown("#### 📊 Comparação com a média do perfil")
            media_cluster = df[df["cluster"] == cluster_id][features].mean().round(2)
            comp = pd.DataFrame({
                "Fundamento": features_pt,
                "Atleta": [round(atleta[f], 2) for f in features],
                "Média do Perfil": media_cluster.values
            })
            st.dataframe(comp.set_index("Fundamento"), use_container_width=True)

# ============================================================
# TAB 2
# ============================================================
with tab2:
    st.subheader("🧪 Montar Atleta Hipotética")
    st.markdown("Ajuste os atributos e descubra a que perfil a atleta pertenceria.")

    col1, col2, col3 = st.columns(3)
    with col1:
        att = st.slider("⚡ Ataque",      0.0, 25.0, 5.0, 0.1)
        blk = st.slider("🛡️ Bloqueio",   0.0, 5.0,  0.5, 0.1)
    with col2:
        srv = st.slider("💥 Saque",       0.0, 5.0,  0.3, 0.1)
        st_  = st.slider("🤝 Levantamento", 0.0, 30.0, 0.0, 0.1)
    with col3:
        dig = st.slider("🔽 Defesa",      0.0, 15.0, 2.0, 0.1)
        rec = st.slider("📥 Recepção",    0.0, 12.0, 1.0, 0.1)

    if st.button("🔍 Prever Perfil", type="primary"):
        entrada = np.array([[att, blk, srv, st_, dig, rec]])
        entrada_scaled = scaler.transform(entrada)
        cluster_pred = modelo.predict(entrada_scaled)[0]

        nomes_cluster = {
            0: "🔴 Atacantes Completas",
            1: "🔵 Centrais Bloqueadoras",
            2: "🟢 Levantadoras",
            3: "🟡 Perfil Base",
            4: "🟣 Especialistas Defensivas"
        }
        cor = cores_cluster[cluster_pred]
        st.markdown(
            f"<div style='background-color:{cor};padding:20px;border-radius:10px;"
            f"color:white;font-weight:bold;font-size:22px;text-align:center;margin-top:20px'>"
            f"Perfil Previsto: {nomes_cluster[cluster_pred]}</div>",
            unsafe_allow_html=True
        )

        valores = [att, blk, srv, st_, dig, rec]
        # Normalizar para o radar (escala 0-100 para melhor visualização)
        maximos = [25.0, 5.0, 5.0, 30.0, 15.0, 12.0]
        valores_norm = [round((v / m) * 100, 1) for v, m in zip(valores, maximos)]

        col_r1, col_r2 = st.columns(2)

        with col_r1:
            st.markdown("#### Valores Inseridos")
            dados_tabela = pd.DataFrame({
                "Fundamento": features_pt,
                "Valor": valores,
                "% do Máximo": [f"{round((v/m)*100,1)}%" for v, m in zip(valores, maximos)]
            })
            st.dataframe(dados_tabela.set_index("Fundamento"), use_container_width=True)

        with col_r2:
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=valores_norm + [valores_norm[0]],
                theta=features_pt + [features_pt[0]],
                fill="toself",
                fillcolor=cor,
                opacity=0.4,
                line=dict(color=cor, width=2),
                name="Atleta Hipotética"
            ))
            fig.update_layout(
                polar=dict(radialaxis=dict(
                    visible=True,
                    range=[0, 100],
                    tickvals=[25, 50, 75, 100],
                    ticktext=["25%", "50%", "75%", "100%"]
                )),
                title="Radar Normalizado (% do máximo possível)",
                height=380
            )
            st.plotly_chart(fig, use_container_width=True)

# ============================================================
# TAB 3
# ============================================================
with tab3:
    st.subheader("📊 Distribuição dos Perfis")

    col1, col2 = st.columns(2)
    with col1:
        contagem = df["perfil"].value_counts().reset_index()
        contagem.columns = ["Perfil", "Atletas"]
        fig_bar = go.Figure(go.Bar(
            x=contagem["Perfil"],
            y=contagem["Atletas"],
            marker_color=list(cores_cluster.values()),
            text=contagem["Atletas"],
            textposition="outside"
        ))
        fig_bar.update_layout(title="Atletas por Perfil", height=400)
        st.plotly_chart(fig_bar, use_container_width=True)

    with col2:
        media_perfis = df.groupby("cluster")[features].mean().round(2)
        media_perfis.columns = features_pt
        media_perfis.index = [
            "Atacantes Completas", "Centrais Bloqueadoras",
            "Levantadoras", "Perfil Base", "Especialistas Defensivas"
        ]
        st.markdown("#### Médias por Perfil")
        st.dataframe(media_perfis, use_container_width=True)

    st.divider()
    st.markdown("#### 📋 Dataset Completo com Perfis")
    df_exibir = df[["nome_media", "country", "posicao_pt", "age"] + features + ["perfil"]].copy()
    df_exibir.columns = ["Atleta", "País", "Posição", "Idade"] + features_pt + ["Perfil"]
    st.dataframe(df_exibir, use_container_width=True)
