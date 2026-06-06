import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go

st.set_page_config(page_title="VNL 2024 — Scouting de Atletas", page_icon="🏐", layout="wide")

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
    nomes_cluster = {0: "🔴 Atacantes Completas", 1: "🔵 Centrais Bloqueadoras", 2: "🟢 Levantadoras", 3: "🟡 Perfil Base", 4: "🟣 Especialistas Defensivas"}
    features = ["attack", "block", "serve", "set", "dig", "receive"]
    modelo, scaler = carregar_modelo()
    X_scaled = scaler.transform(df[features])
    df["cluster"] = modelo.predict(X_scaled)
    df["perfil"] = df["cluster"].map(nomes_cluster)
    return df

modelo, scaler = carregar_modelo()
df = carregar_dados()
features = ["attack", "block", "serve", "set", "dig", "receive"]
cores_cluster = {0: "#e74c3c", 1: "#3498db", 2: "#2ecc71", 3: "#f39c12", 4: "#9b59b6"}

st.title("🏐 VNL 2024 — Segmentação de Perfis de Atletas")
st.markdown("Projeto de *Sports Analytics* com **K-Means Clustering** | Liga das Nações de Voleibol Feminino 2024")
st.divider()

tab1, tab2, tab3 = st.tabs(["🔍 Pesquisar Atleta", "🧪 Prever Perfil", "📊 Visão Geral"])

with tab1:
    st.subheader("Pesquisar Atleta do Dataset")
    col1, col2 = st.columns([1, 2])
    with col1:
        atleta_sel = st.selectbox("Selecione uma atleta:", options=sorted(df["player"].tolist()))
    if atleta_sel:
        atleta = df[df["player"] == atleta_sel].iloc[0]
        cluster_id = atleta["cluster"]
        cor = cores_cluster[cluster_id]
        with col1:
            st.markdown(f"### {atleta['player']}")
            st.markdown(f"**País:** {atleta['country']}")
            st.markdown(f"**Posição:** {atleta['position']}")
            st.markdown(f"**Idade:** {atleta['age']} anos")
            st.markdown(f"<div style='background-color:{cor};padding:10px;border-radius:8px;color:white;font-weight:bold;font-size:16px;text-align:center'>{atleta['perfil']}</div>", unsafe_allow_html=True)
        with col2:
            valores = [atleta[f] for f in features]
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(r=valores + [valores[0]], theta=features + [features[0]], fill="toself", fillcolor=cor, opacity=0.4, line=dict(color=cor, width=2), name=atleta["player"]))
            fig.update_layout(polar=dict(radialaxis=dict(visible=True)), title=f"Radar de Performance — {atleta['player']}", height=400)
            st.plotly_chart(fig, use_container_width=True)
        st.markdown("#### 📊 Comparação com a média do perfil")
        media_cluster = df[df["cluster"] == cluster_id][features].mean().round(2)
        comp = pd.DataFrame({"Feature": features, "Atleta": [round(atleta[f], 2) for f in features], "Média do Perfil": media_cluster.values})
        st.dataframe(comp.set_index("Feature"), use_container_width=True)

with tab2:
    st.subheader("🧪 Prever Perfil de Atleta Fictícia")
    st.markdown("Insira os atributos e descubra a que perfil a atleta pertenceria.")
    col1, col2, col3 = st.columns(3)
    with col1:
        att = st.slider("⚡ Attack", 0.0, 25.0, 5.0, 0.1)
        blk = st.slider("🛡️ Block", 0.0, 5.0, 0.5, 0.1)
    with col2:
        srv = st.slider("💥 Serve", 0.0, 5.0, 0.3, 0.1)
        st_ = st.slider("🤝 Set", 0.0, 30.0, 0.0, 0.1)
    with col3:
        dig = st.slider("🔽 Dig", 0.0, 15.0, 2.0, 0.1)
        rec = st.slider("📥 Receive", 0.0, 12.0, 1.0, 0.1)
    if st.button("🔍 Prever Perfil", type="primary"):
        entrada = np.array([[att, blk, srv, st_, dig, rec]])
        entrada_scaled = scaler.transform(entrada)
        cluster_pred = modelo.predict(entrada_scaled)[0]
        nomes_cluster = {0: "🔴 Atacantes Completas", 1: "🔵 Centrais Bloqueadoras", 2: "🟢 Levantadoras", 3: "🟡 Perfil Base", 4: "🟣 Especialistas Defensivas"}
        cor = cores_cluster[cluster_pred]
        st.markdown(f"<div style='background-color:{cor};padding:20px;border-radius:10px;color:white;font-weight:bold;font-size:22px;text-align:center;margin-top:20px'>Perfil Previsto: {nomes_cluster[cluster_pred]}</div>", unsafe_allow_html=True)
        valores = [att, blk, srv, st_, dig, rec]
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(r=valores + [valores[0]], theta=features + [features[0]], fill="toself", fillcolor=cor, opacity=0.4, line=dict(color=cor, width=2), name="Atleta Fictícia"))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True)), title="Radar — Atleta Fictícia", height=380)
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("📊 Distribuição dos Perfis")
    col1, col2 = st.columns(2)
    with col1:
        contagem = df["perfil"].value_counts().reset_index()
        contagem.columns = ["Perfil", "Atletas"]
        fig_bar = go.Figure(go.Bar(x=contagem["Perfil"], y=contagem["Atletas"], marker_color=list(cores_cluster.values()), text=contagem["Atletas"], textposition="outside"))
        fig_bar.update_layout(title="Atletas por Perfil", height=400)
        st.plotly_chart(fig_bar, use_container_width=True)
    with col2:
        media_perfis = df.groupby("cluster")[features].mean().round(2)
        media_perfis.index = ["Atacantes Completas", "Centrais Bloqueadoras", "Levantadoras", "Perfil Base", "Especialistas Defensivas"]
        st.markdown("#### Médias por Perfil")
        st.dataframe(media_perfis, use_container_width=True)
    st.divider()
    st.markdown("#### 📋 Dataset Completo com Perfis")
    st.dataframe(df[["player", "country", "position", "age"] + features + ["perfil"]], use_container_width=True)
