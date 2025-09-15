import streamlit as st
import pandas as pd
import re
import io
import unicodedata

# ------------------------------------------------------------------------------
# App: Leads x Prospecções — Identificar leads que viraram prospecções
# ------------------------------------------------------------------------------

st.set_page_config(page_title="Leads x Prospecções", layout="wide")
st.title("Leads x Prospecções")
st.write(
    """
    Envie dois arquivos:
    - Arquivo de Leads
    - Arquivo de Prospecções

    O sistema identificará automaticamente, a partir das colunas escolhidas, quais leads viraram prospecções.
    Você pode selecionar múltiplas colunas de identificação (ex.: email, telefone/WhatsApp, documento, login, nome).
    """
)

# ------------------------------------------------------------------------------
# Utilitários
# ------------------------------------------------------------------------------

def read_file(file):
    if file.name.lower().endswith(".csv"):
        # tenta detectar separador automaticamente e faz fallback
        try:
            df = pd.read_csv(file, sep=None, engine="python")
            if df.shape[1] == 1:  # separador errado? tenta vírgula
                file.seek(0)
                df = pd.read_csv(file, sep=",")
            return df
        except Exception:
            file.seek(0)
            return pd.read_csv(file, sep=",")
    else:
        return pd.read_excel(file)

def normalize_text(s: str) -> str:
    if pd.isna(s):
        return ""
    s = str(s).strip().lower()
    # remove acentos
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    # colapsa espaços
    s = re.sub(r"\s+", " ", s)
    return s

def only_digits(s: str) -> str:
    if pd.isna(s):
        return ""
    return re.sub(r"\D+", "", str(s))

def clean_alnum(s: str) -> str:
    if pd.isna(s):
        return ""
    return re.sub(r"[^a-z0-9]+", "", normalize_text(s))

def gen_variants(value: str) -> set:
    # Gera variações para tentar casar chaves mesmo com diferenças pequenas
    base = normalize_text(value)
    if not base:
        return set()
    variants = {
        base,
        base.replace("_", " ").strip(),
        base.replace(" ", "_").strip(),
        clean_alnum(base),
    }
    digits = only_digits(value)
    if digits and len(digits) >= 8:
        variants.add(digits.lstrip("0") or digits)  # remove zeros à esquerda
        variants.add(digits)
    # remove vazios
    variants = {v for v in variants if v}
    return variants

def guess_id_columns(columns: list) -> list:
    # Sugere colunas prováveis para identificação
    keys = ["email", "login", "cpf", "cnpj", "doc", "documento", "telefone", "whats", "cel", "nome", "razao", "razão", "id"]
    ranked = []
    for c in columns:
        lc = c.lower()
        score = sum(k in lc for k in keys)
        ranked.append((score, c))
    ranked.sort(reverse=True)
    # escolhe até 3 melhores colunas com score > 0
    defaults = [c for score, c in ranked if score > 0][:3]
    # fallback: pelo menos 1 coluna
    if not defaults and columns:
        defaults = [columns[0]]
    return defaults

def build_keyframe(df: pd.DataFrame, id_cols: list, side_label: str) -> pd.DataFrame:
    # Cria uma tabela key -> índice da linha
    rows = []
    for idx, row in df.iterrows():
        keys = set()
        for col in id_cols:
            if col not in df.columns:
                continue
            v = row[col]
            keys |= gen_variants(v)
        for k in keys:
            rows.append((k, idx))
    if not rows:
        return pd.DataFrame(columns=["key", f"idx_{side_label}"])
    kf = pd.DataFrame(rows, columns=["key", f"idx_{side_label}"]).drop_duplicates()
    return kf

def to_download_button(df: pd.DataFrame, filename: str, label: str):
    csv_data = df.to_csv(index=False, sep=";", encoding="utf-8-sig")
    st.download_button(label=label, data=csv_data, file_name=filename, mime="text/csv")

# ------------------------------------------------------------------------------
# Upload
# ------------------------------------------------------------------------------

col1, col2 = st.columns(2)
with col1:
    file_leads = st.file_uploader("1️⃣ Arquivo de Leads", type=["xlsx", "xls", "csv"], key="leads")
with col2:
    file_prosp = st.file_uploader("2️⃣ Arquivo de Prospecções", type=["xlsx", "xls", "csv"], key="prosp")

if file_leads and file_prosp:
    df_leads = read_file(file_leads)
    df_prosp = read_file(file_prosp)

    st.subheader("Selecione as colunas de identificação (use múltiplas se possível)")

    with st.form("config_match_form"):
        st.markdown("As colunas escolhidas serão usadas para tentar encontrar correspondências. Ex.: email, telefone/WhatsApp, CPF/CNPJ, login, nome.")
        colunas_leads = df_leads.columns.tolist()
        colunas_prosp = df_prosp.columns.tolist()

        id_cols_leads = st.multiselect(
            "Colunas de identificação no arquivo de Leads",
            options=colunas_leads,
            default=guess_id_columns(colunas_leads),
            key="id_cols_leads",
        )
        id_cols_prosp = st.multiselect(
            "Colunas de identificação no arquivo de Prospecções",
            options=colunas_prosp,
            default=guess_id_columns(colunas_prosp),
            key="id_cols_prosp",
        )

        st.markdown("---")
        st.markdown("Opcional: selecione colunas para exibição nos resultados (além das chaves).")
        display_cols_leads = st.multiselect(
            "Colunas para exibir (Leads)",
            options=colunas_leads,
            default=[c for c in colunas_leads[:5] if c not in id_cols_leads] or colunas_leads[:3],
            key="display_cols_leads",
        )
        display_cols_prosp = st.multiselect(
            "Colunas para exibir (Prospecções)",
            options=colunas_prosp,
            default=[c for c in colunas_prosp[:5] if c not in id_cols_prosp] or colunas_prosp[:3],
            key="display_cols_prosp",
        )

        submitted = st.form_submit_button("Gerar Relatório")

    if submitted:
        if not id_cols_leads or not id_cols_prosp:
            st.error("Selecione ao menos 1 coluna de identificação em cada arquivo.")
            st.stop()

        # Prefixa colunas para evitar colisão de nomes
        df_leads_pref = df_leads.copy()
        df_prosp_pref = df_prosp.copy()
        df_leads_pref.columns = [f"lead__{c}" for c in df_leads_pref.columns]
        df_prosp_pref.columns = [f"prosp__{c}" for c in df_prosp_pref.columns]
        df_leads_pref["idx_lead"] = df_leads_pref.index
        df_prosp_pref["idx_prosp"] = df_prosp_pref.index

        # Mapa de nome original -> nome prefixado
        lead_map = {c: f"lead__{c}" for c in df_leads.columns}
        prosp_map = {c: f"prosp__{c}" for c in df_prosp.columns}

        # Keyframes
        kf_leads = build_keyframe(df_leads, id_cols_leads, "lead")
        kf_prosp = build_keyframe(df_prosp, id_cols_prosp, "prosp")

        if kf_leads.empty or kf_prosp.empty:
            st.warning("Não foi possível gerar chaves de comparação com as colunas escolhidas.")
            st.stop()

        # Matches
        matches = kf_leads.merge(kf_prosp, on="key").drop_duplicates(subset=["idx_lead", "idx_prosp"])
        # Junta dados completos
        matched = (
            matches
            .merge(df_leads_pref, on="idx_lead", how="left")
            .merge(df_prosp_pref, on="idx_prosp", how="left")
        )

        # Leads sem match
        matched_lead_idxs = set(matches["idx_lead"].unique().tolist())
        unmatched_leads = df_leads_pref[~df_leads_pref["idx_lead"].isin(matched_lead_idxs)].copy()

        # Prospecções sem match
        matched_prosp_idxs = set(matches["idx_prosp"].unique().tolist())
        unmatched_prosp = df_prosp_pref[~df_prosp_pref["idx_prosp"].isin(matched_prosp_idxs)].copy()

        # Preparar colunas de exibição
        lead_cols_show = [lead_map[c] for c in id_cols_leads + [c for c in display_cols_leads if c not in id_cols_leads] if c in lead_map]
        prosp_cols_show = [prosp_map[c] for c in id_cols_prosp + [c for c in display_cols_prosp if c not in id_cols_prosp] if c in prosp_map]

        # Resultado: Viraram Prospecção (tabela combinada)
        cols_matched = ["key"] + lead_cols_show + prosp_cols_show
        matched_show = matched[cols_matched].drop_duplicates()

        # Resultado: Leads não encontrados
        unmatched_leads_show = unmatched_leads[lead_cols_show].drop_duplicates()


        # Resumo
        st.subheader("Resumo")
        c1, c2 = st.columns(2)
        with c1:
            st.success(f"Leads que viraram Prospecção: {matched_show[lead_cols_show].drop_duplicates().shape[0]}")
        with c2:
            st.warning(f"Leads sem Prospecção: {unmatched_leads_show.shape[0]}")

        # Abas de resultados
        tab1, tab2 = st.tabs(["Viraram Prospecção", "Leads sem Prospecção"])

        with tab1:
            st.dataframe(matched_show, use_container_width=True)
            to_download_button(matched_show, "leads_viraram_prospeccao.csv", "📥 Baixar (Viraram Prospecção)")

        with tab2:
            st.dataframe(unmatched_leads_show, use_container_width=True)
            to_download_button(unmatched_leads_show, "leads_nao_encontrados.csv", "📥 Baixar (Leads sem Prospecção)")
else:
    st.info("Envie os dois arquivos para começar.")
