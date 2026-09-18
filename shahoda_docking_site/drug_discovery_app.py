"""
Drug Discovery Assistant
===========================================================
Integrated workspace for target discovery, molecular design,
target prediction, 3D visualization, and scaffold data entry.

Run with:
    pip install streamlit streamlit-ketcher rdkit requests fpdf2
    streamlit run drug_discovery_app_one_page.py
"""

import io
import csv
import json
import requests
import streamlit as st
import streamlit.components.v1 as components
from pathlib import Path
from streamlit_ketcher import st_ketcher
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski, AllChem, rdMolDescriptors
from rdkit.Chem.Draw import rdMolDraw2D
from fpdf import FPDF

# ---------------------------------------------------------
# Page config MUST be the first Streamlit call
# ---------------------------------------------------------
st.set_page_config(
    page_title="Drug Discovery Assistant",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------
# Persistent storage helpers
# ---------------------------------------------------------
DATA_FILE = Path("user_data.json")


def load_data():
    if DATA_FILE.exists():
        try:
            with open(DATA_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_data():
    data = {
        "my_compounds": st.session_state.get("my_compounds", []),
        "scaffold_entries": st.session_state.get("scaffold_entries", []),
        "scaffold_header": st.session_state.get("scaffold_header", {}),
        "smiles": st.session_state.get("smiles", ""),
    }
    try:
        with open(DATA_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception:
        pass


# ---------------------------------------------------------
# Global styling
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    .stApp { background: #0b0d12; color: #e2e5ea; }
    .block-container { max-width: 1400px; padding: 0 2rem 4rem; }
    section[data-testid="stSidebar"] { display: none; }
    .topnav {
        position: sticky; top: 0; z-index: 999;
        background: rgba(11,13,18,0.85);
        backdrop-filter: blur(12px);
        border-bottom: 1px solid #1e212b;
        padding: 14px 2rem 12px;
        margin: 0 -2rem 32px;
        display: flex; align-items: center; justify-content: space-between;
    }
    .topnav .logo {
        font-size: 1.15rem; font-weight: 700; letter-spacing: -.2px;
        background: linear-gradient(90deg, #4fd1c5, #63b3ed);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .topnav .nav-links { display: flex; gap: 26px; color: #8b93a1; font-size: 0.85rem; }
    .hero {
        background: radial-gradient(circle at 20% 0%, #16233a 0%, #0f141c 45%, #0b0d12 100%);
        border: 1px solid #1e212b; border-radius: 20px;
        padding: 52px 48px 48px; margin-bottom: 42px;
        position: relative; overflow: hidden;
    }
    .hero .badge {
        display: inline-block; padding: 5px 12px; border-radius: 999px;
        background: rgba(79,209,197,.1); border: 1px solid rgba(79,209,197,.25);
        color: #4fd1c5; font-size: 0.72rem; font-weight: 600;
        letter-spacing: .6px; text-transform: uppercase; margin-bottom: 18px;
    }
    .hero h1 { margin: 0 0 14px; font-size: 2.6rem; font-weight: 700; color: #f1f5f9; }
    .hero h1 span { background: linear-gradient(90deg, #4fd1c5, #63b3ed);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .hero p { margin: 0; color: #9aa1ac; font-size: 1rem; max-width: 640px; }
    .section-head { display: flex; align-items: center; gap: 14px; margin: 52px 0 22px; }
    .section-head .num {
        width: 36px; height: 36px; border-radius: 10px;
        background: linear-gradient(135deg, #4fd1c5, #63b3ed);
        color: #0b0d12; font-weight: 800; font-size: 0.95rem;
        display: flex; align-items: center; justify-content: center; flex-shrink: 0;
    }
    .section-head .title { font-size: 1.45rem; font-weight: 700; color: #e2e5ea; }
    .section-head .desc { color: #8b93a1; font-size: 0.88rem; margin-top: 2px; }
    .card {
        background: #12151c; border: 1px solid #1e212b; border-radius: 14px;
        padding: 18px 20px; margin-bottom: 12px;
    }
    .card b { color: #e2e5ea; }
    .card a { color: #63b3ed; text-decoration: none; }
    .card code {
        background: #0b0d12; color: #4fd1c5; padding: 3px 8px;
        border-radius: 6px; font-size: 0.8rem; word-break: break-all;
        display: inline-block; margin-top: 6px;
    }
    .features { display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; }
    .feature { background: #12151c; border: 1px solid #1e212b; border-radius: 12px; padding: 18px; }
    .feature .ic { font-size: 1.4rem; margin-bottom: 8px; }
    .feature .ttl { font-weight: 650; color: #e2e5ea; font-size: 0.92rem; }
    .feature .txt { color: #8b93a1; font-size: 0.78rem; }
    div[data-testid="stMetric"] {
        background: #12151c; border: 1px solid #1e212b;
        border-radius: 12px; padding: 14px 18px;
    }
    div[data-testid="stMetricLabel"] { color: #8b93a1 !important; font-size: 0.8rem !important; }
    div[data-testid="stMetricValue"] { color: #e2e5ea !important; font-size: 1.3rem !important; }
    div[data-testid="stButton"] > button {
        border-radius: 10px; font-weight: 600; min-height: 42px;
        background: #1a1d26; color: #e2e5ea; border: 1px solid #2a3140;
    }
    div[data-testid="stButton"] > button:hover {
        background: #202531; border-color: #4fd1c5; color: #4fd1c5;
    }
    div[data-testid="stDownloadButton"] > button {
        border-radius: 10px; font-weight: 600; min-height: 42px;
        background: #1a1d26; color: #e2e5ea; border: 1px solid #2a3140;
    }
    div[data-testid="stTextInput"] input, div[data-testid="stTextArea"] textarea {
        background: #12151c; color: #e2e5ea; border: 1px solid #2a3140; border-radius: 10px;
    }
    details { background: #12151c; border: 1px solid #1e212b; border-radius: 10px; }
    iframe { border-radius: 14px; border: 1px solid #1e212b; }
    hr { border-color: #1e212b; margin: 42px 0; }
    .footer { text-align: center; color: #565e6d; font-size: 0.78rem;
        padding: 26px 0 6px; border-top: 1px solid #1e212b; margin-top: 60px; }
    .footer a { color: #63b3ed; text-decoration: none; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Session state
# ---------------------------------------------------------
_saved = load_data()
if "smiles" not in st.session_state:
    st.session_state.smiles = _saved.get("smiles", "")
if "_sync_text_input" not in st.session_state:
    st.session_state["_sync_text_input"] = False
if "my_compounds" not in st.session_state:
    st.session_state.my_compounds = _saved.get("my_compounds", [])
if "scaffold_entries" not in st.session_state:
    st.session_state.scaffold_entries = _saved.get("scaffold_entries", [])
if "scaffold_header" not in st.session_state:
    st.session_state.scaffold_header = _saved.get(
        "scaffold_header",
        {"title": "SCAFFOLD 12", "original_smiles": "", "targetability": ""},
    )

# ---------------------------------------------------------
# Top nav + Hero
# ---------------------------------------------------------
st.markdown("""
<div class="topnav">
    <div class="logo">🧬 Discovery Assistant</div>
    <div class="nav-links">
        <span>Overview</span><span>Diseases</span><span>Compounds</span>
        <span>Editor</span><span>Prediction</span><span>Visualization</span>
    </div>
</div>
<div class="hero">
    <div class="badge">Computational Drug Discovery Platform</div>
    <h1>For my girl only,<br><span>her workspace.</span></h1>
    <p>Search the literature, retrieve known drugs, draw and analyze molecules, predict protein targets, and visualize 3D structures — without leaving the page.</p>
</div>
<div class="features">
    <div class="feature"><div class="ic">📚</div><div class="ttl">Literature Search</div><div class="txt">Live PubMed results.</div></div>
    <div class="feature"><div class="ic">💊</div><div class="ttl">Lead Compounds</div><div class="txt">Known drugs from ChEMBL.</div></div>
    <div class="feature"><div class="ic">🎯</div><div class="ttl">Target Prediction</div><div class="txt">SwissTargetPrediction embedded.</div></div>
    <div class="feature"><div class="ic">🔬</div><div class="ttl">3D Visualization</div><div class="txt">MolView embedded.</div></div>
</div>
""", unsafe_allow_html=True)

# ===========================================================
# SECTION 1 — Disease Discovery (PubMed)
# ===========================================================
st.markdown('<div class="section-head"><div class="num">1</div><div><div class="title">Disease Discovery</div><div class="desc">Search recent PubMed literature for emerging and rare diseases by category.</div></div></div>', unsafe_allow_html=True)

category_map = {"Viral": "viral infection", "Bacterial": "bacterial infection", "Oncology": "cancer OR tumor", "Parasitic": "parasitic infection"}
category = st.radio("Category", list(category_map.keys()), horizontal=True, label_visibility="collapsed")


@st.cache_data(ttl=3600, show_spinner=False)
def search_pubmed_diseases(category_term: str, retmax: int = 8):
    term = f'("rare disease"[tiab] OR "novel disease"[tiab] OR "emerging"[tiab]) AND ({category_term})'
    esearch = requests.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
        params={"db": "pubmed", "term": term, "retmax": retmax, "sort": "date", "retmode": "json"}, timeout=15).json()
    ids = esearch.get("esearchresult", {}).get("idlist", [])
    if not ids:
        return []
    esummary = requests.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi",
        params={"db": "pubmed", "id": ",".join(ids), "retmode": "json"}, timeout=15).json()
    return [{"pmid": p, "title": esummary.get("result", {}).get(p, {}).get("title", "Untitled"),
             "pubdate": esummary.get("result", {}).get(p, {}).get("pubdate", "")} for p in ids]


if st.button("🔍  Search PubMed for candidate diseases", use_container_width=True):
    with st.spinner("Searching PubMed..."):
        st.session_state.disease_results = search_pubmed_diseases(category_map[category])

if st.session_state.get("disease_results"):
    st.caption(f"{len(st.session_state.disease_results)} recent articles")
    for r in st.session_state.disease_results:
        st.markdown(f"""<div class="card"><b>{r['title']}</b><br>
        <span style="color:#8b93a1; font-size:0.8rem;">PMID {r['pmid']} · {r['pubdate']}</span> ·
        <a href="https://pubmed.ncbi.nlm.nih.gov/{r['pmid']}/" target="_blank">View on PubMed</a></div>""", unsafe_allow_html=True)

# ===========================================================
# SECTION 2 — Disease → Protein → Compound (FIXED)
# ===========================================================
st.markdown('<div class="section-head"><div class="num">2</div><div><div class="title">Disease → Protein → Compound</div><div class="desc">Enter a disease, get its associated proteins, then pull inhibitor compounds from ChEMBL.</div></div></div>', unsafe_allow_html=True)


@st.cache_data(ttl=3600, show_spinner=False)
def get_disease_proteins(disease_name: str):
    """
    Step 1: Find proteins associated with a disease using Open Targets Platform.
    Uses direct GraphQL query with proper variable handling.
    Returns (targets_list, efo_name).
    """
    base_url = "https://api.platform.opentargets.org/api/v4/graphql"

    # Step 1a: search for disease EFO ID
    search_query = """
    query SearchDisease($queryString: String!) {
      search(queryString: $queryString, entityNames: ["disease"], page: {index: 0, size: 1}) {
        hits {
          id
          name
        }
      }
    }
    """

    try:
        response = requests.post(
            base_url,
            json={"query": search_query, "variables": {"queryString": disease_name}},
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()

        if "errors" in data:
            st.error(f"GraphQL error: {data['errors']}")
            return [], None

        hits = data.get("data", {}).get("search", {}).get("hits", [])
        if not hits:
            return [], None

        efo_id = hits[0]["id"]
        efo_name = hits[0]["name"]

    except Exception as e:
        st.error(f"Failed to search disease: {type(e).__name__}: {e}")
        return [], None

    # Step 1b: get associated targets
    targets_query = """
    query AssociatedTargets($efoId: String!) {
      disease(efoId: $efoId) {
        id
        name
        associatedTargets(page: {index: 0, size: 25}) {
          rows {
            target {
              id
              approvedSymbol
              approvedName
            }
            score
          }
        }
      }
    }
    """

    try:
        response = requests.post(
            base_url,
            json={"query": targets_query, "variables": {"efoId": efo_id}},
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()

        if "errors" in data:
            st.error(f"GraphQL error: {data['errors']}")
            return [], efo_name

        rows = data.get("data", {}).get("disease", {}).get("associatedTargets", {}).get("rows", [])

        targets = []
        for row in rows:
            t = row.get("target", {})
            targets.append({
                "target_id": t.get("id", ""),
                "symbol": t.get("approvedSymbol", ""),
                "name": t.get("approvedName", ""),
                "score": row.get("score", 0),
            })

        return targets, efo_name

    except Exception as e:
        st.error(f"Failed to fetch targets: {type(e).__name__}: {e}")
        return [], efo_name


@st.cache_data(ttl=3600, show_spinner=False)
def get_compounds_for_target(target_symbol: str, target_name: str):
    """Get active compounds for a protein target from ChEMBL."""
    import time

    def fetch(url, params=None, retries=3, timeout=60):
        for attempt in range(retries):
            try:
                r = requests.get(url, params=params, timeout=timeout)
                r.raise_for_status()
                return r.json()
            except Exception:
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)
        return {}

    search_term = target_symbol or target_name
    if not search_term:
        return []

    target_resp = fetch(
        "https://www.ebi.ac.uk/chembl/api/data/target/search.json",
        params={"q": search_term, "limit": 3},
    )
    targets = target_resp.get("targets", [])
    if not targets:
        return []

    chembl_target_id = targets[0].get("target_chembl_id")
    target_type = targets[0].get("target_type", "")
    pref_name = targets[0].get("pref_name", "")

    if not chembl_target_id:
        return []

    act_resp = fetch(
        "https://www.ebi.ac.uk/chembl/api/data/activity.json",
        params={
            "target_chembl_id": chembl_target_id,
            "pchembl_value__isnull": "false",
            "limit": 30,
        },
    )
    activities = act_resp.get("activities", [])

    best_by_mol = {}
    for act in activities:
        mol_id = act.get("molecule_chembl_id")
        pchembl = act.get("pchembl_value")
        if not mol_id:
            continue
        try:
            pchembl_val = float(pchembl) if pchembl is not None else 0
        except (ValueError, TypeError):
            pchembl_val = 0
        if mol_id not in best_by_mol or pchembl_val > best_by_mol[mol_id]["pchembl"]:
            best_by_mol[mol_id] = {
                "molecule_chembl_id": mol_id,
                "pchembl": pchembl_val,
                "activity_type": act.get("standard_type", ""),
                "activity_value": act.get("standard_value", ""),
                "activity_units": act.get("standard_units", ""),
            }

    results = []
    for mol_id, info in list(best_by_mol.items())[:15]:
        mol_resp = fetch(
            f"https://www.ebi.ac.uk/chembl/api/data/molecule/{mol_id}.json",
        )
        if not mol_resp:
            continue
        smiles = (mol_resp.get("molecule_structures") or {}).get("canonical_smiles")
        name = mol_resp.get("pref_name") or mol_id
        if smiles:
            results.append({
                "chembl_target_id": chembl_target_id,
                "target_type": target_type,
                "target_pref_name": pref_name,
                "compound": name,
                "chembl_id": mol_id,
                "smiles": smiles,
                "activity_type": info["activity_type"],
                "activity_value": info["activity_value"],
                "activity_units": info["activity_units"],
                "pchembl": info["pchembl"],
            })

    return results


# ---------- UI: Step 1 ----------
disease_for_chain = st.text_input(
    "Disease name",
    key="disease_chain",
    placeholder="e.g. type 2 diabetes, malaria, breast cancer...",
    label_visibility="collapsed",
)

if st.button("🔬  Step 1: Find proteins linked to this disease", use_container_width=True) and disease_for_chain:
    with st.spinner("Querying Open Targets for associated proteins..."):
        proteins, efo_name = get_disease_proteins(disease_for_chain)
    st.session_state.chain_proteins = proteins
    st.session_state.chain_disease_name = efo_name or disease_for_chain
    st.session_state.selected_target = None
    st.session_state.chain_compounds = None
    if not proteins:
        st.warning(f"No proteins found for '{disease_for_chain}'. Try a different name or check the spelling.")

# ---------- UI: Step 2 — pick protein ----------
if st.session_state.get("chain_proteins"):
    proteins = st.session_state.chain_proteins
    st.markdown(
        f"""<div class="card">
        <b>Disease matched:</b> {st.session_state.get("chain_disease_name", "")}<br>
        <span style="color:#8b93a1; font-size:0.82rem;">
        Found <b>{len(proteins)}</b> associated protein targets.
        Click <b>Find inhibitors</b> next to any protein to see its compounds from ChEMBL.
        </span>
        </div>""",
        unsafe_allow_html=True,
    )

    for idx, p in enumerate(proteins[:15]):
        col_info, col_btn = st.columns([4, 1])
        with col_info:
            st.markdown(
                f"""<div style="background:#12151c; border:1px solid #1e212b;
                border-radius:12px; padding:12px 16px; margin-bottom:8px;">
                <b style="color:#4fd1c5;">{p['symbol']}</b>
                <span style="color:#8b93a1; font-size:0.82rem;"> — {p['name']}</span><br>
                <span style="color:#8b93a1; font-size:0.72rem;">
                Target ID: {p['target_id']} · Score: {p['score']:.3f}
                </span>
                </div>""",
                unsafe_allow_html=True,
            )
        with col_btn:
            if st.button("Find inhibitors", key=f"chain_{idx}", use_container_width=True):
                st.session_state.selected_target = p
                st.session_state.chain_compounds = None

    # ---------- UI: Step 3 — fetch inhibitors ----------
    if st.session_state.get("selected_target"):
        sel = st.session_state.selected_target
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            f"""<div class="card">
            <b style="color:#4fd1c5;">Selected protein: {sel['symbol']}</b><br>
            <span style="color:#8b93a1; font-size:0.82rem;">{sel['name']}</span>
            </div>""",
            unsafe_allow_html=True,
        )

        if st.button("🔗  Fetch inhibitor compounds from ChEMBL", use_container_width=True, key="fetch_inhibitors"):
            with st.spinner(f"Searching ChEMBL for compounds against {sel['symbol']}..."):
                compounds = get_compounds_for_target(sel["symbol"], sel["name"])
            st.session_state.chain_compounds = compounds

        if st.session_state.get("chain_compounds") is not None:
            compounds = st.session_state.chain_compounds
            if not compounds:
                st.warning("No compounds found for this target in ChEMBL.")
            else:
                st.caption(f"{len(compounds)} compounds found")

                buf = io.StringIO()
                writer = csv.writer(buf)
                writer.writerow([
                    "Target Symbol", "Target Name", "ChEMBL Target ID",
                    "Compound", "ChEMBL ID", "SMILES",
                    "Activity Type", "Activity Value", "Units", "pChEMBL",
                ])
                for c in compounds:
                    writer.writerow([
                        sel["symbol"], sel["name"], c["chembl_target_id"],
                        c["compound"], c["chembl_id"], c["smiles"],
                        c["activity_type"], c["activity_value"],
                        c["activity_units"], c["pchembl"],
                    ])
                st.download_button(
                    "📥  Export compounds as CSV",
                    data=buf.getvalue(),
                    file_name=f"compounds_{sel['symbol']}.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

                for c in compounds:
                    st.markdown(
                        f"""<div class="card">
                        <b>{c['compound']}</b>
                        <span style="color:#8b93a1;">({c['chembl_id']})</span><br>
                        <span style="color:#8b93a1; font-size:0.8rem;">
                        Target: {c['target_pref_name']} ({c['chembl_target_id']}) ·
                        {c['target_type']}
                        </span><br>
                        <span style="color:#4fd1c5; font-size:0.82rem;">
                        {c['activity_type']}: {c['activity_value']} {c['activity_units']}
                        (pChEMBL {c['pchembl']})
                        </span><br>
                        <code>{c['smiles']}</code>
                        </div>""",
                        unsafe_allow_html=True,
                    )
                    if st.button(
                        f"Send {c['compound']} to Editor →",
                        key=f"chain_send_{c['chembl_id']}",
                        use_container_width=True,
                    ):
                        st.session_state.smiles = c["smiles"]
                        st.session_state["_sync_text_input"] = True
                        save_data()
                        st.success("SMILES loaded into the Molecule Editor below.")

# ===========================================================
# SECTION 2C — Inhibitor Analysis
# ===========================================================
st.markdown('<div class="section-head"><div class="num">2C</div><div><div class="title">Inhibitor Analysis & Optimization</div><div class="desc">Analyze the inhibitor\'s pharmacophore features and get modification suggestions.</div></div></div>', unsafe_allow_html=True)


@st.cache_data(ttl=3600, show_spinner=False)
def analyze_inhibitor(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    PHARM_SMARTS = {
        "Hydrogen Bond Donor": ["[#7,#8;!H0]"],
        "Hydrogen Bond Acceptor": ["[#7,#8]"],
        "Aromatic": ["a"],
        "Hydrophobic": ["[#6;!$(C=O),!$(C#N)]"],
        "Positive Ionizable": ["[#7;+0;!$(N-C=O)]"],
        "Negative Ionizable": ["[#8;-,+0;$(O=C)]"],
    }

    features_found = {}
    for feat_name, smarts_list in PHARM_SMARTS.items():
        atom_indices = set()
        for smarts in smarts_list:
            pattern = Chem.MolFromSmarts(smarts)
            if pattern:
                for match in mol.GetSubstructMatches(pattern):
                    atom_indices.update(match)
        if atom_indices:
            features_found[feat_name] = {
                "count": len(atom_indices),
                "atoms": sorted(list(atom_indices)),
            }

    mw = Descriptors.MolWt(mol)
    logp = Crippen.MolLogP(mol)
    hbd = Lipinski.NumHDonors(mol)
    hba = Lipinski.NumHAcceptors(mol)
    tpsa = Descriptors.TPSA(mol)
    rot_bonds = Descriptors.NumRotatableBonds(mol)
    n_rings = rdMolDescriptors.CalcNumRings(mol)
    n_arom_rings = rdMolDescriptors.CalcNumAromaticRings(mol)

    suggestions = []
    if hbd == 0:
        suggestions.append("No hydrogen bond donors found — consider adding a donor group (e.g., -OH, -NH2).")
    if hba == 0:
        suggestions.append("No hydrogen bond acceptors found — consider adding an acceptor group (e.g., carbonyl, ether).")
    if n_arom_rings == 0:
        suggestions.append("No aromatic rings found — consider adding an aromatic ring for pi-stacking.")
    if logp > 5:
        suggestions.append(f"LogP is high ({logp:.1f}) — consider adding polar groups to reduce lipophilicity.")
    elif logp < 1:
        suggestions.append(f"LogP is low ({logp:.1f}) — consider adding hydrophobic groups.")
    if tpsa > 140:
        suggestions.append(f"TPSA is high ({tpsa:.0f} Å²) — may have poor oral absorption.")
    if rot_bonds > 10:
        suggestions.append(f"Many rotatable bonds ({rot_bonds}) — consider rigidifying the structure.")
    if mw > 500:
        suggestions.append(f"Molecular weight is high ({mw:.0f}) — consider fragmenting.")
    if not suggestions:
        suggestions.append("The molecule has balanced properties. Consider modifying specific groups to explore SAR.")

    return {
        "features": features_found,
        "properties": {
            "MW": mw, "LogP": logp, "HBD": hbd, "HBA": hba,
            "TPSA": tpsa, "RotBonds": rot_bonds,
            "Rings": n_rings, "AromaticRings": n_arom_rings,
        },
        "suggestions": suggestions,
    }


inhibitor_smiles = st.text_input(
    "Inhibitor SMILES",
    value=st.session_state.get("smiles", ""),
    key="inhibitor_smiles_input",
    placeholder="Paste an inhibitor SMILES to analyze...",
    label_visibility="collapsed",
)

if st.button("🧪  Analyze Inhibitor", use_container_width=True) and inhibitor_smiles:
    with st.spinner("Analyzing pharmacophore features..."):
        analysis = analyze_inhibitor(inhibitor_smiles)
    st.session_state.inhibitor_analysis = analysis

if st.session_state.get("inhibitor_analysis"):
    analysis = st.session_state.inhibitor_analysis
    st.markdown('<div style="color:#8b93a1; font-size:0.8rem; font-weight:600; letter-spacing:.3px; text-transform:uppercase; margin:20px 0 8px;">Molecular Properties</div>', unsafe_allow_html=True)
    props = analysis["properties"]
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("MW", f"{props['MW']:.1f}")
    p2.metric("LogP", f"{props['LogP']:.2f}")
    p3.metric("HBD", props["HBD"])
    p4.metric("HBA", props["HBA"])
    p5, p6, p7, p8 = st.columns(4)
    p5.metric("TPSA", f"{props['TPSA']:.1f}")
    p6.metric("RotBonds", props["RotBonds"])
    p7.metric("Rings", props["Rings"])
    p8.metric("Aromatic Rings", props["AromaticRings"])

    st.markdown('<div style="color:#8b93a1; font-size:0.8rem; font-weight:600; letter-spacing:.3px; text-transform:uppercase; margin:24px 0 8px;">Pharmacophore Features</div>', unsafe_allow_html=True)
    features = analysis["features"]
    if features:
        for feat_name, feat_data in features.items():
            st.markdown(f"""<div class="card" style="padding:12px 16px;">
            <b style="color:#4fd1c5;">{feat_name}</b>
            <span style="color:#8b93a1; font-size:0.82rem;"> — {feat_data['count']} atom(s)</span><br>
            <span style="color:#8b93a1; font-size:0.72rem;">Atom indices: {feat_data['atoms']}</span>
            </div>""", unsafe_allow_html=True)
    else:
        st.info("No pharmacophore features detected.")

    st.markdown('<div style="color:#8b93a1; font-size:0.8rem; font-weight:600; letter-spacing:.3px; text-transform:uppercase; margin:24px 0 8px;">Modification Suggestions for Derivatives</div>', unsafe_allow_html=True)
    for i, sugg in enumerate(analysis["suggestions"], 1):
        st.markdown(f"""<div class="card" style="padding:12px 16px;">
        <b style="color:#63b3ed;">{i}.</b> {sugg}
        </div>""", unsafe_allow_html=True)

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["Feature", "Count", "Atom Indices"])
    for feat_name, feat_data in features.items():
        writer.writerow([feat_name, feat_data["count"], str(feat_data["atoms"])])
    writer.writerow([])
    writer.writerow(["Property", "Value"])
    for k, v in props.items():
        writer.writerow([k, v])
    writer.writerow([])
    writer.writerow(["Suggestion"])
    for s in analysis["suggestions"]:
        writer.writerow([s])
    st.download_button("📥  Export analysis as CSV", data=buf.getvalue(), file_name="inhibitor_analysis.csv", mime="text/csv", use_container_width=True)

# ===========================================================
# SECTION 3 — Molecule Editor
# ===========================================================
st.markdown('<div class="section-head"><div class="num">3</div><div><div class="title">Molecule Editor</div><div class="desc">Draw or paste a SMILES — molecular properties update in real time.</div></div></div>', unsafe_allow_html=True)

if st.session_state.get("_sync_text_input", False):
    st.session_state["smiles_input"] = st.session_state.smiles
    st.session_state["_sync_text_input"] = False

left, right = st.columns([1.4, 1])
with left:
    st.text_input("SMILES", key="smiles_input", placeholder="Paste SMILES or draw on the canvas below", label_visibility="collapsed")
    st.session_state.smiles = st.session_state.smiles_input
    new_smiles = st_ketcher(st.session_state.smiles, height=500)
    if new_smiles != st.session_state.smiles:
        st.session_state.smiles = new_smiles
        st.session_state["_sync_text_input"] = True
        save_data()
        st.rerun()

smiles = st.session_state.smiles
with right:
    st.markdown('<div style="color:#8b93a1; font-size:0.82rem; font-weight:600; letter-spacing:.3px; text-transform:uppercase; margin-bottom:10px;">Compound Preview</div>', unsafe_allow_html=True)
    if not smiles:
        st.info("Draw a structure, enter a SMILES, or send a compound from section 2.")
    else:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            st.error("Couldn't parse this structure — try adjusting the drawing or SMILES.")
        else:
            AllChem.Compute2DCoords(mol)
            drawer = rdMolDraw2D.MolDraw2DCairo(600, 500)
            opts = drawer.drawOptions()
            opts.minFontSize = 22
            opts.maxFontSize = 32
            opts.bondLineWidth = 3
            opts.addStereoAnnotation = True
            opts.padding = 0.18
            rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
            drawer.FinishDrawing()
            st.image(drawer.GetDrawingText(), use_container_width=True)

if smiles and Chem.MolFromSmiles(smiles) is not None:
    mol = Chem.MolFromSmiles(smiles)
    mw = Descriptors.MolWt(mol)
    logp = Crippen.MolLogP(mol)
    hbd = Lipinski.NumHDonors(mol)
    hba = Lipinski.NumHAcceptors(mol)
    tpsa = Descriptors.TPSA(mol)
    rot_bonds = Descriptors.NumRotatableBonds(mol)
    st.markdown("<br>", unsafe_allow_html=True)
    m1, m2, m3 = st.columns(3)
    m1.metric("Molecular Weight", f"{mw:.1f} g/mol")
    m2.metric("LogP", f"{logp:.2f}")
    m3.metric("TPSA", f"{tpsa:.1f} Å²")
    m4, m5, m6 = st.columns(3)
    m4.metric("H-Bond Donors", hbd)
    m5.metric("H-Bond Acceptors", hba)
    m6.metric("Rotatable Bonds", rot_bonds)
    violations = sum([mw > 500, logp > 5, hbd > 5, hba > 10])
    if violations == 0:
        st.success("✅ Passes Lipinski's Rule of Five")
    else:
        st.warning(f"⚠️ Violates {violations} of Lipinski's Rule of Five — may affect oral absorption")
    st.markdown("<br>", unsafe_allow_html=True)
    col_add1, col_add2 = st.columns([1, 3])
    with col_add1:
        if st.button("➕  Add to My Compounds", use_container_width=True, key="add_to_my_compounds"):
            st.session_state.my_compounds.append({"name": f"Compound {len(st.session_state.my_compounds) + 1}", "smiles": smiles, "percentage": 0.0, "notes": ""})
            save_data()
            st.success(f"Added! You now have {len(st.session_state.my_compounds)} compound(s).")
    with col_add2:
        st.caption("Save this compound to your personal library at the bottom of the page.")

# ===========================================================
# SECTION 4 — Target Prediction (SwissTargetPrediction)
# ===========================================================
st.markdown('<div class="section-head"><div class="num">4</div><div><div class="title">Target Prediction</div><div class="desc">SwissTargetPrediction embedded — open it only when you are ready to run the prediction.</div></div></div>', unsafe_allow_html=True)
saved = st.session_state.get("swiss_query_smiles") or st.session_state.get("smiles", "")
if saved:
    st.markdown(f"""<div class="card"><span style="color:#8b93a1; font-size:0.8rem;">Your current SMILES — copy this into SwissTargetPrediction:</span><br><code>{saved}</code></div>""", unsafe_allow_html=True)
if "show_swiss" not in st.session_state:
    st.session_state.show_swiss = False
if not st.session_state.show_swiss:
    if st.button("🎯  Open SwissTargetPrediction", use_container_width=True, key="open_swiss"):
        st.session_state.show_swiss = True
        st.rerun()
else:
    if st.button("✕  Close SwissTargetPrediction", use_container_width=True, key="close_swiss"):
        st.session_state.show_swiss = False
        st.rerun()
    components.iframe("https://www.swisstargetprediction.ch/", height=900, scrolling=True)

# ===========================================================
# SECTION 5 — 3D Visualization (MolView)
# ===========================================================
st.markdown('<div class="section-head"><div class="num">5</div><div><div class="title">3D Visualization</div><div class="desc">MolView embedded — explore molecular structures in 2D and 3D interactively.</div></div></div>', unsafe_allow_html=True)
st.markdown("""<div class="card"><b>MolView</b> — interactive molecular viewer.<br>
<span style="color:#8b93a1; font-size:0.82rem;">Open the embedded viewer below to search by name, draw a structure, or paste a SMILES, then switch between 2D and 3D rendering modes.</span></div>""", unsafe_allow_html=True)
components.iframe("https://app.molview.com/", height=800, scrolling=True)

# ===========================================================
# SECTION 6 — My Compounds
# ===========================================================
st.markdown('<div class="section-head"><div class="num">6</div><div><div class="title">My Compounds</div><div class="desc">Your personal library — set a percentage for each compound and export the list.</div></div></div>', unsafe_allow_html=True)
if not st.session_state.my_compounds:
    st.info("No compounds added yet. Go to Section 3 (Molecule Editor), draw a molecule, and click **➕ Add to My Compounds**.")
else:
    total = len(st.session_state.my_compounds)
    total_pct = sum(c.get("percentage", 0) or 0 for c in st.session_state.my_compounds)
    sum1, sum2, sum3 = st.columns(3)
    sum1.metric("Compounds", total)
    sum2.metric("Total %", f"{total_pct:.1f}%")
    sum3.metric("Remaining to 100%", f"{max(0, 100 - total_pct):.1f}%")
    if total_pct > 100:
        st.warning(f"⚠️ Total exceeds 100% by {total_pct - 100:.1f}%.")
    elif total_pct == 100:
        st.success("✅ Total is exactly 100%.")
    st.markdown("<br>", unsafe_allow_html=True)
    for i, comp in enumerate(st.session_state.my_compounds):
        st.markdown(f"""<div style="background:#12151c; border:1px solid #1e212b; border-radius:14px; padding:16px 18px 4px; margin-bottom:4px;">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
        <span style="color:#4fd1c5; font-weight:700; font-size:0.85rem; letter-spacing:.4px;">COMPOUND #{i+1}</span>
        <span style="color:#8b93a1; font-size:0.72rem;">{comp['smiles'][:60]}{'...' if len(comp['smiles']) > 60 else ''}</span></div></div>""", unsafe_allow_html=True)
        row1, row2, row3 = st.columns([2, 1.5, 1.5])
        with row1:
            new_name = st.text_input("Name", value=comp["name"], key=f"name_{i}", label_visibility="collapsed", placeholder="Compound name")
            if new_name != comp["name"]:
                st.session_state.my_compounds[i]["name"] = new_name
                save_data()
        with row2:
            new_pct = st.number_input("Percentage (%)", min_value=0.0, max_value=100.0, value=float(comp.get("percentage", 0.0) or 0.0), step=0.5, key=f"pct_{i}", label_visibility="collapsed")
            if new_pct != comp.get("percentage"):
                st.session_state.my_compounds[i]["percentage"] = new_pct
                save_data()
        with row3:
            if st.button("🗑️  Remove", key=f"del_{i}", use_container_width=True):
                st.session_state.my_compounds.pop(i)
                save_data()
                st.rerun()
        with st.expander("📝 Notes & SMILES"):
            new_notes = st.text_area("Notes", value=comp.get("notes", ""), key=f"notes_{i}", placeholder="Notes...", height=80)
            if new_notes != comp.get("notes", ""):
                st.session_state.my_compounds[i]["notes"] = new_notes
                save_data()
            st.code(comp["smiles"], language=None)
        st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    act1, act2, act3 = st.columns(3)
    with act1:
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(["Name", "SMILES", "Percentage (%)", "Notes"])
        for c in st.session_state.my_compounds:
            writer.writerow([c["name"], c["smiles"], c["percentage"], c.get("notes", "")])
        st.download_button("📥  Export as CSV", data=buf.getvalue(), file_name="my_compounds.csv", mime="text/csv", use_container_width=True)
    with act2:
        if st.button("🗑️  Clear All", use_container_width=True, key="clear_all_compounds"):
            st.session_state.my_compounds = []
            save_data()
            st.rerun()
    with act3:
        if st.button("⚖️  Normalize to 100%", use_container_width=True, key="normalize"):
            total_now = sum(c.get("percentage", 0) or 0 for c in st.session_state.my_compounds)
            if total_now > 0:
                for c in st.session_state.my_compounds:
                    c["percentage"] = round((c.get("percentage", 0) or 0) / total_now * 100, 2)
                save_data()
                st.rerun()
            else:
                st.warning("Set at least one percentage above 0 first.")
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div style="color:#8b93a1; font-size:0.8rem; font-weight:600; letter-spacing:.3px; text-transform:uppercase; margin-bottom:10px;">Composition Overview</div>', unsafe_allow_html=True)
    total_pct_now = sum(c.get("percentage", 0) or 0 for c in st.session_state.my_compounds)
    if total_pct_now > 0:
        colors = ["#4fd1c5", "#63b3ed", "#a78bfa", "#f6ad55", "#fc8181", "#68d391", "#f6e05e", "#ed64a6"]
        bar_html = '<div style="display:flex; width:100%; height:38px; border-radius:10px; overflow:hidden; border:1px solid #1e212b;">'
        for idx, c in enumerate(st.session_state.my_compounds):
            pct = c.get("percentage", 0) or 0
            if pct <= 0:
                continue
            color = colors[idx % len(colors)]
            bar_html += f'<div style="width:{pct}%; background:{color}; display:flex; align-items:center; justify-content:center; color:#0b0d12; font-weight:700; font-size:0.75rem;">{pct:.1f}%</div>'
        bar_html += "</div>"
        st.markdown(bar_html, unsafe_allow_html=True)
    else:
        st.caption("Set percentages above to see the composition bar.")

# ===========================================================
# SECTION 7 — Scaffold Data Entry
# ===========================================================
st.markdown('<div class="section-head"><div class="num">7</div><div><div class="title">Scaffold Data Entry</div><div class="desc">Record modifications and targetability scores, then export as PDF.</div></div></div>', unsafe_allow_html=True)
st.markdown('<div style="color:#8b93a1; font-size:0.8rem; font-weight:600; letter-spacing:.3px; text-transform:uppercase; margin-bottom:8px;">Header Information</div>', unsafe_allow_html=True)
h1, h2, h3 = st.columns([1.2, 3, 1.2])
with h1:
    new_title = st.text_input("Scaffold title", value=st.session_state.scaffold_header["title"], key="scaffold_title_input", placeholder="e.g. SCAFFOLD 12")
    if new_title != st.session_state.scaffold_header["title"]:
        st.session_state.scaffold_header["title"] = new_title
        save_data()
with h2:
    new_orig = st.text_input("Original SMILES", value=st.session_state.scaffold_header["original_smiles"], key="scaffold_orig_input", placeholder="Original SMILES")
    if new_orig != st.session_state.scaffold_header["original_smiles"]:
        st.session_state.scaffold_header["original_smiles"] = new_orig
        save_data()
with h3:
    new_targ = st.text_input("Targetability", value=st.session_state.scaffold_header["targetability"], key="scaffold_targ_input", placeholder="e.g. 60%")
    if new_targ != st.session_state.scaffold_header["targetability"]:
        st.session_state.scaffold_header["targetability"] = new_targ
        save_data()
st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div style="color:#8b93a1; font-size:0.8rem; font-weight:600; letter-spacing:.3px; text-transform:uppercase; margin-bottom:8px;">Add New Entry</div>', unsafe_allow_html=True)
with st.form("scaffold_entry_form", clear_on_submit=True):
    c1, c2, c3, c4, c5 = st.columns([1, 2, 5, 1, 1])
    with c1:
        new_d = st.text_input("D", placeholder="D111", label_visibility="collapsed")
    with c2:
        new_mod = st.text_input("Modification", placeholder="Modification", label_visibility="collapsed")
    with c3:
        new_smi = st.text_input("SMILES", placeholder="SMILES string", label_visibility="collapsed")
    with c4:
        new_pct = st.text_input("%", placeholder="%", label_visibility="collapsed")
    with c5:
        submitted = st.form_submit_button("➕ Add", use_container_width=True)
    if submitted:
        if new_d.strip() or new_smi.strip():
            st.session_state.scaffold_entries.append({"D": new_d.strip(), "Modification": new_mod.strip(), "SMILES": new_smi.strip(), "Percentage": new_pct.strip()})
            save_data()
            st.rerun()
        else:
            st.warning("Enter at least a D code or a SMILES.")
if st.session_state.scaffold_entries:
    st.markdown("<br>", unsafe_allow_html=True)
    th1, th2, th3, th4, th5 = st.columns([1, 2, 5, 1, 0.7])
    with th1:
        st.markdown('<div style="color:#4fd1c5; font-weight:700; font-size:0.78rem;">D</div>', unsafe_allow_html=True)
    with th2:
        st.markdown('<div style="color:#4fd1c5; font-weight:700; font-size:0.78rem;">MODIFICATION</div>', unsafe_allow_html=True)
    with th3:
        st.markdown('<div style="color:#4fd1c5; font-weight:700; font-size:0.78rem;">SMILES</div>', unsafe_allow_html=True)
    with th4:
        st.markdown('<div style="color:#4fd1c5; font-weight:700; font-size:0.78rem;">%</div>', unsafe_allow_html=True)
    with th5:
        st.markdown("", unsafe_allow_html=True)
    st.markdown('<hr style="margin:6px 0 12px 0; border-color:#1e212b;">', unsafe_allow_html=True)
    for i, entry in enumerate(st.session_state.scaffold_entries):
        r1, r2, r3, r4, r5 = st.columns([1, 2, 5, 1, 0.7])
        with r1:
            st.markdown(f'<div style="padding:8px 0; color:#e2e5ea; font-size:0.85rem;">{entry["D"]}</div>', unsafe_allow_html=True)
        with r2:
            st.markdown(f'<div style="padding:8px 0; color:#8b93a1; font-size:0.82rem;">{entry["Modification"] or "—"}</div>', unsafe_allow_html=True)
        with r3:
            smi = entry["SMILES"]
            short = smi[:70] + "..." if len(smi) > 70 else smi
            st.markdown(f'<div style="padding:8px 0; color:#4fd1c5; font-size:0.78rem; font-family:monospace; word-break:break-all;" title="{smi}">{short}</div>', unsafe_allow_html=True)
        with r4:
            st.markdown(f'<div style="padding:8px 0; color:#e2e5ea; font-size:0.85rem; font-weight:600;">{entry["Percentage"]}</div>', unsafe_allow_html=True)
        with r5:
            if st.button("✕", key=f"del_scaffold_{i}", use_container_width=True):
                st.session_state.scaffold_entries.pop(i)
                save_data()
                st.rerun()
    st.markdown("<br>", unsafe_allow_html=True)
    exp1, exp2, exp3 = st.columns(3)
    with exp1:
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(["D", "Modification", "SMILES", "%"])
        for e in st.session_state.scaffold_entries:
            writer.writerow([e["D"], e["Modification"], e["SMILES"], e["Percentage"]])
        st.download_button("📥  Export as CSV", data=buf.getvalue(), file_name=f"{st.session_state.scaffold_header['title'] or 'scaffold'}.csv", mime="text/csv", use_container_width=True)
    with exp2:
        def build_pdf():
            pdf = FPDF(orientation="P", unit="mm", format="A4")
            pdf.add_page()
            pdf.set_auto_page_break(auto=True, margin=15)
            pdf.set_font("Helvetica", "B", 14)
            pdf.cell(0, 10, st.session_state.scaffold_header["title"] or "SCAFFOLD", ln=True, align="C")
            pdf.ln(2)
            if st.session_state.scaffold_header["original_smiles"]:
                pdf.set_font("Helvetica", "B", 10)
                pdf.cell(38, 7, "ORIGINAL SMILES:", border=0)
                pdf.set_font("Courier", "", 9)
                pdf.multi_cell(0, 5, st.session_state.scaffold_header["original_smiles"])
                pdf.ln(2)
            if st.session_state.scaffold_header["targetability"]:
                pdf.set_font("Helvetica", "B", 10)
                pdf.cell(38, 7, "TARGETABILITY:", border=0)
                pdf.set_font("Helvetica", "", 10)
                pdf.cell(0, 7, st.session_state.scaffold_header["targetability"], ln=True, align="C")
                pdf.ln(4)
            pdf.set_font("Helvetica", "B", 9)
            pdf.set_fill_color(230, 230, 230)
            col_widths = [15, 30, 120, 15]
            headers = ["D", "MODIFICATION", "SMILES", "%"]
            for w, h in zip(col_widths, headers):
                pdf.cell(w, 8, h, border=1, align="C", fill=True)
            pdf.ln()
            pdf.set_font("Courier", "", 7)
            chars_per_line = 60
            for e in st.session_state.scaffold_entries:
                row = [e["D"], e["Modification"], e["SMILES"], e["Percentage"]]
                smi = row[2]
                lines = [smi[i:i+chars_per_line] for i in range(0, len(smi), chars_per_line)] or [""]
                row_height = 6 * len(lines)
                if pdf.get_y() + row_height > 270:
                    pdf.add_page()
                    pdf.set_font("Helvetica", "B", 9)
                    for w, h in zip(col_widths, headers):
                        pdf.cell(w, 8, h, border=1, align="C", fill=True)
                    pdf.ln()
                    pdf.set_font("Courier", "", 7)
                pdf.cell(col_widths[0], row_height, row[0], border=1, align="C")
                pdf.cell(col_widths[1], row_height, row[1][:20], border=1, align="L")
                x_after_mod = pdf.get_x()
                pdf.set_xy(x_after_mod, pdf.get_y())
                for ln_idx, line in enumerate(lines):
                    pdf.cell(col_widths[2], 6, line, border=0, align="L")
                    if ln_idx < len(lines) - 1:
                        pdf.ln(6)
                        pdf.set_x(x_after_mod)
                pdf.set_xy(x_after_mod + col_widths[2], pdf.get_y() - 6 * len(lines))
                pdf.rect(x_after_mod, pdf.get_y(), col_widths[2], row_height)
                pdf.cell(col_widths[3], row_height, row[3], border=1, align="C")
                pdf.ln(row_height)
            return bytes(pdf.output())
        try:
            pdf_bytes = build_pdf()
            st.download_button("📄  Export as PDF", data=pdf_bytes, file_name=f"{st.session_state.scaffold_header['title'] or 'scaffold'}.pdf", mime="application/pdf", use_container_width=True)
        except Exception as e:
            st.error(f"PDF error: {e}")
    with exp3:
        if st.button("🗑️  Clear All Entries", use_container_width=True, key="clear_scaffold"):
            st.session_state.scaffold_entries = []
            save_data()
            st.rerun()
else:
    st.info("No entries yet — fill in the fields above and click **➕ Add**.")

# ---------------------------------------------------------
# Auto-save + Footer
# ---------------------------------------------------------
save_data()
st.markdown("""<div class="footer">Integrated computational drug-discovery workspace · Data from
<a href="https://pubmed.ncbi.nlm.nih.gov/" target="_blank">PubMed</a>,
<a href="https://www.ebi.ac.uk/chembl/" target="_blank">ChEMBL</a>,
<a href="https://platform.opentargets.org/" target="_blank">Open Targets</a>,
<a href="https://www.swisstargetprediction.ch/" target="_blank">SwissTargetPrediction</a>, and
<a href="https://molview.org/" target="_blank">MolView</a>.</div>""", unsafe_allow_html=True)
