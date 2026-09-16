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
import requests
import streamlit as st
import streamlit.components.v1 as components
from streamlit_ketcher import st_ketcher
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski, AllChem
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
# Global styling — unified dark professional theme
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    /* ---------- Base ---------- */
    .stApp { background: #0b0d12; color: #e2e5ea; }
    .block-container { max-width: 1400px; padding: 0 2rem 4rem; }

    /* ---------- Hide sidebar ---------- */
    section[data-testid="stSidebar"] { display: none; }

    /* ---------- Top navigation bar ---------- */
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
    .topnav .nav-links {
        display: flex; gap: 26px; color: #8b93a1;
        font-size: 0.85rem; font-weight: 500;
    }
    .topnav .nav-links span:hover { color: #4fd1c5; cursor: pointer; }

    /* ---------- Hero ---------- */
    .hero {
        background: radial-gradient(circle at 20% 0%, #16233a 0%, #0f141c 45%, #0b0d12 100%);
        border: 1px solid #1e212b; border-radius: 20px;
        padding: 52px 48px 48px; margin-bottom: 42px;
        position: relative; overflow: hidden;
    }
    .hero::before {
        content: ""; position: absolute; top: -80px; right: -80px;
        width: 320px; height: 320px;
        background: radial-gradient(circle, rgba(79,209,197,.14), transparent 70%);
        pointer-events: none;
    }
    .hero .badge {
        display: inline-block; padding: 5px 12px; border-radius: 999px;
        background: rgba(79,209,197,.1); border: 1px solid rgba(79,209,197,.25);
        color: #4fd1c5; font-size: 0.72rem; font-weight: 600;
        letter-spacing: .6px; text-transform: uppercase; margin-bottom: 18px;
    }
    .hero h1 {
        margin: 0 0 14px; font-size: 2.6rem; font-weight: 700;
        line-height: 1.15; letter-spacing: -1px; color: #f1f5f9;
    }
    .hero h1 span {
        background: linear-gradient(90deg, #4fd1c5, #63b3ed);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .hero p { margin: 0; color: #9aa1ac; font-size: 1rem; max-width: 640px; line-height: 1.6; }

    /* ---------- Section headers ---------- */
    .section-head { display: flex; align-items: center; gap: 14px; margin: 52px 0 22px; }
    .section-head .num {
        width: 36px; height: 36px; border-radius: 10px;
        background: linear-gradient(135deg, #4fd1c5, #63b3ed);
        color: #0b0d12; font-weight: 800; font-size: 0.95rem;
        display: flex; align-items: center; justify-content: center; flex-shrink: 0;
    }
    .section-head .title { font-size: 1.45rem; font-weight: 700; color: #e2e5ea; letter-spacing: -.3px; }
    .section-head .desc { color: #8b93a1; font-size: 0.88rem; margin-top: 2px; }

    /* ---------- Cards ---------- */
    .card {
        background: #12151c; border: 1px solid #1e212b; border-radius: 14px;
        padding: 18px 20px; margin-bottom: 12px;
        transition: border-color .15s ease, background .15s ease;
    }
    .card:hover { border-color: #2a3140; background: #141822; }
    .card b { color: #e2e5ea; }
    .card a { color: #63b3ed; text-decoration: none; }
    .card a:hover { text-decoration: underline; }
    .card code {
        background: #0b0d12; color: #4fd1c5; padding: 3px 8px;
        border-radius: 6px; font-size: 0.8rem; word-break: break-all;
        display: inline-block; margin-top: 6px;
    }

    /* ---------- Feature strip ---------- */
    .features { display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; margin-bottom: 8px; }
    .feature { background: #12151c; border: 1px solid #1e212b; border-radius: 12px; padding: 18px; }
    .feature .ic { font-size: 1.4rem; margin-bottom: 8px; }
    .feature .ttl { font-weight: 650; color: #e2e5ea; font-size: 0.92rem; margin-bottom: 4px; }
    .feature .txt { color: #8b93a1; font-size: 0.78rem; line-height: 1.5; }

    /* ---------- Metrics ---------- */
    div[data-testid="stMetric"] {
        background: #12151c; border: 1px solid #1e212b;
        border-radius: 12px; padding: 14px 18px;
    }
    div[data-testid="stMetricLabel"] { color: #8b93a1 !important; font-size: 0.8rem !important; }
    div[data-testid="stMetricValue"] { color: #e2e5ea !important; font-size: 1.3rem !important; }

    /* ---------- Buttons ---------- */
    div[data-testid="stButton"] > button {
        border-radius: 10px; font-weight: 600; min-height: 42px;
        background: #1a1d26; color: #e2e5ea; border: 1px solid #2a3140;
        transition: all .15s ease;
    }
    div[data-testid="stButton"] > button:hover {
        background: #202531; border-color: #4fd1c5; color: #4fd1c5;
    }
    div[data-testid="stLinkButton"] > a {
        border-radius: 10px !important; font-weight: 600 !important;
        background: #1a1d26 !important; color: #e2e5ea !important;
        border: 1px solid #2a3140 !important; text-decoration: none !important;
    }
    div[data-testid="stLinkButton"] > a:hover { border-color: #4fd1c5 !important; color: #4fd1c5 !important; }
    div[data-testid="stDownloadButton"] > button {
        border-radius: 10px; font-weight: 600; min-height: 42px;
        background: #1a1d26; color: #e2e5ea; border: 1px solid #2a3140;
    }
    div[data-testid="stDownloadButton"] > button:hover { border-color: #4fd1c5; color: #4fd1c5; }

    /* ---------- Form submit button ---------- */
    div[data-testid="stFormSubmitButton"] > button {
        border-radius: 10px; font-weight: 600; min-height: 42px;
        background: #1a1d26; color: #e2e5ea; border: 1px solid #2a3140;
        transition: all .15s ease;
    }
    div[data-testid="stFormSubmitButton"] > button:hover {
        background: #202531; border-color: #4fd1c5; color: #4fd1c5;
    }

    /* ---------- Inputs ---------- */
    div[data-testid="stTextInput"] input,
    div[data-testid="stTextArea"] textarea,
    div[data-testid="stNumberInput"] input {
        background: #12151c; color: #e2e5ea;
        border: 1px solid #2a3140; border-radius: 10px;
    }

    /* ---------- Expander ---------- */
    details { background: #12151c; border: 1px solid #1e212b; border-radius: 10px; }

    /* ---------- Iframe ---------- */
    iframe { border-radius: 14px; border: 1px solid #1e212b; background: #12151c; }

    /* ---------- Divider ---------- */
    hr { border-color: #1e212b; margin: 42px 0; }

    /* ---------- Alerts ---------- */
    div[data-testid="stAlert"] { border-radius: 10px; border: 1px solid #1e212b; background: #12151c; }

    /* ---------- Footer ---------- */
    .footer {
        text-align: center; color: #565e6d; font-size: 0.78rem;
        padding: 26px 0 6px; border-top: 1px solid #1e212b; margin-top: 60px;
    }
    .footer a { color: #63b3ed; text-decoration: none; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------
if "smiles" not in st.session_state:
    st.session_state.smiles = ""
if "_sync_text_input" not in st.session_state:
    st.session_state["_sync_text_input"] = False
if "my_compounds" not in st.session_state:
    st.session_state.my_compounds = []
if "scaffold_entries" not in st.session_state:
    st.session_state.scaffold_entries = []
if "scaffold_header" not in st.session_state:
    st.session_state.scaffold_header = {
        "title": "SCAFFOLD 12",
        "original_smiles": "",
        "targetability": "",
    }

# ---------------------------------------------------------
# Top navigation bar
# ---------------------------------------------------------
st.markdown(
    """
    <div class="topnav">
        <div class="logo">🧬 Discovery Assistant</div>
        <div class="nav-links">
            <span>Overview</span>
            <span>Diseases</span>
            <span>Compounds</span>
            <span>Editor</span>
            <span>Prediction</span>
            <span>Visualization</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Hero
# ---------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <div class="badge">Computational Drug Discovery Platform</div>
        <h1>From disease to lead compound,<br><span>all in one workspace.</span></h1>
        <p>Search the literature, retrieve known drugs, draw and analyze molecules, predict protein targets, and visualize 3D structures — without leaving the page.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Feature strip
# ---------------------------------------------------------
st.markdown(
    """
    <div class="features">
        <div class="feature">
            <div class="ic">📚</div>
            <div class="ttl">Literature Search</div>
            <div class="txt">Live PubMed results for emerging diseases across four therapeutic areas.</div>
        </div>
        <div class="feature">
            <div class="ic">💊</div>
            <div class="ttl">Lead Compounds</div>
            <div class="txt">Known drugs and SMILES pulled directly from ChEMBL.</div>
        </div>
        <div class="feature">
            <div class="ic">🎯</div>
            <div class="ttl">Target Prediction</div>
            <div class="txt">SwissTargetPrediction embedded for in-app protein target prediction.</div>
        </div>
        <div class="feature">
            <div class="ic">🔬</div>
            <div class="ttl">3D Visualization</div>
            <div class="txt">MolView embedded for interactive molecular structure exploration.</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ===========================================================
# SECTION 1 — Disease Discovery (PubMed)
# ===========================================================
st.markdown(
    """
    <div class="section-head">
        <div class="num">1</div>
        <div>
            <div class="title">Disease Discovery</div>
            <div class="desc">Search recent PubMed literature for emerging and rare diseases by category.</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

category_map = {
    "Viral": "viral infection",
    "Bacterial": "bacterial infection",
    "Oncology": "cancer OR tumor",
    "Parasitic": "parasitic infection",
}
category = st.radio("Category", list(category_map.keys()), horizontal=True, label_visibility="collapsed")


@st.cache_data(ttl=3600, show_spinner=False)
def search_pubmed_diseases(category_term: str, retmax: int = 8):
    term = f'("rare disease"[tiab] OR "novel disease"[tiab] OR "emerging"[tiab]) AND ({category_term})'
    esearch = requests.get(
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
        params={"db": "pubmed", "term": term, "retmax": retmax, "sort": "date", "retmode": "json"},
        timeout=15,
    ).json()
    ids = esearch.get("esearchresult", {}).get("idlist", [])
    if not ids:
        return []
    esummary = requests.get(
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi",
        params={"db": "pubmed", "id": ",".join(ids), "retmode": "json"},
        timeout=15,
    ).json()
    results = []
    for pmid in ids:
        doc = esummary.get("result", {}).get(pmid, {})
        if doc:
            results.append({
                "pmid": pmid,
                "title": doc.get("title", "Untitled"),
                "pubdate": doc.get("pubdate", ""),
            })
    return results


if st.button("🔍  Search PubMed for candidate diseases", use_container_width=True):
    with st.spinner("Searching PubMed..."):
        st.session_state.disease_results = search_pubmed_diseases(category_map[category])

if st.session_state.get("disease_results"):
    st.caption(f"{len(st.session_state.disease_results)} recent articles")
    for r in st.session_state.disease_results:
        st.markdown(
            f"""<div class="card">
            <b>{r['title']}</b><br>
            <span style="color:#8b93a1; font-size:0.8rem;">PMID {r['pmid']} · {r['pubdate']}</span> ·
            <a href="https://pubmed.ncbi.nlm.nih.gov/{r['pmid']}/" target="_blank">View on PubMed</a>
            </div>""",
            unsafe_allow_html=True,
        )

# ===========================================================
# SECTION 2 — Lead Compound Lookup (ChEMBL)
# ===========================================================
st.markdown(
    """
    <div class="section-head">
        <div class="num">2</div>
        <div>
            <div class="title">Lead Compound Lookup</div>
            <div class="desc">Retrieve known drugs, ChEMBL IDs, and SMILES for a given indication.</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

disease_query = st.text_input(
    "Disease or indication name",
    key="disease_query",
    placeholder="e.g. malaria, tuberculosis, glioblastoma...",
    label_visibility="collapsed",
)


@st.cache_data(ttl=3600, show_spinner=False)
def chembl_lookup(indication: str):
    resp = requests.get(
        "https://www.ebi.ac.uk/chembl/api/data/drug_indication.json",
        params={"search": indication, "limit": 5},
        timeout=15,
    ).json()
    indications = resp.get("drug_indications", [])
    out = []
    for ind in indications:
        molecule_chembl_id = ind.get("molecule_chembl_id")
        if not molecule_chembl_id:
            continue
        mol_resp = requests.get(
            f"https://www.ebi.ac.uk/chembl/api/data/molecule/{molecule_chembl_id}.json",
            timeout=15,
        ).json()
        smiles = (mol_resp.get("molecule_structures") or {}).get("canonical_smiles")
        pref_name = mol_resp.get("pref_name") or molecule_chembl_id
        if smiles:
            out.append({
                "name": pref_name,
                "chembl_id": molecule_chembl_id,
                "smiles": smiles,
                "indication": ind.get("efo_term", indication),
            })
    return out


if st.button("🔗  Retrieve lead compounds from ChEMBL", use_container_width=True) and disease_query:
    with st.spinner("Querying ChEMBL..."):
        st.session_state.compound_results = chembl_lookup(disease_query)

if st.session_state.get("compound_results"):
    for c in st.session_state.compound_results:
        st.markdown(
            f"""<div class="card">
            <b>{c['name']}</b> <span style="color:#8b93a1;">({c['chembl_id']})</span><br>
            <span style="color:#8b93a1; font-size:0.8rem;">Indication: {c['indication']}</span><br>
            <code>{c['smiles']}</code>
            </div>""",
            unsafe_allow_html=True,
        )
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Send to Molecule Editor →", key=f"send_{c['chembl_id']}", use_container_width=True):
                st.session_state.smiles = c["smiles"]
                st.session_state["_sync_text_input"] = True
                st.success("SMILES loaded into the Molecule Editor below.")
        with col_b:
            if st.button("Use for Target Prediction", key=f"swiss_{c['chembl_id']}", use_container_width=True):
                st.session_state["swiss_query_smiles"] = c["smiles"]
                st.success("SMILES saved — see the target prediction section below.")
elif st.session_state.get("compound_results") == []:
    st.warning("No known drugs found — try a broader or alternate name.")

# ===========================================================
# SECTION 3 — Molecule Editor
# ===========================================================
st.markdown(
    """
    <div class="section-head">
        <div class="num">3</div>
        <div>
            <div class="title">Molecule Editor</div>
            <div class="desc">Draw or paste a SMILES — molecular properties update in real time.</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if st.session_state.get("_sync_text_input", False):
    st.session_state["smiles_input"] = st.session_state.smiles
    st.session_state["_sync_text_input"] = False

left, right = st.columns([1.4, 1])

with left:
    st.text_input(
        "SMILES",
        key="smiles_input",
        placeholder="Paste SMILES or draw on the canvas below",
        label_visibility="collapsed",
    )
    st.session_state.smiles = st.session_state.smiles_input

    new_smiles = st_ketcher(st.session_state.smiles, height=500)

    if new_smiles != st.session_state.smiles:
        st.session_state.smiles = new_smiles
        st.session_state["_sync_text_input"] = True
        st.rerun()

smiles = st.session_state.smiles

with right:
    st.markdown(
        '<div style="color:#8b93a1; font-size:0.82rem; font-weight:600; '
        'letter-spacing:.3px; text-transform:uppercase; margin-bottom:10px;">Compound Preview</div>',
        unsafe_allow_html=True,
    )
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

# ---------- Properties + Add to My Compounds ----------
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
            st.session_state.my_compounds.append({
                "name": f"Compound {len(st.session_state.my_compounds) + 1}",
                "smiles": smiles,
                "percentage": 0.0,
                "notes": "",
            })
            st.success(f"Added! You now have {len(st.session_state.my_compounds)} compound(s).")
    with col_add2:
        st.caption("Save this compound to your personal library at the bottom of the page, where you can set its percentage.")

# ===========================================================
# SECTION 4 — Target Prediction (SwissTargetPrediction)
# ===========================================================
st.markdown(
    """
    <div class="section-head">
        <div class="num">4</div>
        <div>
            <div class="title">Target Prediction</div>
            <div class="desc">SwissTargetPrediction embedded — open it only when you are ready to run the prediction.</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

saved = st.session_state.get("swiss_query_smiles") or st.session_state.get("smiles", "")

if saved:
    st.markdown(
        f"""<div class="card">
        <span style="color:#8b93a1; font-size:0.8rem;">Your current SMILES — copy this into SwissTargetPrediction:</span><br>
        <code>{saved}</code>
        </div>""",
        unsafe_allow_html=True,
    )

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

    components.iframe(
        "https://www.swisstargetprediction.ch/",
        height=900,
        scrolling=True,
    )

# ===========================================================
# SECTION 5 — 3D Visualization (MolView)
# ===========================================================
st.markdown(
    """
    <div class="section-head">
        <div class="num">5</div>
        <div>
            <div class="title">3D Visualization</div>
            <div class="desc">MolView embedded — explore molecular structures in 2D and 3D interactively.</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="card">
        <b>MolView</b> — interactive molecular viewer.<br>
        <span style="color:#8b93a1; font-size:0.82rem;">
        Open the embedded viewer below to search by name, draw a structure, or paste a SMILES,
        then switch between 2D and 3D rendering modes.
        </span>
    </div>
    """,
    unsafe_allow_html=True,
)

components.iframe(
    "https://app.molview.com/",
    height=800,
    scrolling=True,
)

# ===========================================================
# SECTION 6 — My Compounds Library
# ===========================================================
st.markdown(
    """
    <div class="section-head">
        <div class="num">6</div>
        <div>
            <div class="title">My Compounds</div>
            <div class="desc">Your personal library — set a percentage for each compound and export the list.</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if not st.session_state.my_compounds:
    st.info("No compounds added yet. Go to Section 3 (Molecule Editor), draw a molecule, and click **➕ Add to My Compounds**.")
else:
    total = len(st.session_state.my_compounds)
    total_pct = sum(c.get("percentage", 0) or 0 for c in st.session_state.my_compounds)

    sum1, sum2, sum3 = st.columns(3)
    sum1.metric("Compounds", total)
    sum2.metric("Total %", f"{total_pct:.1f}%")
    remaining = max(0, 100 - total_pct)
    sum3.metric("Remaining to 100%", f"{remaining:.1f}%")

    if total_pct > 100:
        st.warning(f"⚠️ Total exceeds 100% by {total_pct - 100:.1f}%.")
    elif total_pct == 100:
        st.success("✅ Total is exactly 100%.")

    st.markdown("<br>", unsafe_allow_html=True)

    for i, comp in enumerate(st.session_state.my_compounds):
        st.markdown(
            f"""<div style="background:#12151c; border:1px solid #1e212b; border-radius:14px;
            padding:16px 18px 4px; margin-bottom:4px;">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                <span style="color:#4fd1c5; font-weight:700; font-size:0.85rem; letter-spacing:.4px;">
                    COMPOUND #{i+1}
                </span>
                <span style="color:#8b93a1; font-size:0.72rem;">
                    {comp['smiles'][:60]}{'...' if len(comp['smiles']) > 60 else ''}
                </span>
            </div>
            </div>""",
            unsafe_allow_html=True,
        )

        row1, row2, row3 = st.columns([2, 1.5, 1.5])
        with row1:
            new_name = st.text_input(
                "Name",
                value=comp["name"],
                key=f"name_{i}",
                label_visibility="collapsed",
                placeholder="Compound name",
            )
            st.session_state.my_compounds[i]["name"] = new_name
        with row2:
            new_pct = st.number_input(
                "Percentage (%)",
                min_value=0.0,
                max_value=100.0,
                value=float(comp.get("percentage", 0.0) or 0.0),
                step=0.5,
                key=f"pct_{i}",
                label_visibility="collapsed",
            )
            st.session_state.my_compounds[i]["percentage"] = new_pct
        with row3:
            if st.button("🗑️  Remove", key=f"del_{i}", use_container_width=True):
                st.session_state.my_compounds.pop(i)
                st.rerun()

        with st.expander("📝 Notes & SMILES"):
            new_notes = st.text_area(
                "Notes",
                value=comp.get("notes", ""),
                key=f"notes_{i}",
                placeholder="e.g. active against Plasmodium falciparum, tested 2024...",
                height=80,
            )
            st.session_state.my_compounds[i]["notes"] = new_notes
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
        csv_data = buf.getvalue()

        st.download_button(
            "📥  Export as CSV",
            data=csv_data,
            file_name="my_compounds.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with act2:
        if st.button("🗑️  Clear All", use_container_width=True, key="clear_all_compounds"):
            st.session_state.my_compounds = []
            st.rerun()

    with act3:
        if st.button("⚖️  Normalize to 100%", use_container_width=True, key="normalize"):
            total_now = sum(c.get("percentage", 0) or 0 for c in st.session_state.my_compounds)
            if total_now > 0:
                for c in st.session_state.my_compounds:
                    c["percentage"] = round((c.get("percentage", 0) or 0) / total_now * 100, 2)
                st.rerun()
            else:
                st.warning("Set at least one percentage above 0 first.")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        '<div style="color:#8b93a1; font-size:0.8rem; font-weight:600; '
        'letter-spacing:.3px; text-transform:uppercase; margin-bottom:10px;">'
        'Composition Overview</div>',
        unsafe_allow_html=True,
    )

    total_pct_now = sum(c.get("percentage", 0) or 0 for c in st.session_state.my_compounds)
    if total_pct_now > 0:
        colors = ["#4fd1c5", "#63b3ed", "#a78bfa", "#f6ad55", "#fc8181", "#68d391", "#f6e05e", "#ed64a6"]
        bar_html = (
            '<div style="display:flex; width:100%; height:38px; border-radius:10px; '
            'overflow:hidden; border:1px solid #1e212b;">'
        )
        for idx, c in enumerate(st.session_state.my_compounds):
            pct = c.get("percentage", 0) or 0
            if pct <= 0:
                continue
            color = colors[idx % len(colors)]
            bar_html += (
                f'<div style="width:{pct}%; background:{color}; display:flex; '
                f'align-items:center; justify-content:center; color:#0b0d12; '
                f'font-weight:700; font-size:0.75rem;" title="{c["name"]}: {pct}%">'
                f'{pct:.1f}%</div>'
            )
        bar_html += "</div>"
        st.markdown(bar_html, unsafe_allow_html=True)
    else:
        st.caption("Set percentages above to see the composition bar.")

# ===========================================================
# SECTION 7 — Scaffold Data Entry
# ===========================================================
st.markdown(
    """
    <div class="section-head">
        <div class="num">7</div>
        <div>
            <div class="title">Scaffold Data Entry</div>
            <div class="desc">Record modifications and targetability scores, then export as PDF.</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------- Header inputs ----------
st.markdown(
    '<div style="color:#8b93a1; font-size:0.8rem; font-weight:600; '
    'letter-spacing:.3px; text-transform:uppercase; margin-bottom:8px;">'
    'Header Information</div>',
    unsafe_allow_html=True,
)
h1, h2, h3 = st.columns([1.2, 3, 1.2])
with h1:
    new_title = st.text_input(
        "Scaffold title",
        value=st.session_state.scaffold_header["title"],
        key="scaffold_title_input",
        placeholder="e.g. SCAFFOLD 12",
    )
    st.session_state.scaffold_header["title"] = new_title
with h2:
    new_orig = st.text_input(
        "Original SMILES",
        value=st.session_state.scaffold_header["original_smiles"],
        key="scaffold_orig_input",
        placeholder="O=C(NC(Cc1ccccc1)C#N)C1CCCN1C(=O)C(NC(=O)C1CC1)C(C)C",
    )
    st.session_state.scaffold_header["original_smiles"] = new_orig
with h3:
    new_targ = st.text_input(
        "Targetability",
        value=st.session_state.scaffold_header["targetability"],
        key="scaffold_targ_input",
        placeholder="e.g. 60%",
    )
    st.session_state.scaffold_header["targetability"] = new_targ

st.markdown("<br>", unsafe_allow_html=True)

# ---------- Add new row form (FIXED with st.form + clear_on_submit) ----------
st.markdown(
    '<div style="color:#8b93a1; font-size:0.8rem; font-weight:600; '
    'letter-spacing:.3px; text-transform:uppercase; margin-bottom:8px;">'
    'Add New Entry</div>',
    unsafe_allow_html=True,
)

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
            st.session_state.scaffold_entries.append({
                "D": new_d.strip(),
                "Modification": new_mod.strip(),
                "SMILES": new_smi.strip(),
                "Percentage": new_pct.strip(),
            })
            st.rerun()
        else:
            st.warning("Enter at least a D code or a SMILES.")

# ---------- Entries table ----------
if st.session_state.scaffold_entries:
    st.markdown("<br>", unsafe_allow_html=True)

    # Header row
    th1, th2, th3, th4, th5 = st.columns([1, 2, 5, 1, 0.7])
    with th1:
        st.markdown('<div style="color:#4fd1c5; font-weight:700; font-size:0.78rem; letter-spacing:.5px;">D</div>', unsafe_allow_html=True)
    with th2:
        st.markdown('<div style="color:#4fd1c5; font-weight:700; font-size:0.78rem; letter-spacing:.5px;">MODIFICATION</div>', unsafe_allow_html=True)
    with th3:
        st.markdown('<div style="color:#4fd1c5; font-weight:700; font-size:0.78rem; letter-spacing:.5px;">SMILES</div>', unsafe_allow_html=True)
    with th4:
        st.markdown('<div style="color:#4fd1c5; font-weight:700; font-size:0.78rem; letter-spacing:.5px;">%</div>', unsafe_allow_html=True)
    with th5:
        st.markdown("", unsafe_allow_html=True)

    st.markdown('<hr style="margin:6px 0 12px 0; border-color:#1e212b;">', unsafe_allow_html=True)

    # Body rows
    for i, entry in enumerate(st.session_state.scaffold_entries):
        r1, r2, r3, r4, r5 = st.columns([1, 2, 5, 1, 0.7])
        with r1:
            st.markdown(
                f'<div style="padding:8px 0; color:#e2e5ea; font-size:0.85rem;">{entry["D"]}</div>',
                unsafe_allow_html=True,
            )
        with r2:
            st.markdown(
                f'<div style="padding:8px 0; color:#8b93a1; font-size:0.82rem;">{entry["Modification"] or "—"}</div>',
                unsafe_allow_html=True,
            )
        with r3:
            smi = entry["SMILES"]
            short = smi[:70] + "..." if len(smi) > 70 else smi
            st.markdown(
                f'<div style="padding:8px 0; color:#4fd1c5; font-size:0.78rem; '
                f'font-family:monospace; word-break:break-all;" title="{smi}">{short}</div>',
                unsafe_allow_html=True,
            )
        with r4:
            st.markdown(
                f'<div style="padding:8px 0; color:#e2e5ea; font-size:0.85rem; font-weight:600;">{entry["Percentage"]}</div>',
                unsafe_allow_html=True,
            )
        with r5:
            if st.button("✕", key=f"del_scaffold_{i}", use_container_width=True):
                st.session_state.scaffold_entries.pop(i)
                st.rerun()

    # ---------- Export buttons ----------
    st.markdown("<br>", unsafe_allow_html=True)
    exp1, exp2, exp3 = st.columns(3)

    with exp1:
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(["D", "Modification", "SMILES", "%"])
        for e in st.session_state.scaffold_entries:
            writer.writerow([e["D"], e["Modification"], e["SMILES"], e["Percentage"]])
        st.download_button(
            "📥  Export as CSV",
            data=buf.getvalue(),
            file_name=f"{st.session_state.scaffold_header['title'] or 'scaffold'}.csv",
            mime="text/csv",
            use_container_width=True,
        )

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

                x0 = pdf.get_x()
                y0 = pdf.get_y()
                pdf.cell(col_widths[0], row_height, row[0], border=1, align="C")
                pdf.cell(col_widths[1], row_height, row[1][:20], border=1, align="L")
                x_after_mod = pdf.get_x()

                pdf.set_xy(x_after_mod, y0)
                for ln_idx, line in enumerate(lines):
                    pdf.cell(col_widths[2], 6, line, border=0, align="L")
                    if ln_idx < len(lines) - 1:
                        pdf.ln(6)
                        pdf.set_x(x_after_mod)
                pdf.set_xy(x_after_mod + col_widths[2], y0)
                pdf.rect(x_after_mod, y0, col_widths[2], row_height)
                pdf.cell(col_widths[3], row_height, row[3], border=1, align="C")
                pdf.ln(row_height)

            return bytes(pdf.output())

        try:
            pdf_bytes = build_pdf()
            st.download_button(
                "📄  Export as PDF",
                data=pdf_bytes,
                file_name=f"{st.session_state.scaffold_header['title'] or 'scaffold'}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        except Exception as e:
            st.error(f"PDF error: {e}")

    with exp3:
        if st.button("🗑️  Clear All Entries", use_container_width=True, key="clear_scaffold"):
            st.session_state.scaffold_entries = []
            st.rerun()

else:
    st.info("No entries yet — fill in the fields above and click **➕ Add**.")

# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.markdown(
    """
    <div class="footer">
        Integrated computational drug-discovery workspace ·
        Data from <a href="https://pubmed.ncbi.nlm.nih.gov/" target="_blank">PubMed</a>,
        <a href="https://www.ebi.ac.uk/chembl/" target="_blank">ChEMBL</a>,
        <a href="https://www.swisstargetprediction.ch/" target="_blank">SwissTargetPrediction</a>,
        and <a href="https://molview.org/" target="_blank">MolView</a>.
    </div>
    """,
    unsafe_allow_html=True,
)
