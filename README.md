# AIcrete Solutions — Deployment Package

## Files to include in your GitHub repo

| File | Purpose |
|---|---|
| `app_merged_v2_v3.py` | Main Streamlit app |
| `model.pkl` | Compressive Strength RF model (V1, R²=0.98) |
| `model_Slump_Flow_mm.pkl` | Slump Flow model (V2, R²=0.85, with cement type) |
| `model_Flexural_MOR_MPa.pkl` | Flexural MOR model (V3, R²=0.84) |
| `model_Peak_Flexural_MPa.pkl` | Peak Flexural model (V3, R²=0.72) |
| `model_Split_Tensile_MPa.pkl` | Split Tensile model (V3, R²=0.80) |
| `model_Porosity_pct.pkl` | Porosity model (V3, R²=0.66) |
| `Data UHPC.xlsx` | Training dataset (Benchmarking + SHAP pages) |
| `requirements.txt` | Python dependencies (pinned sklearn) |
| `.streamlit/config.toml` | Streamlit server config |
| `logo.png` | Your logo (add manually) |

## DO NOT include
- `model_v2_slump.pkl` — old broken model (units issue)
- `model_v3_flexural.pkl` — superseded
- `model_v3_porosity.pkl` — superseded
- `model_Chloride_Coulombs.pkl` — not used in app (R²=0.42)
- `model_Elastic_Modulus_GPa.pkl` — not used in app (R²=0.39)

## Deploy steps
1. Push all files above to a private GitHub repo
2. Go to share.streamlit.io → New app
3. Select repo, branch: main, file: app_merged_v2_v3.py
4. Deploy
5. In GoDaddy DNS → add CNAME: app → your-app.streamlit.app
6. In Streamlit settings → Custom domain → app.aicretesolutions.co.uk

## Local run
```bash
pip install -r requirements.txt
streamlit run app_merged_v2_v3.py
```
