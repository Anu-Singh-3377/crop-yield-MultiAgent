# AgroAI — AI-Powered Sustainable Farming Decision Support System

AgroAI brings together farm-profile heuristics, weather observations, crop-image classification, yield estimates, market analysis, and sustainability-oriented advice in a web dashboard and FastAPI service.

> AgroAI is a decision-support prototype. Its outputs are estimates, not guaranteed outcomes or a substitute for local agricultural expertise.

## Project overview

Agricultural decisions depend on interacting factors: soil and water conditions, crop health, weather, expected harvest, market conditions, and resource constraints. These inputs are often considered separately, and some may be incomplete or uncertain.

This project provides separate agricultural agents and a conditional LangGraph workflow that selects applicable agents, returns their structured outputs, and combines available evidence into an advisory. The React dashboard submits inputs to the backend and displays the returned results; it does not generate predictions in the browser.

## Problem and approach

Weather variability, crop-health risks, uncertain yields, changing prices, and resource-management needs can make planning difficult. AgroAI provides a single workflow for considering these factors together, while reporting unavailable results and uncertainty rather than treating estimates as guarantees.

For a full farm assessment, the current workflow is:

```text
Farm profile and crop inputs
              ↓
Conditional Farmer, Weather, Disease, Yield, and Market agents
              ↓
Structured sustainability assessment
              ↓
Local Advisory Agent
              ↓
Agent status, results, warnings, and recommended actions
```

Agents are selected according to the requested analysis and available inputs. For example, yield-only analysis does not call the disease classifier. A disease-image request can also request weather when location information is supplied.

## Current features

- **Farm profile:** save a farmer profile and request crop recommendations.
- **Farmer Advisor:** rule-based soil/crop compatibility, water-requirement, and sustainability calculations.
- **Weather:** retrieve current conditions and a seven-day forecast from Open-Meteo by location or coordinates; derive basic agricultural risk indicators.
- **Crop disease:** independent image-classification code and training script are present. A trained disease model and image dataset are **not** included, so inference requires supplying both.
- **Yield prediction:** XGBoost inference using the included model artifact, with engineered features and SHAP-based important factors.
- **Market Researcher:** structured crop reports based on the bundled CSV and fallback logic. It is not connected to a live market-price service.
- **Local Advisory:** rule-based, conservative actions based on the outputs supplied by other agents; it does not use an LLM to invent upstream results.
- **Orchestration:** a LangGraph workflow validates inputs, selects agents by intent, captures individual agent failures, combines available results, and validates the response.
- **Dashboard:** responsive React/Vite pages for farm analysis and individual agent workflows. Agent status is shown from the completed backend response; the API does not stream live progress.
- **Profile and recommendation persistence:** SQLite models support farmer profiles and related records. There is no API for persistent AgroAI execution history; the History page does not represent stored runs.

## Agent architecture

```text
                              AgroAI API
                                  │
                                  ▼
                       LangGraph orchestrator
                ┌────────┬───────┼───────┬─────────┐
                ▼        ▼       ▼       ▼         ▼
             Farmer   Weather  Disease  Yield     Market
             Advisor   Agent   Agent    Agent    Researcher
                └────────┴───────┼───────┴─────────┘
                                  ▼
                         Local Advisory Agent
                                  │
                                  ▼
                 Structured results and agent status
```

### Farmer Advisor

- **Purpose:** assess crop options using farmer and farm details.
- **Input:** farmer location, farm size, soil type, water availability, preferred crops, and budget.
- **Processing:** applies code-defined soil compatibility scores, crop water-requirement values, and sustainability calculations. These are heuristics, not a trained model.
- **Output and integration:** crop candidates, water requirements, compatibility, estimated costs, and sustainability metrics; the results can be passed to the Local Advisory Agent.

### Weather Agent

- **Purpose:** provide current weather, forecast, and basic agricultural risk indicators.
- **Input:** a location or latitude/longitude.
- **Processing:** geocodes named locations and requests current and daily forecast data from the configured weather provider. Open-Meteo is the default.
- **Output and integration:** normalized temperature, humidity, rainfall, forecast days, and risks such as heat, heavy rain, or high humidity. The results can inform advisory actions.

### Crop Disease Agent

- **Purpose:** classify an uploaded crop image using a saved model.
- **Input:** an image and optional crop hint.
- **Processing:** loads the checkpoint at `artifacts/crop_disease_model.pt` and applies the checkpoint's image-size resize and tensor conversion before classification. The current training code defines a custom CNN; it is not an EfficientNet/ResNet transfer-learning implementation.
- **Output and integration:** crop, predicted disease class, confidence, and risk level. This is a model classification, not a confirmed diagnosis.
- **Availability:** the checkpoint and dataset are not present, so the endpoint cannot produce a real classification until a dataset is supplied and the model is trained.

### Yield Prediction Agent

- **Purpose:** estimate crop yield from crop and available environmental/agricultural inputs.
- **Input:** crop, soil pH and moisture, temperature, rainfall, fertilizer and pesticide usage, and sustainability score. Location is reported but is not a model feature; historical yield is returned as context but is not used as a model feature.
- **Processing:** loads `artifacts/yield_model.joblib`, imputes missing numeric values using training medians, scales numeric inputs, one-hot encodes crop, and derives rainfall-temperature, moisture-pH, and fertilizer-per-rainfall features. It uses the saved XGBoost pipeline and computes feature impacts with SHAP.
- **Output and integration:** estimated yield in tons, a residual-based range, model version, and important factors. The estimate is uncertain and is not guaranteed.

### Market Researcher

- **Purpose:** return structured crop-market comparisons for a region.
- **Input:** one or more crop names, region, and farm size.
- **Processing:** uses the bundled market CSV where applicable, plus fallback values and heuristic calculations. Some fallback and price calculations are randomized, and profitability includes fixed assumptions; the result is not a validated forecast or a live quote.
- **Output and integration:** price-trend and demand estimates, profitability fields, and a recommendation score; the report can be passed to the Local Advisory Agent.

### Local Advisory Agent

- **Purpose:** produce recommendations from supplied agent results without fabricating missing weather, market, disease, or yield data.
- **Input:** available farmer recommendations, market analysis, weather, disease, and yield outputs.
- **Processing:** applies rules to supplied structured results and labels uncertainty and warnings.
- **Output and integration:** crop, disease, weather, yield, and market assessments plus sustainability actions, recommended actions, and warnings. Missing upstream results are marked unavailable.

## Machine learning and datasets

### Yield model

The repository includes a trained artifact at `artifacts/yield_model.joblib`. Training code is in `src/agroai/data/yield_pipeline.py` and `scripts/train_yield_model.py`.

| Item | Implementation |
|---|---|
| Target | `Crop_Yield_ton` |
| Dataset | `data/farmer_advisor_dataset.csv` |
| Input columns | `Crop_Type`, `Soil_pH`, `Soil_Moisture`, `Temperature_C`, `Rainfall_mm`, `Fertilizer_Usage_kg`, `Pesticide_Usage_kg`, `Sustainability_Score` |
| Models | Ridge regression baseline and XGBoost regressor |
| Preprocessing | Median imputation and standard scaling for numeric features; most-frequent imputation and one-hot encoding for crop |
| Feature engineering | Rainfall × temperature, moisture × soil pH, and fertilizer ÷ rainfall |
| Evaluation code | MAE, RMSE, and R² on validation/test splits |
| Explanation | SHAP TreeExplainer feature impacts during inference |
| Saved artifact | `artifacts/yield_model.joblib` |

No performance score or real-world yield improvement is claimed here. The training script prints measured split metrics when it is run.

To retrain the yield model from the repository root:

```bash
python scripts/train_yield_model.py
```

### Disease classifier and dataset

The disease training script expects an `ImageFolder`-style dataset:

```text
data/disease/
├── train/<class-name>/<images>
├── val/<same-class-names>/<images>
└── test/<same-class-names>/<images>
```

The training code resizes images to 256 × 256 by default and converts them to tensors. It defines a custom CNN, uses cross-entropy loss and Adam, and reports precision, recall, F1, and a confusion matrix when training completes. The script does not implement image augmentation.

The repository's [disease dataset notes](data/disease/README.md) identify the referenced Kaggle dataset as [`vipoooool/new-plant-diseases-dataset`](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset), Kaggle dataset ID `78313`, with notebook metadata source/version ID `182633`. The notes report version 2 and a “Data files © Original Authors” license notice. The dataset itself is not included; verify permission and licensing before downloading or using it.

Train only after obtaining and organizing an appropriately licensed dataset:

```bash
python scripts/train_disease_model.py
```

By default, the script reads `data/disease/`, saves the checkpoint to `artifacts/crop_disease_model.pt`, and writes an adjacent `.evaluation.json` report. The API loads the checkpoint from that same default path. No trained disease artifact or disease evaluation report is currently included.

### Other bundled CSV data

| File | Main fields / purpose |
|---|---|
| `data/farmer_advisor_dataset.csv` | Farm ID, soil, weather, crop, inputs, crop yield, and sustainability score; also used as the yield-model training dataset. |
| `data/market_researcher_dataset.csv` | Product, market price, demand/supply indices, competitor price, economic/weather indicators, and seasonal/consumer fields; used by market-analysis code. |

Matching CSV copies are also present at the repository root for legacy application code. The CSVs are not live data feeds. No dataset row counts or source claims are made here.

## React frontend

The frontend is a React 19 application built with Vite and styled with project CSS (no component or chart library is configured). Its navigation includes Dashboard, Farm Analysis, Crop Health, Weather, Yield Prediction, Market Insights, Advisory, and History.

The farm-analysis page provides a vertical workflow for farmer profile, weather/location, crop image, yield inputs, market research, and sustainability/advisory. It submits to the backend and displays returned results, agent success/failure/skipped status, warnings, loading states, and request errors. Image uploads are supported in the UI. The History page is present, but saved analysis history is not available from the backend.

## Technology stack

| Layer | Technology |
|---|---|
| Frontend | React 19, React Router, Vite, custom CSS |
| Backend/API | Python, FastAPI, Pydantic |
| Orchestration | LangGraph |
| Yield ML | scikit-learn, XGBoost, SHAP, pandas, NumPy, joblib |
| Disease ML | PyTorch, torchvision, Pillow (training/inference dependencies; model artifact and image dataset absent) |
| Persistence | SQLite and SQLAlchemy |
| Weather provider | Open-Meteo by default; configurable provider URLs |
| Tests | pytest, pytest-asyncio, httpx |
| Deployment | No deployment configuration is included in this repository. |

## Repository structure

```text
.
├── artifacts/
│   └── yield_model.joblib
├── data/
│   ├── disease/
│   │   └── README.md
│   ├── farmer_advisor_dataset.csv
│   └── market_researcher_dataset.csv
├── docs/
│   └── api_documentation.md
├── frontend/
│   ├── public/images/
│   └── src/
│       ├── components/
│       ├── hooks/
│       ├── services/
│       ├── App.jsx
│       ├── pages.jsx
│       └── styles.css
├── scripts/
│   ├── setup_database.py
│   ├── train_disease_model.py
│   └── train_yield_model.py
├── src/
│   ├── agents/
│   ├── agroai/
│   │   ├── agents/
│   │   ├── data/
│   │   └── orchestration/
│   ├── api/
│   └── database/
├── tests/
├── farmer_profile.html
├── main.py
├── requirements.txt
└── README.md
```

## API

The API router is mounted under `/api`. The main routes implemented in the source are:

| Method | Endpoint | Purpose |
|---|---|---|
| `POST` | `/api/agro/analyze` | Run the intent-based AgroAI workflow; accepts structured inputs and optional base64 image data. |
| `GET` | `/api/weather` | Retrieve normalized weather and agricultural risks by location or coordinates. |
| `POST` | `/api/yield/predict` | Request yield-model inference. |
| `POST` | `/api/disease/detect` | Classify an uploaded image; requires a trained disease checkpoint. |
| `POST` | `/api/farmers/` | Save a farmer profile to SQLite. |
| `GET` | `/api/recommendations/{farmer_id}` | Get and persist farmer crop recommendations. |
| `GET` | `/api/market-analysis/{region}` | Get market analysis for optional `crops` query parameters. |
| `GET` | `/api/sustainable-practices/{farmer_id}/{crop}` | Return Farmer Advisor sustainable-practice suggestions. |

The older `POST /analyze-farming-profile` route is also present in `main.py`, outside the `/api` prefix. It is a legacy prototype and returns hard-coded sample recommendation values; do not use it as a real analysis result.

FastAPI's interactive OpenAPI page is available at `http://127.0.0.1:8000/docs` when running locally. The checked-in `docs/api_documentation.md` describes older routes and may not include the newer AgroAI endpoints; `/docs` reflects the running application.

## Installation and running

### Requirements

- Python version compatible with the packages in `requirements.txt` (Python 3.11 was used for project validation).
- Node.js and npm for the React frontend.
- Disease model training additionally requires the dataset described above; CUDA is optional.

### Backend

From the repository root, create and activate a virtual environment, then install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Initialize the SQLite schema and bundled initial crop/market records if needed:

```powershell
python scripts/setup_database.py
```

Start FastAPI:

```powershell
python main.py
```

The application binds to `0.0.0.0:8000` by default. Open `http://127.0.0.1:8000/` for the dashboard when a frontend build exists; otherwise the legacy `farmer_profile.html` page is served.

### Frontend development

In a second terminal:

```powershell
cd frontend
npm install
npm run dev
```

The Vite server runs at `http://127.0.0.1:5173` by default and proxies `/api` to `http://127.0.0.1:8000`. Start the backend first. To use a different backend port, set `AGROAI_BACKEND_URL` in `frontend/.env`.

### Production frontend build

Build the React app before starting FastAPI:

```powershell
cd frontend
npm install
npm run build
cd ..
python main.py
```

FastAPI serves the generated `frontend/dist` files when present. Build output and installed frontend dependencies are excluded from Git.

## Environment variables

The root `.env.example` contains optional weather-provider settings:

| Variable | Purpose |
|---|---|
| `WEATHER_API_URL` | Override the default Open-Meteo forecast URL. |
| `WEATHER_GEOCODING_API_URL` | Override the default Open-Meteo geocoding URL. |
| `WEATHER_API_KEY` | Optional provider key; the default Open-Meteo configuration does not require one. |

`frontend/.env.example` documents:

| Variable | Purpose |
|---|---|
| `AGROAI_BACKEND_URL` | Backend target used by the Vite `/api` development proxy. |
| `VITE_API_BASE_URL` | Optional browser API base override; empty uses same-origin `/api`. |

Copy an example file to `.env` only when needed. Never commit API keys or other secrets.

## Testing

Run backend tests from the repository root:

```powershell
python -m pytest
```

The pytest suite covers the AgroAI graph and API route, disease-agent behavior, local advisory, market researcher, weather service (with mocked provider responses), and yield prediction. The frontend package currently defines development, build, and preview scripts; it does not define a frontend test script.

Build-check the frontend with:

```powershell
cd frontend
npm run build
```

## Results and limitations

- The yield pipeline implements validation, preprocessing, train/validation/test evaluation, model persistence, inference, and SHAP explanations. Its metrics are computed by the training script; this README does not claim unverified scores.
- A yield artifact is present. Prediction quality should be evaluated for the intended crops, regions, and data before operational use.
- Disease training/inference code exists, but the dataset and trained checkpoint do not. Disease results are unavailable until a dataset is obtained with appropriate permissions and the model is trained.
- The weather agent calls the configured provider, but the resulting forecast depends on that provider and location.
- Market analysis uses local CSV/fallback heuristics, not current live prices. Its price/profit figures are not validated market forecasts.
- Farmer Advisor and Local Advisory calculations are code-defined heuristics/rules, not experimentally validated impact guarantees.
- No benchmark-backed water reduction, yield increase, pesticide reduction, or profitability improvement results are included.

## Security and operational notes

- Keep `.env` files and API credentials private; use environment variables rather than source-code constants.
- The repository does not configure API authentication or CORS middleware. Add and verify appropriate controls before exposing the service publicly.
- Validate inputs and restrict image size/type at any public deployment boundary; the current dashboard applies an 8 MB image-size check, but API-wide upload limits and production controls should be reviewed separately.
- Do not treat disease classifications, model yield estimates, heuristic market outputs, or advisory actions as guaranteed outcomes. Confirm high-impact decisions with local expertise.

## Screenshots

No project screenshots are checked into the repository. Add verified dashboard screenshots here when available.

## Future enhancements

- Obtain and license-review the disease dataset, train the classifier, and publish its evaluation report and checkpoint.
- Replace heuristic market fallbacks with a documented, current, region-aware data source and validate the estimates.
- Add backend-backed AgroAI execution history if historical run tracking is required.
- Publish reproducible yield-model evaluation details for representative data and intended deployment regions.
- Add production deployment configuration, API authentication, and reviewed CORS/network settings.

## Research direction

The current conditional agent graph is a foundation for studying coordinated agricultural decision support. Further work can evaluate agent selection, evidence provenance, uncertainty handling, and recommendation quality against expert-reviewed cases and local agricultural data. These are research directions, not claims of validated impact.

## Contributors

No formal contributor roster is maintained in this repository. See the Git history for contribution details.

## License

No `LICENSE` file is present in the repository. No project license is specified here; obtain permission before redistributing or reusing the project or any third-party dataset.
