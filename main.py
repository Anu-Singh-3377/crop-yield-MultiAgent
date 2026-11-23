import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Optional
import os
from pathlib import Path
from datetime import datetime
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from src.api.routes import router as api_router

# Initialize FastAPI app
app = FastAPI(title="Sustainable Farming AI System")
app.include_router(api_router, prefix="/api")
FRONTEND_DIST = Path(__file__).resolve().parent / "frontend" / "dist"

@app.get("/", response_class=HTMLResponse)
async def farmer_profile_page():
    frontend_index = FRONTEND_DIST / "index.html"
    if frontend_index.is_file():
        return FileResponse(frontend_index)
    with open("farmer_profile.html", "r", encoding="utf-8") as profile_page:
        return profile_page.read()

@app.get("/recommendations/{farmer_id}", response_class=HTMLResponse)
async def recommendations_page(farmer_id: int):
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Farming Recommendations</title>
        <style>
            body {{ font-family: Arial, sans-serif; max-width: 1000px; margin: 0 auto; padding: 24px; background: #f2f7f2; color: #243424; }}
            h1 {{ color: #27682f; }} .card {{ background: white; border-radius: 10px; padding: 18px; margin: 14px 0; box-shadow: 0 2px 8px #d9e2d9; }}
            .score {{ display: inline-block; margin-right: 18px; font-weight: bold; }} .error {{ color: #a12626; }}
            button {{ background: #27682f; color: white; border: 0; border-radius: 6px; padding: 10px 16px; cursor: pointer; }}
            .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 12px; }}
        </style>
    </head>
    <body>
        <button onclick="location.href='/'">Back to farmer profile</button>
        <h1>Personalized Farming Recommendations</h1>
        <div id="content">Loading recommendations...</div>
        <script>
            const farmerId = {farmer_id};
            const crops = new URLSearchParams(window.location.search).get('crops') || '';
            fetch(`/api/recommendations/${{farmerId}}${{crops ? '?crops=' + encodeURIComponent(crops) : ''}}`)
              .then(async response => {{
                const data = await response.json();
                if (!response.ok) throw new Error(data.detail || 'Recommendations could not be generated');
                return data;
              }})
              .then(data => {{
                const content = document.getElementById('content');
                const list = data.recommendations || [];
                if (!list.length) {{
                  content.innerHTML = '<div class=\"card\">No recommendations were generated. Please add preferred crops and try again.</div>';
                  return;
                }}
                content.innerHTML = '<div class=\"card\"><b>Farmer:</b> #' + data.farmer_id + '<br><b>Generated:</b> ' + data.timestamp + '</div>' +
                  list.map(item => {{
                    const market = item.market_analysis || {{}};
                    const trend = market.market_trend || {{}};
                    const profitability = market.profitability || {{}};
                    const metrics = item.sustainability_metrics || {{}};
                    return '<article class=\"card\"><h2>' + item.crop + '</h2>' +
                      '<div><span class=\"score\">Overall: ' + (item.overall_score * 100).toFixed(1) + '%</span>' +
                      '<span class=\"score\">Sustainability: ' + (item.sustainability_score * 100).toFixed(1) + '%</span></div>' +
                      '<p><b>Water requirement:</b> ' + Math.round(item.water_requirement).toLocaleString() + ' cubic metres</p>' +
                      '<p><b>Estimated cost:</b> Rs. ' + Math.round(item.estimated_cost).toLocaleString() + '</p>' +
                      '<p><b>Soil compatibility:</b> ' + (item.soil_compatibility * 100).toFixed(1) + '%</p>' +
                      '<p><b>Market demand:</b> ' + (trend.demand_level || 'N/A') + ' &nbsp; <b>Price trend:</b> ' + (trend.price_trend || 'N/A') + '</p>' +
                      '<p><b>Expected yield:</b> ' + (profitability.estimated_yield || 0).toFixed(1) + ' tons</p>' +
                      '<p><b>Potential revenue:</b> Rs. ' + Math.round(profitability.potential_revenue || 0).toLocaleString() + '</p>' +
                      '<p><b>Water efficiency:</b> ' + ((metrics.water_efficiency || 0) * 100).toFixed(1) + '% &nbsp; <b>Soil health:</b> ' + ((metrics.soil_health || 0) * 100).toFixed(1) + '%</p>' +
                      '</article>';
                  }}).join('');
              }})
              .catch(error => {{
                document.getElementById('content').innerHTML = '<div class=\"card error\">' + error.message + '</div>';
              }});
        </script>
    </body>
    </html>
    """

# Database setup
DATABASE_URL = "sqlite:///farming.db"
engine = create_engine(DATABASE_URL)

# Load datasets
try:
    farmer_data = pd.read_csv('farmer_advisor_dataset.csv')
    market_data = pd.read_csv('market_researcher_dataset.csv')
except FileNotFoundError:
    print("Error: Dataset files not found!")
    exit(1)

class FarmerInput(BaseModel):
    name: str
    location: str
    farm_size: float
    soil_type: str
    water_availability: str
    preferred_crops: Optional[List[str]] = None
    budget: float

class Recommendation(BaseModel):
    crop_name: str
    sustainability_score: float
    profitability_score: float
    water_efficiency_score: float
    expected_yield: float
    estimated_profit: float
    water_requirement: float
    carbon_footprint: float

class FarmerAdvisor:
    def __init__(self, farmer_data):
        self.farmer_data = farmer_data

    def analyze_farmer_profile(self, farmer_input: FarmerInput) -> dict:
        # Analyze farmer's profile and return relevant insights
        profile_analysis = {
            "soil_suitability": self._analyze_soil(farmer_input.soil_type),
            "water_efficiency": self._analyze_water_availability(farmer_input.water_availability),
            "farm_size_analysis": self._analyze_farm_size(farmer_input.farm_size)
        }
        return profile_analysis

    def _analyze_soil(self, soil_type: str) -> dict:
        # Implement soil analysis logic
        return {"suitability": "high", "recommendations": []}

    def _analyze_water_availability(self, water_availability: str) -> dict:
        # Implement water analysis logic
        return {"efficiency": "medium", "recommendations": []}

    def _analyze_farm_size(self, farm_size: float) -> dict:
        # Implement farm size analysis logic
        return {"scale": "medium", "recommendations": []}

class MarketResearcher:
    def __init__(self, market_data):
        self.market_data = market_data

    def analyze_market_trends(self, location: str, crop_list: List[str]) -> dict:
        # Analyze market trends for given location and crops
        market_analysis = {
            "demand_trends": self._analyze_demand(crop_list),
            "price_trends": self._analyze_prices(crop_list),
            "profitability": self._calculate_profitability(crop_list)
        }
        return market_analysis

    def _analyze_demand(self, crop_list: List[str]) -> dict:
        # Implement demand analysis logic
        return {"trends": {}, "recommendations": []}

    def _analyze_prices(self, crop_list: List[str]) -> dict:
        # Implement price analysis logic
        return {"trends": {}, "recommendations": []}

    def _calculate_profitability(self, crop_list: List[str]) -> dict:
        # Implement profitability calculation logic
        return {"scores": {}, "recommendations": []}

@app.post("/analyze-farming-profile", response_model=Recommendation)
async def analyze_farming_profile(farmer_input: FarmerInput):
    try:
        # Initialize agents
        farmer_advisor = FarmerAdvisor(farmer_data)
        market_researcher = MarketResearcher(market_data)

        # Get analyses from both agents
        farmer_analysis = farmer_advisor.analyze_farmer_profile(farmer_input)
        market_analysis = market_researcher.analyze_market_trends(
            farmer_input.location,
            farmer_input.preferred_crops or []
        )

        # Combine analyses and generate recommendation
        recommendation = _generate_recommendation(
            farmer_analysis,
            market_analysis,
            farmer_input
        )

        # Store recommendation in database
        _store_recommendation(recommendation, farmer_input)

        return recommendation

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def _generate_recommendation(
    farmer_analysis: dict,
    market_analysis: dict,
    farmer_input: FarmerInput
) -> Recommendation:
    # Implement recommendation generation logic
    return Recommendation(
        crop_name="Sample Crop",
        sustainability_score=0.85,
        profitability_score=0.75,
        water_efficiency_score=0.90,
        expected_yield=1000.0,
        estimated_profit=5000.0,
        water_requirement=500.0,
        carbon_footprint=200.0
    )

def _store_recommendation(recommendation: Recommendation, farmer_input: FarmerInput):
    # Store recommendation in SQLite database
    with engine.connect() as conn:
        # Insert farmer data
        farmer_result = conn.execute(
            text("""
                INSERT INTO farmers (name, location, farm_size, soil_type, water_availability)
                VALUES (:name, :location, :farm_size, :soil_type, :water_availability)
            """),
            {
                "name": farmer_input.name,
                "location": farmer_input.location,
                "farm_size": farmer_input.farm_size,
                "soil_type": farmer_input.soil_type,
                "water_availability": farmer_input.water_availability
            }
        )
        farmer_id = farmer_result.lastrowid

        # Insert recommendation
        conn.execute(
            text("""
                INSERT INTO recommendations (
                    farmer_id, crop_id, sustainability_score,
                    profitability_score, water_efficiency_score
                )
                VALUES (:farmer_id, :crop_id, :sustainability_score,
                        :profitability_score, :water_efficiency_score)
            """),
            {
                "farmer_id": farmer_id,
                "crop_id": 1,  # This should be replaced with actual crop_id
                "sustainability_score": recommendation.sustainability_score,
                "profitability_score": recommendation.profitability_score,
                "water_efficiency_score": recommendation.water_efficiency_score
            }
        )
        conn.commit()

if (FRONTEND_DIST / "assets").is_dir():
    app.mount(
        "/assets",
        StaticFiles(directory=str(FRONTEND_DIST / "assets")),
        name="agroai-frontend-assets",
    )
if (FRONTEND_DIST / "images").is_dir():
    app.mount(
        "/images",
        StaticFiles(directory=str(FRONTEND_DIST / "images")),
        name="agroai-frontend-images",
    )

@app.get("/{frontend_path:path}", include_in_schema=False)
async def react_frontend_routes(frontend_path: str):
    if frontend_path == "api" or frontend_path.startswith("api/"):
        raise HTTPException(status_code=404, detail="Not Found")
    frontend_index = FRONTEND_DIST / "index.html"
    if frontend_index.is_file():
        return FileResponse(frontend_index)
    raise HTTPException(status_code=404, detail="Not Found")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 