import base64
import binascii
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime
import pandas as pd
import os
from pydantic import Field

from ..database.database import get_db
from ..database.models import Farmer, Crop, Recommendation, MarketData
from ..agents.farmer_advisor import FarmerAdvisor, FarmerProfile
from ..agents.market_researcher import MarketResearcher
from ..agroai.agents.yield_prediction import (
    YieldPredictionAgent,
    YieldPredictionRequest,
    YieldPredictionResponse,
)
from ..agents.disease_agent import (
    DiseaseAgent,
    DiseasePredictionResponse,
)
from fastapi import UploadFile, File
from ..agents.weather_agent import (
    WeatherAgent,
    WeatherAPIError,
    WeatherResponse,
)
from ..agroai.orchestration import (
    AgroAIOrchestrator,
    AgroAIRequest,
    AgroAIResponse,
)

router = APIRouter()
agroai_orchestrator = AgroAIOrchestrator()


class AgroAnalysisRequest(AgroAIRequest):
    image_base64: Optional[str] = Field(default=None, exclude=True)

# Load data (this would be replaced with actual data loading)
try:
    data_dir = 'data' if os.path.exists('data/farmer_advisor_dataset.csv') else '.'
    farmer_data = pd.read_csv(f'{data_dir}/farmer_advisor_dataset.csv')
    market_data = pd.read_csv(f'{data_dir}/market_researcher_dataset.csv')
except FileNotFoundError:
    print("Warning: Dataset files not found. Using mock data.")
    farmer_data = pd.DataFrame()
    market_data = pd.DataFrame()

# Initialize agents
farmer_advisor = FarmerAdvisor(farmer_data)
market_researcher = MarketResearcher(market_data)
yield_prediction_agent = YieldPredictionAgent


@router.post("/agro/analyze", response_model=AgroAIResponse)
async def analyze_agro(request: AgroAnalysisRequest):
    """Run the requested AgroAI agents and return their structured results."""
    payload = request.model_dump()
    if request.image_base64 is not None:
        if request.image is not None:
            raise HTTPException(
                status_code=422,
                detail="Provide either image or image_base64, not both.",
            )
        try:
            payload["image"] = base64.b64decode(
                request.image_base64,
                validate=True,
            )
        except (ValueError, binascii.Error) as exc:
            raise HTTPException(
                status_code=422,
                detail="image_base64 must contain valid Base64 data.",
            ) from exc

    return await agroai_orchestrator.ainvoke(payload)


@router.get("/weather", response_model=WeatherResponse)
async def get_weather(
    location: Optional[str] = None,
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
):
    """Return normalized current weather and agricultural risks."""
    try:
        return await WeatherAgent().get_weather(
            location=location,
            latitude=latitude,
            longitude=longitude,
        )
    except WeatherAPIError as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

@router.post("/yield/predict", response_model=YieldPredictionResponse)
async def predict_yield(request: YieldPredictionRequest):
    """Predict crop yield using the trained XGBoost yield model."""
    try:
        agent = yield_prediction_agent()
        return agent.predict(request)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

@router.post("/disease/detect", response_model=DiseasePredictionResponse)
async def detect_disease(
    image: UploadFile = File(...),
    crop: Optional[str] = None,
):
    """Run crop disease classification with the independent disease agent."""
    try:
        image_bytes = await image.read()
        agent = DiseaseAgent()
        return agent.predict(image_bytes, crop_hint=crop)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

@router.post("/farmers/", response_model=dict)
async def create_farmer(
    farmer: FarmerProfile,
    db: Session = Depends(get_db)
):
    """Create a new farmer profile."""
    db_farmer = Farmer(
        name=farmer.name,
        location=farmer.location,
        farm_size=farmer.farm_size,
        soil_type=farmer.soil_type,
        water_availability=farmer.water_availability
    )
    db.add(db_farmer)
    db.commit()
    db.refresh(db_farmer)
    return {"farmer_id": db_farmer.farmer_id, "message": "Farmer profile created successfully"}

@router.get("/recommendations/{farmer_id}")
async def get_recommendations(
    farmer_id: int,
    crops: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Get farming recommendations for a specific farmer."""
    farmer = db.query(Farmer).filter(Farmer.farmer_id == farmer_id).first()
    if not farmer:
        raise HTTPException(status_code=404, detail="Farmer not found")
    
    # Create farmer profile
    farmer_profile = FarmerProfile(
        name=farmer.name,
        location=farmer.location,
        farm_size=farmer.farm_size,
        soil_type=farmer.soil_type,
        water_availability=farmer.water_availability,
        preferred_crops=(
            [crop.strip() for crop in crops.split(',') if crop.strip()]
            if crops else None
        ),
        budget=10000  # This would come from the farmer's actual data
    )
    
    # Get recommendations from both agents
    crop_recommendations = farmer_advisor.get_crop_recommendations(farmer_profile)
    
    # Get market analysis for recommended crops
    crops = [rec['crop'] for rec in crop_recommendations]
    market_analysis = market_researcher.generate_market_report(
        crops,
        farmer_profile.location,
        farmer_profile.farm_size
    )
    
    # Combine recommendations
    final_recommendations = []
    market_by_crop = {
        rec['crop'].lower(): rec
        for rec in market_analysis
    }
    for crop_rec in crop_recommendations:
        market_rec = market_by_crop.get(crop_rec['crop'].lower())
        if market_rec is None:
            continue
        combined_rec = {
            **crop_rec,
            'market_analysis': market_rec,
            'overall_score': (
                crop_rec['sustainability_score'] * 0.4 +
                market_rec['recommendation_score'] * 0.6
            )
        }
        final_recommendations.append(combined_rec)
    
    # Sort by overall score
    final_recommendations.sort(key=lambda x: x['overall_score'], reverse=True)
    
    # Store recommendations in database
    for rec in final_recommendations[:3]:  # Store top 3 recommendations
        db_recommendation = Recommendation(
            farmer_id=farmer_id,
            crop_id=1,  # This would be the actual crop ID
            sustainability_score=rec['sustainability_score'],
            profitability_score=rec['market_analysis']['recommendation_score'],
            water_efficiency_score=rec['sustainability_metrics'].water_efficiency,
            details=str(rec)  # Convert to string for storage
        )
        db.add(db_recommendation)
    
    db.commit()
    
    return {
        "farmer_id": farmer_id,
        "recommendations": final_recommendations,
        "timestamp": datetime.utcnow().isoformat()
    }

@router.get("/market-analysis/{region}")
async def get_market_analysis(
    region: str,
    crops: Optional[List[str]] = None,
    db: Session = Depends(get_db)
):
    """Get market analysis for specific crops in a region."""
    if not crops:
        crops = ['rice', 'wheat', 'corn', 'soybeans']  # Default crops
    
    market_report = market_researcher.generate_market_report(
        crops=crops,
        region=region,
        farm_size=10.0  # Default farm size for analysis
    )
    
    return {
        "region": region,
        "market_analysis": market_report,
        "timestamp": datetime.utcnow().isoformat()
    }

@router.get("/sustainable-practices/{farmer_id}/{crop}")
async def get_sustainable_practices(
    farmer_id: int,
    crop: str,
    db: Session = Depends(get_db)
):
    """Get sustainable farming practices for a specific crop."""
    farmer = db.query(Farmer).filter(Farmer.farmer_id == farmer_id).first()
    if not farmer:
        raise HTTPException(status_code=404, detail="Farmer not found")
    
    farmer_profile = FarmerProfile(
        name=farmer.name,
        location=farmer.location,
        farm_size=farmer.farm_size,
        soil_type=farmer.soil_type,
        water_availability=farmer.water_availability,
        preferred_crops=[crop],
        budget=10000  # This would come from the farmer's actual data
    )
    
    practices = farmer_advisor.generate_sustainable_practices(
        farmer_profile,
        crop
    )
    
    return {
        "farmer_id": farmer_id,
        "crop": crop,
        "sustainable_practices": practices,
        "timestamp": datetime.utcnow().isoformat()
    } 