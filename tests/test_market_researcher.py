import pandas as pd
import pytest

from src.agents.market_researcher import (
    MarketResearcher,
    MarketResearchInput,
    MarketResearchOutput,
)


@pytest.fixture
def researcher():
    data = pd.DataFrame([
        {
            "Product": "Wheat",
            "Market_Price_per_ton": 300.0,
            "Demand_Index": 80.0,
            "Supply_Index": 40.0,
        },
        {
            "Product": "Wheat",
            "Market_Price_per_ton": 320.0,
            "Demand_Index": 90.0,
            "Supply_Index": 45.0,
        },
        {
            "Product": "Rice",
            "Market_Price_per_ton": 400.0,
            "Demand_Index": 100.0,
            "Supply_Index": 50.0,
        },
    ])
    return MarketResearcher(data)


def test_structured_run_returns_typed_agent_output(researcher):
    result = researcher.run(MarketResearchInput(
        crops=["Wheat", "Rice"],
        region="Karnataka",
        farm_size=2.5,
    ))

    assert isinstance(result, MarketResearchOutput)
    assert result.agent_name == "market_researcher"
    assert result.status == "completed"
    assert result.region == "Karnataka"
    assert {item.crop for item in result.recommendations} == {"Wheat", "Rice"}
    assert all(item.market_trend.crop for item in result.recommendations)
    assert all(item.recommendation_score >= 0 for item in result.recommendations)
    assert result.model_dump()["recommendations"]


def test_legacy_report_interface_is_unchanged(researcher):
    result = researcher.generate_market_report(
        crops=["Wheat"],
        region="Karnataka",
        farm_size=2.5,
    )

    assert isinstance(result, list)
    assert result[0]["crop"] == "Wheat"
    assert {"market_trend", "profitability", "recommendation_score"} <= set(result[0])


@pytest.mark.parametrize(
    "values",
    [
        {"crops": [], "region": "Karnataka", "farm_size": 2.5},
        {"crops": ["Wheat"], "region": "", "farm_size": 2.5},
        {"crops": ["Wheat"], "region": "Karnataka", "farm_size": 0},
    ],
)
def test_structured_input_validates_required_market_parameters(values):
    with pytest.raises(ValueError):
        MarketResearchInput(**values)
