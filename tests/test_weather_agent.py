from unittest.mock import AsyncMock

import pytest

from src.agents.weather_agent import (
    WeatherAgent,
    WeatherAPIError,
    WeatherService,
)


class MockWeatherService:
    async def geocode(self, location):
        return {
            "name": "Bengaluru",
            "latitude": 12.9716,
            "longitude": 77.5946,
            "admin1": "Karnataka",
            "country": "India",
        }

    async def get_weather(self, latitude, longitude):
        return {
            "current": {
                "temperature_2m": 34.5,
                "relative_humidity_2m": 88,
                "precipitation": 6.2,
            },
            "daily": {
                "time": ["2026-11-23", "2026-11-24"],
                "temperature_2m_max": [35.0, 33.0],
                "temperature_2m_min": [22.0, 21.0],
                "precipitation_sum": [12.0, 8.0],
                "precipitation_probability_max": [70, 45],
            },
        }


@pytest.mark.asyncio
async def test_location_weather_normalization():
    agent = WeatherAgent(MockWeatherService())
    weather = await agent.get_weather(location="Bengaluru")

    assert weather.location == "Bengaluru, Karnataka, India"
    assert weather.latitude == 12.9716
    assert weather.longitude == 77.5946
    assert weather.temperature == 34.5
    assert weather.humidity == 88
    assert weather.rainfall == 6.2
    assert len(weather.forecast) == 2
    assert weather.forecast[0].temperature_max == 35.0
    assert weather.forecast[0].rainfall == 12.0
    assert {risk.type for risk in weather.agricultural_risks} >= {
        "high_temperature",
        "high_humidity",
        "moderate_rain",
    }


@pytest.mark.asyncio
async def test_coordinates_weather_normalization():
    agent = WeatherAgent(MockWeatherService())
    weather = await agent.get_weather(latitude=12.9716, longitude=77.5946)

    assert weather.location == "12.9716, 77.5946"
    assert weather.temperature == 34.5


@pytest.mark.asyncio
async def test_missing_location_and_coordinates_is_rejected():
    agent = WeatherAgent(MockWeatherService())

    with pytest.raises(ValueError, match="location or both latitude"):
        await agent.get_weather()


@pytest.mark.asyncio
async def test_partial_coordinates_are_rejected():
    agent = WeatherAgent(MockWeatherService())

    with pytest.raises(ValueError, match="location or both latitude"):
        await agent.get_weather(latitude=12.9716)


@pytest.mark.asyncio
async def test_invalid_coordinates_are_rejected():
    agent = WeatherAgent(MockWeatherService())

    with pytest.raises(ValueError, match="outside the valid range"):
        await agent.get_weather(latitude=91, longitude=77)


@pytest.mark.asyncio
async def test_geocoding_failure_is_propagated():
    service = MockWeatherService()
    service.geocode = AsyncMock(
        side_effect=WeatherAPIError("Location 'Unknown place' was not found")
    )
    agent = WeatherAgent(service)

    with pytest.raises(WeatherAPIError, match="was not found"):
        await agent.get_weather(location="Unknown place")


@pytest.mark.asyncio
async def test_weather_provider_failure_is_propagated():
    service = MockWeatherService()
    service.get_weather = AsyncMock(
        side_effect=WeatherAPIError("Unable to contact weather provider")
    )
    agent = WeatherAgent(service)

    with pytest.raises(WeatherAPIError, match="Unable to contact"):
        await agent.get_weather(latitude=12.9716, longitude=77.5946)


def test_optional_api_key_is_environment_configured(monkeypatch):
    monkeypatch.setenv("WEATHER_API_KEY", "test-key")
    service = WeatherService(api_key=None)

    assert service.api_key == "test-key"
    assert service._headers()["X-API-Key"] == "test-key"
