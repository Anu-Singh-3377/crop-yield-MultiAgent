from __future__ import annotations

import os
from typing import Any, Dict, Iterable, List, Optional

import httpx
from pydantic import BaseModel, Field
from dotenv import load_dotenv


load_dotenv()


DEFAULT_WEATHER_API_URL = "https://api.open-meteo.com/v1/forecast"
DEFAULT_GEOCODING_API_URL = "https://geocoding-api.open-meteo.com/v1/search"


class WeatherForecastDay(BaseModel):
    date: str
    temperature_max: Optional[float] = None
    temperature_min: Optional[float] = None
    rainfall: Optional[float] = None
    precipitation_probability: Optional[float] = None


class AgriculturalRisk(BaseModel):
    type: str
    severity: str
    description: str


class WeatherResponse(BaseModel):
    location: str
    temperature: Optional[float] = None
    humidity: Optional[float] = None
    rainfall: Optional[float] = None
    forecast: List[WeatherForecastDay] = Field(default_factory=list)
    agricultural_risks: List[AgriculturalRisk] = Field(default_factory=list)
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    provider_status: str = "ok"


class WeatherAPIError(RuntimeError):
    pass


class WeatherService:
    """HTTP client for a configured current-weather and forecast API."""

    def __init__(
        self,
        weather_api_url: Optional[str] = None,
        geocoding_api_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout_seconds: float = 10.0,
        client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        self.weather_api_url = (
            weather_api_url
            or os.getenv("WEATHER_API_URL")
            or DEFAULT_WEATHER_API_URL
        )
        self.geocoding_api_url = (
            geocoding_api_url
            or os.getenv("WEATHER_GEOCODING_API_URL")
            or DEFAULT_GEOCODING_API_URL
        )
        self.api_key = api_key if api_key is not None else os.getenv("WEATHER_API_KEY")
        self.timeout_seconds = timeout_seconds
        self._provided_client = client

    def _headers(self) -> Dict[str, str]:
        headers = {"Accept": "application/json"}
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        return headers

    async def geocode(self, location: str) -> Dict[str, Any]:
        params = {"name": location, "count": 1, "language": "en", "format": "json"}
        data = await self._get_json(self.geocoding_api_url, params)
        results = data.get("results") or []
        if not results:
            raise WeatherAPIError(f"Location '{location}' was not found")
        result = results[0]
        return {
            "name": result.get("name") or location,
            "latitude": float(result["latitude"]),
            "longitude": float(result["longitude"]),
            "admin1": result.get("admin1"),
            "country": result.get("country"),
        }

    async def get_weather(
        self,
        latitude: float,
        longitude: float,
    ) -> Dict[str, Any]:
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "current": "temperature_2m,relative_humidity_2m,precipitation",
            "daily": (
                "temperature_2m_max,temperature_2m_min,"
                "precipitation_sum,precipitation_probability_max"
            ),
            "timezone": "auto",
            "forecast_days": 7,
        }
        return await self._get_json(self.weather_api_url, params)

    async def _get_json(
        self,
        url: str,
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        try:
            if self._provided_client is not None:
                response = await self._provided_client.get(
                    url,
                    params=params,
                    headers=self._headers(),
                )
            else:
                async with httpx.AsyncClient(
                    timeout=self.timeout_seconds,
                ) as client:
                    response = await client.get(
                        url,
                        params=params,
                        headers=self._headers(),
                    )
            response.raise_for_status()
            data = response.json()
            if not isinstance(data, dict):
                raise WeatherAPIError("Weather provider returned a non-object response")
            return data
        except httpx.TimeoutException as exc:
            raise WeatherAPIError("Weather provider timed out") from exc
        except httpx.HTTPStatusError as exc:
            raise WeatherAPIError(
                f"Weather provider returned HTTP {exc.response.status_code}"
            ) from exc
        except httpx.RequestError as exc:
            raise WeatherAPIError("Unable to contact weather provider") from exc


class WeatherAgent:
    """Normalizes configured weather API data for agricultural use."""

    def __init__(self, service: Optional[WeatherService] = None) -> None:
        self.service = service or WeatherService()

    async def get_weather(
        self,
        location: Optional[str] = None,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
    ) -> WeatherResponse:
        location_name = (location or "").strip()
        if location_name:
            geocoded = await self.service.geocode(location_name)
            location_name = self._display_location(geocoded)
            latitude = geocoded["latitude"]
            longitude = geocoded["longitude"]
        elif latitude is None or longitude is None:
            raise ValueError("Provide a location or both latitude and longitude")

        if not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
            raise ValueError("Latitude or longitude is outside the valid range")

        try:
            raw = await self.service.get_weather(latitude, longitude)
        except WeatherAPIError:
            raise

        normalized = self._normalize_response(
            raw=raw,
            location_name=location_name or f"{latitude:.4f}, {longitude:.4f}",
            latitude=latitude,
            longitude=longitude,
        )
        return normalized

    def _display_location(self, geocoded: Dict[str, Any]) -> str:
        parts = [
            geocoded.get("name"),
            geocoded.get("admin1"),
            geocoded.get("country"),
        ]
        return ", ".join(str(part) for part in parts if part)

    def _normalize_response(
        self,
        raw: Dict[str, Any],
        location_name: str,
        latitude: float,
        longitude: float,
    ) -> WeatherResponse:
        current = raw.get("current") or {}
        daily = raw.get("daily") or {}

        temperature = self._optional_float(
            current.get("temperature_2m")
        )
        humidity = self._optional_float(
            current.get("relative_humidity_2m")
        )
        rainfall = self._optional_float(
            current.get("precipitation")
        )

        dates = daily.get("time") or []
        max_temperatures = daily.get("temperature_2m_max") or []
        min_temperatures = daily.get("temperature_2m_min") or []
        rainfall_totals = daily.get("precipitation_sum") or []
        rain_probabilities = daily.get("precipitation_probability_max") or []

        forecast = [
            WeatherForecastDay(
                date=str(date),
                temperature_max=self._optional_float(
                    max_temperatures[index]
                    if index < len(max_temperatures)
                    else None
                ),
                temperature_min=self._optional_float(
                    min_temperatures[index]
                    if index < len(min_temperatures)
                    else None
                ),
                rainfall=self._optional_float(
                    rainfall_totals[index]
                    if index < len(rainfall_totals)
                    else None
                ),
                precipitation_probability=self._optional_float(
                    rain_probabilities[index]
                    if index < len(rain_probabilities)
                    else None
                ),
            )
            for index, date in enumerate(dates)
        ]

        return WeatherResponse(
            location=location_name,
            temperature=temperature,
            humidity=humidity,
            rainfall=rainfall,
            forecast=forecast,
            agricultural_risks=self.calculate_agricultural_risks(
                temperature=temperature,
                humidity=humidity,
                rainfall=rainfall,
                forecast=forecast,
            ),
            latitude=latitude,
            longitude=longitude,
        )

    @staticmethod
    def _optional_float(value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def calculate_agricultural_risks(
        self,
        temperature: Optional[float],
        humidity: Optional[float],
        rainfall: Optional[float],
        forecast: Iterable[WeatherForecastDay],
    ) -> List[AgriculturalRisk]:
        risks: List[AgriculturalRisk] = []

        if temperature is not None:
            if temperature >= 38:
                risks.append(AgriculturalRisk(
                    type="heat_stress",
                    severity="high",
                    description="Extreme heat may cause crop stress and increase irrigation demand.",
                ))
            elif temperature >= 32:
                risks.append(AgriculturalRisk(
                    type="high_temperature",
                    severity="medium",
                    description="High temperature may increase evapotranspiration.",
                ))

        if humidity is not None and humidity >= 85:
            risks.append(AgriculturalRisk(
                type="high_humidity",
                severity="medium",
                description="High humidity may increase fungal disease risk.",
            ))

        if rainfall is not None:
            if rainfall >= 20:
                risks.append(AgriculturalRisk(
                    type="heavy_rain",
                    severity="high",
                    description="Heavy rainfall may cause waterlogging or runoff.",
                ))
            elif rainfall >= 5:
                risks.append(AgriculturalRisk(
                    type="moderate_rain",
                    severity="low",
                    description="Moderate rainfall may be useful for soil moisture.",
                ))

        forecast_list = list(forecast)
        forecast_rainfall = [
            day.rainfall
            for day in forecast_list
            if day.rainfall is not None
        ]
        if forecast_rainfall and sum(forecast_rainfall) >= 50:
            risks.append(AgriculturalRisk(
                type="sustained_rainfall",
                severity="medium",
                description="The forecast indicates substantial rainfall that may delay field work.",
            ))

        if not risks:
            risks.append(AgriculturalRisk(
                type="no_major_risk",
                severity="low",
                description="No major agricultural weather risk was identified from current conditions.",
            ))
        return risks
