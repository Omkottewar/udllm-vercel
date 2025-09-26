import os
from fastapi import FastAPI, Request
from pydantic import BaseModel
from openai import OpenAI
import logging
from fastapi.middleware.cors import CORSMiddleware

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for testing, allow all
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
class PlayerData(BaseModel):
    status: str
    data: list
    video: list = []
    avg: float
    point: int

@app.post("/coach")
async def coach_advice(player_data: PlayerData, request: Request):
    # Log incoming request info
    logger.info("📥 Received request from: %s", request.client.host)
    logger.info("📥 Payload: %s", player_data.json())

    if not player_data.data:
        logger.warning("⚠️ No player data provided!")
        return {"advice": "No player data provided."}

    stats = player_data.data[0]

    prompt = f"""
    You are an expert football coach AI. Analyze the following player's match performance stats
    and give constructive, motivational advice. Focus on both strengths and areas to improve.

    Player Stats:
    Position: {stats.get("position")}
    Goals: {stats.get("goals")}
    Assists: {stats.get("assists")}
    Time Played: {stats.get("time")} minutes
    Streak: {stats.get("streak")}

    Technical Attributes:
    Two-Footed: {stats.get("twoFooted", {}).get("value")}
    Dribbling: {stats.get("dribbling", {}).get("value")}
    First Touch: {stats.get("firstTouch", {}).get("value")}
    Agility: {stats.get("agility", {}).get("value")}
    Speed: {stats.get("speed", {}).get("value")}
    Power: {stats.get("power", {}).get("value")}

    Highlights:
    Workrate: {stats.get("highlights", {}).get("workrate")}
    Ball Possessions: {stats.get("highlights", {}).get("ballPossessions")}
    Total Distance: {stats.get("highlights", {}).get("totalDistance")} km
    Sprint Distance: {stats.get("highlights", {}).get("sprintDistance")} km
    Top Speed: {stats.get("highlights", {}).get("topSpeed")}
    Kicking Power: {stats.get("highlights", {}).get("kickingPower")}

    Trends:
    Two-footed: {stats.get("two_footed_trend")}
    Dribbling: {stats.get("dribbling_trend")}
    First Touch: {stats.get("first_touch_trend")}
    Agility: {stats.get("agility_trend")}
    Speed: {stats.get("speed_trend")}
    Power: {stats.get("power_trend")}

    Write the advice as:
    - Highlight key strengths with praise.
    - Point out 2–3 specific areas to improve.
    - Give practical, position-specific training tips.
    - Keep tone encouraging, as if motivating a real player.
    """

    try:
        logger.info("🤖 Sending request to OpenAI API...")
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a professional football coach."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=400,
            temperature=0.7,
        )

        advice = response.choices[0].message.content
        logger.info("✅ Received advice from OpenAI API")

    except Exception as e:
        logger.error("❌ OpenAI API call failed: %s", e, exc_info=True)
        return {"advice": "Error fetching advice from AI."}

    logger.info("📤 Returning advice to client")
    return {"advice": advice}


