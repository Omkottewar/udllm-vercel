
import random
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

# -------------------------
# Multiple prompt templates
# -------------------------
PROMPT_TEMPLATES = [
    # Style 1: Motivational + strengths + improvement
    """
    You are an expert football coach AI. Analyze the following player's match performance
    and give constructive, motivational advice. Be inspiring but also realistic.

    Player Stats:
    {stats}

    Write the advice as:
    - Highlight key strengths with praise.
    - Mention 2–3 areas to improve.
    - Give practical training tips.
    - Keep the tone like a supportive coach.
    """,

    # Style 2: Tactical feedback
    """
    You are a tactical football coach. Based on the player’s stats below,
    provide feedback that mixes technical analysis with mindset coaching.

    Player Stats:
    {stats}

    Write the advice as:
    - Focus on tactical awareness and positioning.
    - Highlight 1–2 strong attributes.
    - Suggest drills or practice methods.
    - End with a motivating message.
    """,

    # Style 3: Personal trainer approach
    """
    You are acting as a personal football trainer for this player.
    Analyze the stats and create a short improvement plan.

    Player Stats:
    {stats}

    Write the advice as:
    - Point out fitness & stamina elements.
    - Suggest skill-based exercises.
    - Provide 1 mental/psychological tip.
    - Keep it short and actionable.
    """,

    # Style 4: Match commentary style
    """
    Imagine you are a football commentator turned coach.
    Analyze the player's stats as if reviewing their match highlights.

    Player Stats:
    {stats}

    Write the advice as:
    - Praise strong "moments".
    - Mention weak points as "missed opportunities".
    - Suggest training improvements.
    - Keep tone energetic and engaging.
    """,
]

@app.post("/coach")
async def coach_advice(player_data: PlayerData, request: Request):
    logger.info("📥 Received request from: %s", request.client.host)
    logger.info("📥 Payload: %s", player_data.json())

    if not player_data.data:
        logger.warning("⚠️ No player data provided!")
        return {"advice": "No player data provided."}

    stats = player_data.data[0]

    # Convert stats dict into a readable string
    stats_str = "\n".join([f"{k}: {v}" for k, v in stats.items()])

    # Pick a random template
    selected_prompt = random.choice(PROMPT_TEMPLATES).format(stats=stats_str)

    try:
        logger.info("🤖 Sending request to OpenAI API with random prompt...")
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a professional football coach."},
                {"role": "user", "content": selected_prompt}
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
