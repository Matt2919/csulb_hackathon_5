import discord
from discord.ext import commands, tasks
import os
from dotenv import load_dotenv

#Initialize
load_dotenv()
TOKEN = os.getenv("DISCORD_TOKEN")

GESTURE_FILE = "/Users/andrewntran/PycharmProjects/csulb_hackathon_5/gestures.txt"
CHANNEL_ID = 1431763684849487975

intents = discord.Intents.default()
bot = commands.Bot(command_prefix="!", intents=intents)

previous_length = 0


@bot.event
async def on_ready():
    print(f"Logged in as {bot.user}")
    print(f"Watching: {GESTURE_FILE}")
    watch_gestures.start()


@tasks.loop(seconds=2)
async def watch_gestures():
    global previous_length

    if not os.path.exists(GESTURE_FILE):
        print(f"gestures.txt not found at: {GESTURE_FILE}")
        return

    with open(GESTURE_FILE, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    if len(lines) > previous_length:
        new_lines = lines[previous_length:]
        previous_length = len(lines)

        channel = bot.get_channel(CHANNEL_ID)
        if not channel:
            print("Channel not found. Check channel ID or permissions.")
            return

        for gesture in new_lines:
            print(f"New gesture detected: {gesture}")
            await channel.send(f"Gesture detected: **{gesture}**")


print("Token loaded:", bool(TOKEN)) 
bot.run(TOKEN)
