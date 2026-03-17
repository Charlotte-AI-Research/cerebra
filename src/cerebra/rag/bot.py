"""
bot.py — Cerebra Discord bot

Responds when mentioned in a Discord server,
using the Cerebra RAG pipeline to answer questions.
"""

import os
import discord
from dotenv import load_dotenv

from chat import ask

load_dotenv()

TOKEN = os.getenv("DISCORD_TOKEN")


class CerebraBot(discord.Client):
    async def on_ready(self):
        print(f"Logged in as {self.user} (ID: {self.user.id})")
        print("Cerebra is ready.")
        print("------")

    async def on_message(self, message):
        # Ignore messages from the bot itself
        if message.author == self.user:
            return

        # Only respond when mentioned
        if self.user not in message.mentions:
            return

        # Strip the mention from the query
        query = message.content.replace(f"<@{self.user.id}>", "").strip()

        if not query:
            await message.channel.send("Hey! Ask me anything about CAIR or UNC Charlotte. 👋")
            return

        async with message.channel.typing():
            try:
                response = ask(query)

                if not response or not response.strip():
                    await message.channel.send(
                        "I wasn't able to generate a response. Try rephrasing your question!"
                    )
                    return

                # Split into chunks if response exceeds Discord's 2000 char limit
                if len(response) > 2000:
                    for i in range(0, len(response), 2000):
                        chunk = response[i : i + 2000]
                        if chunk.strip():
                            await message.channel.send(chunk)
                else:
                    await message.channel.send(response)

            except Exception as e:
                print(f"[ERROR] Failed to process query: {e}")
                await message.channel.send(
                    "Sorry, I ran into an error. Please try again!"
                )


intents = discord.Intents.default()
intents.message_content = True

client = CerebraBot(intents=intents)

if __name__ == "__main__":
    if not TOKEN:
        print("[ERROR] DISCORD_TOKEN not found in .env")
    else:
        client.run(TOKEN)