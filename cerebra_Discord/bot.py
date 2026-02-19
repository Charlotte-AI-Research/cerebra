import os
import discord
from dotenv import load_dotenv
from rag import RAGPipeline

load_dotenv()

TOKEN = os.getenv('DISCORD_TOKEN')

class CAIRBot(discord.Client):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print("Initializing RAG Pipeline...")
        self.rag = RAGPipeline()
        print("RAG Pipeline Ready.")

    async def on_ready(self):
        print(f'Logged in as {self.user} (ID: {self.user.id})')
        print('------')

    async def on_message(self, message):
        # Don't respond to ourselves
        if message.author == self.user:
            return

        if self.user in message.mentions:
            async with message.channel.typing():
                query = message.content.replace(f'<@{self.user.id}>', '').strip()

                if not query:
                    await message.channel.send("Hello! How can I help you with CAIR today?")
                    return

                try:
                    response = self.rag.query(query)
                    # Split into chunks if response exceeds Discord's 2000 char limit
                    if len(response) > 2000:
                        for i in range(0, len(response), 2000):
                            await message.channel.send(response[i:i+2000])
                    else:
                        await message.channel.send(response)
                except Exception as e:
                    print(f"Error processing query: {e}")
                    await message.channel.send("Sorry, I encountered an error while processing your request.")

intents = discord.Intents.default()
intents.message_content = True

client = CAIRBot(intents=intents)

if __name__ == "__main__":
    if not TOKEN:
        print("Error: DISCORD_TOKEN not found in .env")
    else:
        client.run(TOKEN)