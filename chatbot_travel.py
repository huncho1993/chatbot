import openai
import os
import time
import random
from dotenv import load_dotenv
from datetime import datetime


# Load OpenAI API Key from .env
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY") #in order to work require my secret API key


class GPTTravelChatbot:
    def __init__(self):
        self.user_name = None
        self.history = []
        self.default_prompt = (
            "You are TravelBuddy, a friendly travel assistant helping users plan trips around the world. "
            "You provide helpful, fun, and informative answers about destinations, activities, seasons, budgets, and travel tips. "
            "Keep responses clear, engaging, and travel-themed."
        )

    def ask_openai(self, prompt):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",  # Use gpt-4 if you have access
                messages=[{"role": "system", "content": self.default_prompt}] +
                        self.history + [{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=500
            )
            answer = response["choices"][0]["message"]["content"]
            self.history.append({"role": "user", "content": prompt})
            self.history.append({"role": "assistant", "content": answer})
            return answer
        except Exception as e:
            return f"Error: {e}"

    def chat(self):
        print("🌍 Welcome to TravelBuddy - AI Travel Assistant")
        print("Type 'exit' to end or 'reset' to start over.\n")

        while True:
            user_input = input("You: ")
            if user_input.lower() == "exit":
                print("TravelBuddy: Bon voyage! Safe travels! 🌟")
                break
            elif user_input.lower() == "reset":
                self.history = []
                print("TravelBuddy: Conversation reset. Ask me anything about travel!")
                continue
            elif user_input.lower() == "history":
                print("\n=== Conversation History ===")
                for msg in self.history:
                    role = msg["role"]
                    content = msg["content"]
                    print(f"{role.capitalize()}: {content}")
                print("===========================\n")
                continue

            response = self.ask_openai(user_input)
            print(f"TravelBuddy: {response}")

if __name__ == "__main__":
    bot = GPTTravelChatbot()
    bot.chat()
