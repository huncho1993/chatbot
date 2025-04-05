import openai
import time
from typing import List, Dict, Any

# IMPORTANT: Replace this with your actual OpenAI API key
# In production, use environment variables: os.environ.get("OPENAI_API_KEY")
openai.api_key = "" #my secret api key

class TravelChatbot:
    def __init__(self):
        # Initialize conversation history
        self.conversation_history = []
        self.user_preferences = {
            "destination": None,
            "budget": None,
            "travel_style": None,
            "interests": [],
            "travel_dates": None,
            "group_composition": None
        }
        
        # Set up system message that defines the chatbot's behavior
        self.system_message = {
            "role": "system",
            "content": """You are TravelBuddy, a helpful travel assistant chatbot. 
            Your purpose is to provide accurate, concise travel information and suggestions to users. 
            You have knowledge about global destinations, attractions, travel tips, and basic itinerary planning. 
            You should be friendly, enthusiastic about travel, but also practical. 
            Never recommend illegal activities or unsafe travel practices. 
            If you don't know specific details about a destination, acknowledge this and provide general advice instead.
            Use a conversational tone and ask clarifying questions when needed."""
        }
    
    def extract_preferences(self, message: str) -> None:
        """Extract travel preferences from user messages and update the user_preferences dict"""
        # Destination detection
        destinations = ["europe", "asia", "africa", "australia", "japan", "thailand", 
                        "france", "italy", "spain", "greece", "bali", "mexico"]
        for dest in destinations:
            if dest.lower() in message.lower():
                self.user_preferences["destination"] = dest
                
        # Budget detection
        budget_terms = {
            "budget": ["cheap", "budget", "affordable", "inexpensive"],
            "moderate": ["moderate", "mid-range", "reasonable"],
            "luxury": ["luxury", "expensive", "high-end", "5-star"]
        }
        
        for budget_level, terms in budget_terms.items():
            for term in terms:
                if term in message.lower():
                    self.user_preferences["budget"] = budget_level
                    
        # Travel interests detection
        interests = ["beach", "mountain", "hiking", "food", "culture", "history", 
                    "museums", "nightlife", "shopping", "adventure", "relaxation"]
        for interest in interests:
            if interest.lower() in message.lower():
                if interest not in self.user_preferences["interests"]:
                    self.user_preferences["interests"].append(interest)
        
        # Simple date detection
        months = ["january", "february", "march", "april", "may", "june", 
                 "july", "august", "september", "october", "november", "december"]
        seasons = ["spring", "summer", "fall", "autumn", "winter"]
        
        for month in months:
            if month.lower() in message.lower():
                self.user_preferences["travel_dates"] = month.capitalize()
                
        for season in seasons:
            if season.lower() in message.lower():
                self.user_preferences["travel_dates"] = season.capitalize()
    
    def generate_context_prompt(self) -> str:
        """Generate a context prompt based on extracted user preferences"""
        context = "Based on our conversation, I understand that "
        
        preferences_added = False
        
        # Add known preferences to context
        if self.user_preferences["destination"]:
            context += f"you're interested in traveling to {self.user_preferences['destination']}. "
            preferences_added = True
        
        if self.user_preferences["budget"]:
            context += f"You're looking for {self.user_preferences['budget']} options. "
            preferences_added = True
            
        if self.user_preferences["interests"]:
            interests_str = ", ".join(self.user_preferences["interests"])
            context += f"You've expressed interest in {interests_str}. "
            preferences_added = True
            
        if self.user_preferences["travel_dates"]:
            context += f"You're planning to travel during {self.user_preferences['travel_dates']}. "
            preferences_added = True
        
        if not preferences_added:
            return ""  # No preferences detected yet
            
        context += "Please keep these preferences in mind when providing travel recommendations."
        return context
    
    def process_message(self, user_message: str) -> str:
        """Process the user message and generate a response"""
        # Extract any preferences from the message
        self.extract_preferences(user_message)
        
        # Add user message to conversation history
        self.conversation_history.append({"role": "user", "content": user_message})
        
        # Build the messages to send to the API
        messages = [self.system_message]
        
        # Add context if we have user preferences
        context_prompt = self.generate_context_prompt()
        if context_prompt:
            messages.append({"role": "system", "content": context_prompt})
        
        # Add additional prompt engineering techniques for specific scenarios
        if "safety" in user_message.lower() or "safe" in user_message.lower():
            safety_prompt = {
                "role": "system",
                "content": """When discussing destination safety:
                1. Acknowledge the user's concern about safety
                2. Provide general safety information about the destination 
                3. Recommend checking official government travel advisories
                4. Offer practical safety tips
                5. Avoid making definitive safety guarantees"""
            }
            messages.append(safety_prompt)
            
        # Add few-shot examples for certain types of questions
        if "recommend" in user_message.lower() or "suggestion" in user_message.lower():
            few_shot_examples = {
                "role": "system",
                "content": """Example of how to respond to recommendation requests:
                
                User: "Can you recommend places to visit in Italy?"
                TravelBuddy: "Italy offers amazing destinations for every type of traveler! Here are some highlights:
                
                - Rome: For history lovers (Colosseum, Vatican, Roman Forum)
                - Florence: Renaissance art and architecture (Uffizi, Duomo)
                - Venice: Unique canal city with stunning architecture
                - Amalfi Coast: Breathtaking coastal scenery and charming towns
                - Tuscany: Rolling hills, vineyards, and medieval villages
                
                What interests you most about Italy? Art, history, food, nature, or something else?"
                """
            }
            messages.append(few_shot_examples)
        
        # Add the last few conversation exchanges for context (limiting to prevent token overflow)
        # Include a maximum of 5 recent exchanges (10 messages - 5 from user, 5 from assistant)
        recent_history = self.conversation_history[-10:] if len(self.conversation_history) > 10 else self.conversation_history
        messages.extend(recent_history)
        
        # Call the OpenAI API with retry logic for rate limiting
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-4",  # You can also use "gpt-3.5-turbo" for lower cost
                    messages=messages,
                    max_tokens=500,
                    temperature=0.7
                )
                
                assistant_response = response.choices[0].message["content"]
                
                # Add the assistant's response to the conversation history
                self.conversation_history.append({"role": "assistant", "content": assistant_response})
                
                return assistant_response
                
            except openai.error.RateLimitError:
                if attempt < max_retries - 1:
                    # Exponential backoff: wait 1s, then 2s, then 4s, etc.
                    time.sleep(2 ** attempt)
                    continue
                else:
                    return "I'm experiencing high demand right now. Please try again in a moment."
                    
            except Exception as e:
                return f"I apologize, but I encountered an error: {str(e)}"
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Return the full conversation history"""
        return self.conversation_history
    
    def reset_conversation(self) -> None:
        """Reset the conversation history and user preferences"""
        self.conversation_history = []
        self.user_preferences = {
            "destination": None,
            "budget": None,
            "travel_style": None,
            "interests": [],
            "travel_dates": None,
            "group_composition": None
        }


# Interactive command line interface for testing
def run_interactive_chatbot():
    print("🌍 TravelBuddy Chatbot")
    print("Type 'exit' to end the conversation or 'reset' to start over.\n")
    
    chatbot = TravelChatbot()
    
    while True:
        user_input = input("You: ")
        
        if user_input.lower() == "exit":
            print("\nThank you for using TravelBuddy! Have a great trip! 👋")
            break
            
        if user_input.lower() == "reset":
            chatbot.reset_conversation()
            print("\nConversation has been reset. Let's start again! 🔄\n")
            continue
            
        response = chatbot.process_message(user_input)
        print(f"\nTravelBuddy: {response}\n")


# Simple web interface using Flask (if available)
def create_web_app():
    try:
        from flask import Flask, request, jsonify, render_template_string
        
        app = Flask(__name__)
        chatbot = TravelChatbot()
        
        # Simple HTML template for the chatbot interface
        HTML_TEMPLATE = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>TravelBuddy Chatbot</title>
            <style>
                body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
                #chat-container { height: 400px; overflow-y: scroll; border: 1px solid #ccc; padding: 10px; margin-bottom: 10px; }
                #user-input { width: 80%; padding: 8px; }
                button { padding: 8px 15px; }
                .user-message { background-color: #e6f7ff; padding: 8px; border-radius: 10px; margin: 5px 0; text-align: right; }
                .bot-message { background-color: #f2f2f2; padding: 8px; border-radius: 10px; margin: 5px 0; }
            </style>
        </head>
        <body>
            <h1>🌍 TravelBuddy Chatbot</h1>
            <div id="chat-container"></div>
            <div>
                <input type="text" id="user-input" placeholder="Ask about travel destinations...">
                <button onclick="sendMessage()">Send</button>
                <button onclick="resetChat()">Reset Chat</button>
            </div>
            
            <script>
                function sendMessage() {
                    const userInput = document.getElementById('user-input');
                    const message = userInput.value.trim();
                    
                    if (message) {
                        // Add user message to chat
                        addMessage(message, 'user');
                        userInput.value = '';
                        
                        // Send to server
                        fetch('/api/chat', {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json',
                            },
                            body: JSON.stringify({ message: message }),
                        })
                        .then(response => response.json())
                        .then(data => {
                            // Add bot response to chat
                            addMessage(data.response, 'bot');
                        })
                        .catch(error => {
                            console.error('Error:', error);
                            addMessage('Sorry, something went wrong.', 'bot');
                        });
                    }
                }
                
                function addMessage(message, sender) {
                    const chatContainer = document.getElementById('chat-container');
                    const messageDiv = document.createElement('div');
                    messageDiv.className = sender === 'user' ? 'user-message' : 'bot-message';
                    messageDiv.textContent = message;
                    chatContainer.appendChild(messageDiv);
                    chatContainer.scrollTop = chatContainer.scrollHeight;
                }
                
                function resetChat() {
                    fetch('/api/reset', { method: 'POST' })
                    .then(response => response.json())
                    .then(data => {
                        document.getElementById('chat-container').innerHTML = '';
                        addMessage('Chat has been reset. How can I help with your travel plans?', 'bot');
                    });
                }
                
                // Press Enter to send
                document.getElementById('user-input').addEventListener('keypress', function(e) {
                    if (e.key === 'Enter') {
                        sendMessage();
                    }
                });
                
                // Initial greeting
                addMessage('Hello! I\'m TravelBuddy, your personal travel assistant. How can I help you plan your next adventure?', 'bot');
            </script>
        </body>
        </html>
        """
        
        @app.route('/')
        def index():
            return render_template_string(HTML_TEMPLATE)
            
        @app.route('/api/chat', methods=['POST'])
        def chat():
            user_message = request.json.get('message', '')
            response = chatbot.process_message(user_message)
            return jsonify({'response': response})
            
        @app.route('/api/reset', methods=['POST'])
        def reset():
            chatbot.reset_conversation()
            return jsonify({'status': 'success'})
            
        return app
        
    except ImportError:
        print("Flask not installed. Running in console mode instead.")
        return None


if __name__ == "__main__":
    # Try to run web interface, fall back to console if Flask is not available
    app = create_web_app()
    
    if app:
        print("Starting web interface on http://127.0.0.1:5000")
        app.run(debug=True)
    else:
        run_interactive_chatbot()
