import os
import re
import vertexai
from dotenv import load_dotenv
from vertexai.generative_models import GenerativeModel, Content, Part

_ = load_dotenv()

gemini_model = os.getenv("GEMINI_MODEL")
google_project = os.getenv("GOOGLE_PROJECT")
project_region = os.getenv("PROJECT_REGION")

vertexai.init(project=google_project, location=project_region)

class Agent:
    def __init__(self, context=""):
        self.context = context
        self.messages = []
        self.model = GenerativeModel(gemini_model)

    def get_prompt(self):
        return {
            "context": self.context,
            "messages": self.messages
        }

    def step(self, message):
        self.messages.append({"role": "user", "content": message})
        prompt = self.get_prompt()
        history = []

        if prompt["context"]:
            history.append(Content(role="model", parts=[Part.from_text(prompt["context"])]))

        for msg in prompt["messages"][:-1]:
            history.append(Content(role=msg["role"], parts=[Part.from_text(msg["content"])]))

        chat = self.model.start_chat(history=history)
        last_user_message = prompt["messages"][-1]["content"]
        response = chat.send_message(last_user_message)
        self.messages.append({"role": "model", "content": response.text})
        return response.text

    def inject(self, message):
        self.messages.append({"role": "user", "content": message})
        return

# --- Actions ---
fruit_prices = {
    "apple": 2.00,
    "banana": 1.4,
    "orange": 1.2,
    "grapes": 1.6
}

def get_fruit_price(fruit):
    print(f"Calling get_fruit_price({fruit})")
    if fruit in fruit_prices:
        return f"The price of an {fruit} is ${fruit_prices[fruit]:.2f}."
    else:
        return f"Sorry, I don't know the price of {fruit}."

def calculate_total_price(fruits):
    total = 0.0
    fruit_list = fruits.split(",")
    for item in fruit_list:
        fruit, quantity = item.split(": ")
        quantity = int(quantity)
        if fruit in fruit_prices:
            total += fruit_prices[fruit] * quantity
        else:
            return f"Sorry, I don't know the price of {fruit}."
    return f"The total price is ${total:.2f}."

known_actions = {
    "get_fruit_price": get_fruit_price,
    "calculate_total_price": calculate_total_price
}

action_re = re.compile(r'Action:\s*(\w+)\s*:\s*(.+)')

# --- Prompt ---
prompt = """
You run in a loop of Thought, Action, PAUSE, Observation.
At the end of the loop you output an Answer.
Use Thought to describe your thoughts about the question you have been asked. 
Use Action to run one or more actions available to you — then return **PAUSE only**.
Always end your message with PAUSE when you take an action.

Your available actions are: 

calculate_total_price:
e.g. calculate_total_price: apple: 2, banana: 3
Runs a calculation for the total price based on the quantity and prices of the fruits.

get_fruit_price: 
e.g get_fruit_price: apple
returns the price of the fruit when given its name

Example session: 

Question: What is the total price for 2 apples and 3 bananas?
Thought: I should get the price of each fruit first.

Action: get_fruit_price: apple
PAUSE

---

(Then the system responds with:)

Observation: The price of an apple is $2.00

---

Thought: Now I need the price of the bananas.

Action: get_fruit_price: banana
PAUSE

---

(Then the system responds with:)

Observation: The price of a banana is $1.4

---

Thought: Now I can calculate the total.

Action: calculate_total_price: apple: 2, banana: 3
PAUSE

---

(Then the system responds with:)
Observation: The total price is $5.4

---

Answer: The total price for 2 apples and 3 bananas is $5.4.

""".strip()

# --- Query runner ---
def query(question):
    agent = Agent(prompt)
    current_prompt = f"Question: {question}"

    while True:
        response = agent.step(current_prompt)
        print(response)

        if "Answer:" in response:
            break

        if "PAUSE" not in response:
            print("Expected 'PAUSE' in response. Exiting early.")
            break

        matches = [
            action_re.match(line)
            for line in response.splitlines()
            if action_re.match(line)
        ]

        if not matches:
            print("No actions found. Exiting.")
            break

        for match in matches:
            action, action_input = match.groups()
            if action not in known_actions:
                print(f"Unknown action: {action}")
                continue
            result = known_actions[action](action_input.strip())
            current_prompt = f"Observation: {result}"
            agent.inject(current_prompt)

# --- Test ---
query("What is the price of an orange?")
