import os
import re

import vertexai
from dotenv import load_dotenv
from vertexai.generative_models import GenerativeModel, Content, Part

vertexai.init(project='techwings-production', location='europe-west4')

_ = load_dotenv()

gemini_model = os.getenv("GEMINI_MODEL")

class Agent:
    """
    A class representing an AI agent that can engage in coversations using Gemini
    """

    def __init__(self, context=""):
        """
        Initialise the agent with an optional context message for Gemini

        Args:
            context (str, optional): The context (acts like OpenAI's system message)
        """
        self.context = context
        self.messages = []
        self.model = GenerativeModel(gemini_model)
        self.chat = self.model.start_chat()


    def __call__(self, message):
        """
        Add user message, run generation, and save response

        Args:
            message (str): The user's message

        Returns:
            str: The model's reply
        """
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

    def get_prompt(self):
        """
        Format the prompt for Gemini API
        """
        return {
            "context": self.context,
            "messages": self.messages
        }


prompt = """
You run in a loop of Thought, Action, PAUSE, Observation.
At the end of the loop you output an Answer.
Use Thought to describe your thoughts about the question you have been asked. 
Use Action to run one or more actions available to you - then return PAUSE.
Observation will be the result of running those Actions.

Your available actions are: 

calculate_total_price:
e.g. calculate_total_price: apple: 2, banana: 3
Runs a calculation for the total price based on the quantity and prices of the fruits.

get_fruit_price: 
e.g get_fruit_price: apple
returns the price of the fruit when given its name

Example session: 

Question: What is the total price for 2 apples and 3 bananas?
Thought: I should calculate the total price by getting the price of each fruit and then adding them together.
Action: get_fruit_price: apple
PAUSE
Observation: The price of an apple is $2.00

Action: get_fruit_price: banana
PAUSE
Observation: The price of an apple is $1.4

Action: calculate_total_price: apple: 2, banana: 3
PAUSE
Observation: The total price is $5.4

You then output: 

Answer: The total price for 2 apples and 3 bananas is $5.4.
""".strip()

fruit_prices = {
    "apple": 2.00,
    "banana": 1.4,
    "orange": 1.2,
    "grapes": 1.6,
    "kiwi": 0.8,
    "pear": 0.6,
    "mango": 0.4,
    "strawberry": 1.0,
    "blueberry": 0.5,
    "pineapple": 0.7,
    "melon": 0.9,
    "lemon": 0.3,
    "grapefruit": 1.1,
}

def get_fruit_price(fruit):
    if fruit in fruit_prices:
        return f"The price of a {fruit} is ${fruit_prices[fruit]}."
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

action_re = re.compile(r'^Action: (\w+): (\.*)$')

def query(question):
    bot = Agent(prompt)
    result = bot(question) #__call__
    print(result)
    actions = [
        action_re.match(a)
        for a in result.split("\n")
        if action_re.match(a)
    ]
    if actions:
        action, action_input = actions[0].groups()
        if action not in known_actions:
            raise Exception(f"Unknown action: {action}: {action_input}")
        print(f"Running action: {action} with input: {action_input}")
        observation = known_actions[action](action_input)
        print(f"Observation: {observation}")
    else:
        return

query("What is the price of an orange?")

