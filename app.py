""" ChatGPT chat bot web app
"""
import json
import logging
import math
import os
import re
import secrets
import string
import sys
from datetime import datetime
from typing import List, Optional

import openai
import tiktoken
import uvicorn
import yaml
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel
from starlette.responses import JSONResponse

from util.html import wrap_urls

# Load environment variables from .env file
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)  # Log output will go to stdout

# Set the OpenAI API key from the environment variable
openai.api_key = os.environ.get("OPENAI_API_KEY")

MODEL_NAME = "gpt-4o-mini"
MAX_RESPONSE_TOKENS = 640
TOKEN_LIMIT = 16384


class Message(BaseModel):
    """Message model"""

    role: str
    content: str
    response_time: Optional[float] = 0.0


class History(BaseModel):
    """History model"""

    history: List[Message]


class ConversationData(BaseModel):
    """Conversation data model"""

    profile_id: Optional[str] = ""
    history: List[Message]
    message: Optional[str] = ""
    user_profile: Optional[str] = ""
    agent: Optional[str] = ""
    language: Optional[str] = ""


class UserProfile(BaseModel):
    """User profile model"""

    profile: str


class FileName(BaseModel):
    """User profile model"""

    file_name: str


class UpdateSystemCardData(FileName):
    updated_system_card_content: str


# Create a FastAPI application instance
app = FastAPI()
http_basic_auth = HTTPBasic()


def generate_password(length=11):
    """Generate a random password"""
    logging.debug("generate_password(length=%s)", length)
    all_characters = string.ascii_letters + string.digits + string.punctuation
    password = "".join(secrets.choice(all_characters) for i in range(length))
    return password


# Replace these values with your actual username and password
CORRECT_USERNAME = os.environ.get("FIN_GENIE_USERNAME", generate_password())
CORRECT_PASSWORD = os.environ.get("FIN_GENIE_PASSWORD", generate_password())


def get_current_user(credentials: HTTPBasicCredentials = Depends(http_basic_auth)):
    """Get current user"""
    correct_username_and_password = secrets.compare_digest(
        credentials.username, CORRECT_USERNAME
    ) and secrets.compare_digest(credentials.password, CORRECT_PASSWORD)
    if not correct_username_and_password:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username


CORS_ORIGINS = os.environ.get("CORS_ORIGINS", "http://localhost:5173").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)


def get_system_card(agent: str | None = ""):
    """Get system card"""
    agent_path = "./agents/agent.txt"  # default path
    if agent and agent.upper().startswith("EDU"):
        agent_path = "./agents/educate.txt"
    # Read the content of the agent "system_card.txt" file
    with open(agent_path, "r", encoding="UTF-8") as sys_card:
        system = sys_card.read()
        return system


# Read the content of the "syscard_scoring.txt" file
with open("syscard_scoring.txt", "r", encoding="UTF-8") as file:
    syscard_scoring = file.read()


def num_tokens_from_messages(messages, model=MODEL_NAME):
    """Return the number of tokens used by a list of messages. Based on cookbook:
    https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb.
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:  # model not found
        encoding = tiktoken.get_encoding("cl100k_base")  # use default encoding
    if model in {
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
    }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = (
            4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        )
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif "gpt-3.5-turbo" in model:
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613")
    elif "gpt-4" in model:
        return num_tokens_from_messages(messages, model="gpt-4-0613")
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}."""
        )
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens


def prepare_conversation(
    profile_id: str | None, history: List[Message], agent: str | None = ""
):
    """Prepare conversation for the OpenAI API"""
    # Create a list of message dictionaries for the OpenAI API
    messages = [{"role": msg.role, "content": msg.content} for msg in history]
    system_card = get_system_card(agent)

    # Append the system message from the "system_card.txt" file
    messages.append({"role": "system", "content": system_card})

    conv_history_tokens = num_tokens_from_messages(messages)

    # If the conversation history is too long, remove messages from the start of the history
    while conv_history_tokens + MAX_RESPONSE_TOKENS >= TOKEN_LIMIT:
        del messages[1]
        conv_history_tokens = num_tokens_from_messages(messages)

    return messages


# Define the route for generating a response using the OpenAI API
# The route accepts POST requests
@app.post("/generate")
def generate(data: ConversationData) -> JSONResponse:
    """Generate ChatGPT recommendation"""
    messages = prepare_conversation(data.profile_id, data.history, data.agent)

    # Call the OpenAI API to generate a response using the GPT-3.5-turbo model
    start_time = datetime.now()
    response = openai.ChatCompletion.create(
        model=MODEL_NAME,
        messages=messages,
        max_tokens=MAX_RESPONSE_TOKENS,
        n=1,
        temperature=0.3333333333333333,
    )
    response_time = datetime.now().timestamp() - start_time.timestamp()
    json_response = jsonable_encoder(
        {
            "message": response["choices"][0]["message"]["content"].strip(),
            "response_time": response_time,
        }
    )
    # Return the AI message as a JSON response
    return JSONResponse(content=json_response, status_code=200)


# Define the route for generating a response using the OpenAI API
# The route accepts POST requests
@app.post("/generate-stream", response_class=StreamingResponse)
def generate_stream(data: ConversationData) -> StreamingResponse:
    """Generate ChatGPT recommendation"""
    messages = prepare_conversation(data.profile_id, data.history, data.agent)

    # Call the OpenAI API to generate a response using the GPT-3.5-turbo model
    response = openai.ChatCompletion.create(
        model=MODEL_NAME,
        messages=messages,
        max_tokens=MAX_RESPONSE_TOKENS,
        n=1,
        temperature=0.3333333333333333,
        stream=True,
    )

    async def openai_streamer(response):
        """Stream the response from OpenAI"""
        wrapped_chunks = {}
        for event in response:
            if not event["choices"][0]["finish_reason"]:  # end of stream reached
                event_text = event["choices"][0]["delta"]["content"]
                yield event_text.encode()

    # Return the AI message as a StreamingResponse
    return StreamingResponse(openai_streamer(response), media_type="text/event-stream")


# Route for fetching saved user profiles
@app.get("/profiles")
def send_profiles_info() -> JSONResponse:
    """Send profile information"""
    info = {}
    profiles = os.listdir("./profiles")
    for profile in profiles:
        if "json" in profile:
            with open(f"./profiles/{profile}", encoding="utf-8") as f_profile:
                info[profile.split(".json")[0]] = json.load(f_profile)
                summary = open(f"./profiles/{profile.split('.json')[0]}-summary.txt")
                info[profile.split(".json")[0]]["summary"] = summary.readlines()[0]
    return JSONResponse(content=info, status_code=200)


# Route for generating profile summary
@app.post("/summary")
def summarize(profile: UserProfile) -> JSONResponse:
    """Summarize financial profile with ChatGPT"""
    content = (
        f"Here's JSON describing person's financial situation: {profile}"
        + " summarize. focus on income, debts/loans, credit cards, assets. "
        + "make it as an introduction to the person. do it in 1-2 sentences. include numbers."
    )
    messages = [{"role": "user", "content": content}]
    response = openai.ChatCompletion.create(
        model=MODEL_NAME,
        messages=messages,
        max_tokens=MAX_RESPONSE_TOKENS,
        n=1,
        temperature=0.2,
    )
    return JSONResponse(
        content=response["choices"][0]["message"]["content"].strip(), status_code=200
    )


# Route for savings conversation history
@app.post("/save_history")
def save_history(data: History) -> JSONResponse:
    """Save conversation history to file"""
    history = data.history

    # Create a list of message dictionaries for the OpenAI API
    messages = [{"role": msg.role, "content": msg.content} for msg in history]
    if messages == []:
        return JSONResponse(content="No data to write", status_code=400)

    try:
        file_path = "conversationHistory.txt"
        with open(file_path, "w", encoding="utf-8") as _file:
            for message in messages:
                _file.write(json.dumps(message))
                _file.write("\n")
        return JSONResponse(
            content="Conversation History successfully written to file", status_code=200
        )
    except Exception as ex:  # pylint: disable=broad-except
        return JSONResponse(content=str(ex), status_code=500)


# Route for conversation scoring
@app.post("/score")
def score_conversation(data: ConversationData):
    """Score conversation with ChatGPT"""
    conv = "Evaluate the following conversation according to your system card:\n"
    response_time = 0.0
    total: int = 0
    for msg in data.history:
        conv += f"{msg.role}: {msg.content}\n"
        if msg.response_time:
            response_time += float(msg.response_time)
            total += 1
    avg_resp_time = response_time / total
    response_time_score = 9
    if avg_resp_time > 30:
        response_time_score = 0
    else:
        response_time_score = math.floor(-0.3 * avg_resp_time + 9.15)
    messages = [{"role": "user", "content": conv}]
    messages.append({"role": "system", "content": syscard_scoring})
    response = openai.ChatCompletion.create(
        model=MODEL_NAME,
        messages=messages,
        max_tokens=MAX_RESPONSE_TOKENS,
        n=1,
        temperature=0.2,
    )
    pattern = r'"customer_satisfaction"\s*:\s*(\d+)'
    match = re.search(pattern, response["choices"][0]["message"]["content"].strip())
    csat_score = -1 if not match else int(match.group(1))
    pattern = r'"nps_score"\s*:\s*(\d+)'
    match = re.search(pattern, response["choices"][0]["message"]["content"].strip())
    nps_score = -1 if not match else int(match.group(1))
    pattern = r'"ces_score"\s*:\s*(\d+)'
    match = re.search(pattern, response["choices"][0]["message"]["content"].strip())
    ces_score = -1 if not match else int(match.group(1))
    pattern = r'"response_quality"\s*:\s*(\d+)'
    match = re.search(pattern, response["choices"][0]["message"]["content"].strip())
    response_quality = -1 if not match else int(match.group(1))
    json_response = jsonable_encoder(
        {
            "response_time_score": response_time_score,
            "csat_score": csat_score,
            "nps_score": nps_score,
            "ces_score": ces_score,
            "response_quality": response_quality,
        }
    )
    return JSONResponse(content=json_response, status_code=200)


@app.get("/get-system-card-list")
def get_system_card_list():
    try:
        directory_path = os.path.join(os.getcwd(), "agents")
        file_list = [
            f
            for f in os.listdir(directory_path)
            if os.path.isfile(os.path.join(directory_path, f))
        ]
        for file in file_list:
            return JSONResponse(file_list, status_code=200)
    except Exception as ex:  # pylint: disable=broad-except
        return JSONResponse(content=str(ex), status_code=500)


@app.post("/read-system-card/{file_name}")
def read_system_card(file_name: str) -> JSONResponse:
    try:
        directory_path = os.path.join(os.getcwd(), "agents", file_name)
        if os.path.exists(directory_path):
            with open(directory_path, "r") as file:
                # Read the file contents
                file_contents = file.read()
            return JSONResponse(file_contents, status_code=200)

        else:
            return JSONResponse(
                content=f"The file '{directory_path}' does not exist.", status_code=600
            )
    except Exception as ex:  # pylint: disable=broad-except
        return JSONResponse(content=str(ex), status_code=500)


@app.put("/update-system-card")
def update_system_card(updated_data: UpdateSystemCardData) -> JSONResponse:
    try:
        directory_path = os.path.join(os.getcwd(), "agents", updated_data.file_name)
        if os.path.exists(directory_path):
            with open(directory_path, "w") as file:
                # Write the new content to the file
                file.write(updated_data.updated_system_card_content)
            return JSONResponse("System card data updated sucessfully", status_code=200)
        else:
            return JSONResponse(
                content=f"The file '{directory_path}' does not exist.", status_code=600
            )
    except Exception as ex:  # pylint: disable=broad-except
        return JSONResponse(content=str(ex), status_code=500)


# Start the Flask application, listening on all interfaces and port 8080
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8080)
