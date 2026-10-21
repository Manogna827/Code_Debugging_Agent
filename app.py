import os
import uvicorn

from fastapi import FastAPI
from langserve import add_routes

from langchain_core.tools import tool
from langchain_core.runnables import RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent

from pydantic import BaseModel, Field


# ============================================================
# 1. GOOGLE API KEY
# ============================================================

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError(
        "GOOGLE_API_KEY is not set."
    )


# ============================================================
# 2. GEMMA AGENT CORE
# ============================================================

llm_flash = ChatGoogleGenerativeAI(
    model="gemma-4-31b-it",
    google_api_key=GOOGLE_API_KEY,
    temperature=0
)


# ============================================================
# 3. CODE DEBUGGING TOOL
# ============================================================

@tool
def debug_code(
    code: str,
    programming_language: str
) -> str:
    """
    Analyze code and identify the error,
    its location, reason and corrected code.
    """

    prompt = f"""

You are an expert programming debugger.

PROGRAMMING LANGUAGE:
{programming_language}

SOURCE CODE:
{code}

Analyze the code carefully.

Find the actual programming error or errors.

Return ONLY the following four sections:

1. WHAT THE ERROR IS
2. WHERE THE ERROR IS
3. WHY IT HAPPENS
4. CORRECTED CODE

Rules:

- Identify syntax errors.
- Identify runtime errors.
- Identify logical errors when clearly present.
- Mention the exact line or code statement where possible.
- Explain the reason in simple English.
- Provide the complete corrected code.
- Do not invent errors.
- Do not add unnecessary information.
- Do not return JSON.
- Do not return Python dictionaries.
- Do not use Markdown code fences.

If the code has no major error, say:

WHAT THE ERROR IS:
No major error was found.

WHERE THE ERROR IS:
No error location.

WHY IT HAPPENS:
The code appears to be correct.

CORRECTED CODE:
Return the original code.
"""

    try:

        response = llm_flash.invoke(
            prompt
        )

        content = response.content

        if isinstance(
            content,
            str
        ):

            return content.strip()

        return str(content)

    except Exception as e:

        return (
            "Code debugging failed: "
            + str(e)
        )


# ============================================================
# 4. TOOLS AVAILABLE TO GEMMA
# ============================================================

tools = [
    debug_code
]


# ============================================================
# 5. CODE DEBUGGING AGENT
# ============================================================

system_prompt = """

You are a Code Debugging AI Agent.

Your job is to debug programming code.

The student provides only:

1. Programming Language
2. Source Code

You must automatically analyze the code.

Use the debugging tool when necessary.

Your final answer MUST contain exactly these four sections:

WHAT THE ERROR IS:

WHERE THE ERROR IS:

WHY IT HAPPENS:

CORRECTED CODE:

Do not ask the student for an error message.

Do not expose internal reasoning.

Do not expose tool calls.

Do not output JSON.

Do not output Python dictionaries.

Do not add unnecessary sections.

Use simple English suitable for a college student.

The corrected code must be complete and runnable.
"""


# ============================================================
# 6. CREATE LANGCHAIN AGENT
# ============================================================

agent = create_agent(

    model=llm_flash,

    tools=tools,

    system_prompt=system_prompt

)


# ============================================================
# 7. INPUT SCHEMA
# ============================================================

class AgentInput(BaseModel):

    programming_language: str = Field(
        description=(
            "Programming language of the code, "
            "for example Java, Python, C, C++, JavaScript"
        )
    )

    code: str = Field(
        description=(
            "Complete source code that needs debugging"
        )
    )


# ============================================================
# 8. FORMAT INPUT FOR GEMMA AGENT
# ============================================================

def format_for_agent(x) -> dict:

    if isinstance(x, dict):

        programming_language = x.get(
            "programming_language",
            ""
        )

        code = x.get(
            "code",
            ""
        )

    else:

        programming_language = getattr(
            x,
            "programming_language",
            ""
        )

        code = getattr(
            x,
            "code",
            ""
        )


    user_input = f"""

CODE DEBUGGING INPUT

==================================================
PROGRAMMING LANGUAGE
==================================================

{programming_language}


==================================================
SOURCE CODE
==================================================

{code}


==================================================
TASK
==================================================

Analyze and debug this code.

Return exactly:

1. What the error is
2. Where the error is
3. Why it happens
4. Corrected code

Do not ask for an error message.
"""


    return {

        "messages": [

            (
                "user",
                user_input
            )

        ]

    }


# ============================================================
# 9. EXTRACT FINAL TEXT
# ============================================================

def extract_text_response(
    agent_output
) -> str:

    if isinstance(
        agent_output,
        str
    ):

        return agent_output.strip()


    if not isinstance(
        agent_output,
        dict
    ):

        return str(agent_output)


    messages = agent_output.get(
        "messages"
    )


    if messages is None:

        for value in agent_output.values():

            if (
                isinstance(
                    value,
                    dict
                )
                and
                "messages" in value
            ):

                messages = value[
                    "messages"
                ]

                break


    if not messages:

        return (
            "No debugging result was generated."
        )


    for message in reversed(
        messages
    ):

        if message.__class__.__name__ != "AIMessage":

            continue


        content = getattr(
            message,
            "content",
            ""
        )


        if isinstance(
            content,
            str
        ):

            text = content.strip()

            if text:

                return text


        if isinstance(
            content,
            list
        ):

            text_parts = []


            for item in content:

                if isinstance(
                    item,
                    dict
                ):

                    item_type = item.get(
                        "type",
                        ""
                    )


                    if item_type in {
                        "thinking",
                        "reasoning"
                    }:

                        continue


                    if item_type in {
                        "text",
                        "output_text"
                    }:

                        text = item.get(
                            "text",
                            ""
                        )

                        if text:

                            text_parts.append(
                                text
                            )

                elif isinstance(
                    item,
                    str
                ):

                    text_parts.append(
                        item
                    )


            final_text = "\n".join(
                text_parts
            ).strip()


            if final_text:

                return final_text


    return (
        "No readable debugging result "
        "was generated."
    )


# ============================================================
# 10. LANGSERVE CHAIN
# ============================================================

formatted_agent_chain = (

    RunnableLambda(
        format_for_agent
    )

    | agent

    | RunnableLambda(
        extract_text_response
    )

).with_types(

    input_type=AgentInput,

    output_type=str

)


# ============================================================
# 11. FASTAPI APPLICATION
# ============================================================

app = FastAPI(

    title=(
        "Code Debugging AI Agent API"
    ),

    version="1.0",

    description=(
        "Simple Code Debugging AI Agent using "
        "LangChain, Gemma-4-31B-IT, "
        "FastAPI and LangServe."
    )

)


# ============================================================
# 12. LANGSERVE ROUTER
# ============================================================

add_routes(

    app,

    formatted_agent_chain,

    path="/debug-agent"

)


# ============================================================
# 13. HOME ENDPOINT
# ============================================================

@app.get("/")
def home():

    return {

        "message":
            "Code Debugging AI Agent is running.",

        "agent_core":
            "gemma-4-31b-it",

        "endpoint":
            "/debug-agent",

        "playground":
            "/debug-agent/playground/"

    }


# ============================================================
# 14. RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    port = int(

        os.environ.get(
            "PORT",
            8000
        )

    )

    uvicorn.run(

        "app:app",

        host="0.0.0.0",

        port=port

    )
