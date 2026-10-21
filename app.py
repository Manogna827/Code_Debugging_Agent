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
# 3. CODE ANALYSIS TOOL
# ============================================================

@tool
def code_analysis(
    code: str,
    error_message: str,
    programming_language: str
) -> str:
    """
    Analyze the provided code and identify
    syntax errors, logical errors and runtime errors.
    """

    prompt = f"""

You are an expert programming code debugger.

PROGRAMMING LANGUAGE:
{programming_language}

CODE:
{code}

ERROR MESSAGE:
{error_message}

Analyze the code carefully.

Identify:

1. Syntax errors
2. Runtime errors
3. Logical errors
4. Incorrect imports
5. Incorrect variables or functions
6. Missing code
7. Incorrect indentation
8. Problems related to the given error message

IMPORTANT:

Only identify problems that are actually supported
by the provided code and error.

Do not invent errors.

Explain the problems in simple language.

Return normal readable text.
Do not return JSON.
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
            "Code analysis failed: "
            + str(e)
        )


# ============================================================
# 4. CODE FIX TOOL
# ============================================================

@tool
def code_fix(
    code: str,
    error_message: str,
    analysis: str,
    programming_language: str
) -> str:
    """
    Generate a corrected version of the
    student's code.
    """

    prompt = f"""

You are an expert programming debugger.

PROGRAMMING LANGUAGE:
{programming_language}

ORIGINAL CODE:
{code}

ERROR MESSAGE:
{error_message}

CODE ANALYSIS:
{analysis}

Fix the code.

IMPORTANT RULES:

1. Preserve the original functionality.
2. Fix the actual errors.
3. Do not unnecessarily rewrite the entire program.
4. Do not introduce new libraries unless necessary.
5. Make sure indentation is correct.
6. Make sure syntax is valid.
7. Make sure variables and functions are used correctly.
8. Make sure the corrected code can run.
9. Do not invent missing requirements.
10. Do not add explanations inside the code.

Return ONLY the complete corrected code.

Do not use Markdown code fences.
Do not return JSON.
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
            "Code fixing failed: "
            + str(e)
        )


# ============================================================
# 5. DEBUGGING EXPLANATION TOOL
# ============================================================

@tool
def debugging_explanation(
    original_code: str,
    corrected_code: str,
    error_message: str
) -> str:
    """
    Explain what was wrong with the original code
    and what was changed.
    """

    prompt = f"""

You are a programming instructor.

ORIGINAL CODE:
{original_code}

ERROR MESSAGE:
{error_message}

CORRECTED CODE:
{corrected_code}

Explain the debugging process.

Provide:

1. What caused the error
2. Why the error occurred
3. What was changed
4. Why the corrected code works
5. How the student can avoid this error

Use simple English suitable for a college student.

Do not return JSON.
Return normal readable text.
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
            "Debugging explanation failed: "
            + str(e)
        )


# ============================================================
# 6. TOOLS AVAILABLE TO GEMMA
# ============================================================

tools = [
    code_analysis,
    code_fix,
    debugging_explanation
]


# ============================================================
# 7. CODE DEBUGGING AGENT
# ============================================================

system_prompt = """

You are a Code Debugging AI Agent.

Your purpose is to help students find and fix
programming errors.

The student provides:

1. Programming Language
2. Source Code
3. Error Message


YOUR WORKFLOW:

STEP 1:
Analyze the provided code and error using the
Code Analysis tool.

STEP 2:
Use the Code Fix tool to generate the corrected code.

STEP 3:
Use the Debugging Explanation tool to explain
the problem and solution.

STEP 4:
Combine everything into a final debugging report.


IMPORTANT:

YOU ARE THE AGENT CORE.

You must decide when a tool is required.

Use the tools available to you.

Do not invent information.

Do not assume that a tool was successful if
it returned an error.

Do not expose internal reasoning.

Do not expose tool calls.

Do not output JSON.

Do not output Python dictionaries.


FINAL RESPONSE FORMAT:

CODE DEBUGGING REPORT

Programming Language:

Error:

Problem Identified:

Corrected Code:

Explanation:

How to Avoid This Error:


The final response must be understandable
to a college student.

The Corrected Code must contain the complete
corrected program.

Do not use Markdown code fences around the
corrected code.
"""


# ============================================================
# 8. CREATE LANGCHAIN AGENT
# ============================================================

agent = create_agent(

    model=llm_flash,

    tools=tools,

    system_prompt=system_prompt

)


# ============================================================
# 9. INPUT SCHEMA
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
            "The complete source code that needs debugging"
        )
    )

    error_message: str = Field(
        description=(
            "Compiler error, runtime error, or "
            "description of the problem"
        )
    )


# ============================================================
# 10. FORMAT INPUT FOR GEMMA AGENT
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

        error_message = x.get(
            "error_message",
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

        error_message = getattr(
            x,
            "error_message",
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
ERROR MESSAGE
==================================================

{error_message}


==================================================
TASK
==================================================

Debug this code.

You are the Agent Core.

Select and use the appropriate tools.

Follow the complete workflow:

Code Analysis
        ↓
Code Fix
        ↓
Debugging Explanation
        ↓
Final Debugging Report

Return the final debugging report as readable text.
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
# 11. EXTRACT FINAL TEXT
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
            "No final debugging report "
            "was generated."
        )


    for message in reversed(
        messages
    ):

        # Ignore tool messages
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


                    # Remove Gemini thinking/reasoning
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
        "No final readable debugging "
        "report was generated."
    )


# ============================================================
# 12. LANGSERVE CHAIN
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
# 13. FASTAPI APPLICATION
# ============================================================

app = FastAPI(

    title=(
        "Code Debugging AI Agent API"
    ),

    version="1.0",

    description=(
        "Code Debugging AI Agent using "
        "LangChain, Gemma-4-31B-IT, "
        "FastAPI and LangServe."
    )

)


# ============================================================
# 14. LANGSERVE ROUTER
# ============================================================

add_routes(

    app,

    formatted_agent_chain,

    path="/debug-agent"

)


# ============================================================
# 15. HOME ENDPOINT
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
# 16. RUN APPLICATION
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
