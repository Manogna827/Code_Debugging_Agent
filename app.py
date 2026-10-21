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
    programming_language: str,
    code: str,
    error_message: str
) -> str:
    """
    Analyze source code and identify syntax,
    runtime and logical problems.
    """

    prompt = f"""

You are an expert code analysis assistant.

PROGRAMMING LANGUAGE:
{programming_language}

SOURCE CODE:
{code}

ERROR MESSAGE:
{error_message if error_message else "No error message provided."}

Analyze the code carefully.

Identify:

1. Syntax errors
2. Compilation errors
3. Runtime errors
4. Logical errors
5. Incorrect conditions
6. Incorrect loops
7. Variable problems
8. Type-related problems
9. Exception risks
10. Other important programming mistakes

For every important problem explain:

- Problem
- Location
- Type of error
- Why it happens
- Severity

IMPORTANT:

Do not invent errors.

If the code is correct, clearly say that
no obvious error was found.

Do not claim that you executed the code.

Return clear normal text.
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
    programming_language: str,
    code: str,
    analysis: str
) -> str:
    """
    Generate corrected code based on the
    identified programming problems.
    """

    prompt = f"""

You are an expert programming mentor.

PROGRAMMING LANGUAGE:
{programming_language}

ORIGINAL CODE:
{code}

CODE ANALYSIS:
{analysis}

Create a corrected version of the code.

Requirements:

1. Preserve the original purpose.
2. Fix the identified errors.
3. Do not unnecessarily rewrite the program.
4. Use valid syntax for the specified language.
5. Keep the code beginner-friendly.
6. Do not invent functionality that was not requested.

Return:

CORRECTED CODE:
Provide the complete corrected source code.

IMPORTANT:

Do not claim that you executed or compiled the code.

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
            "Code fixing failed: "
            + str(e)
        )


# ============================================================
# 5. CODE EXPLANATION TOOL
# ============================================================

@tool
def code_explanation(
    programming_language: str,
    original_code: str,
    analysis: str,
    corrected_code: str
) -> str:
    """
    Explain the identified bug and the changes
    made in the corrected code.
    """

    prompt = f"""

You are a programming teacher helping a beginner.

PROGRAMMING LANGUAGE:
{programming_language}

ORIGINAL CODE:
{original_code}

BUG ANALYSIS:
{analysis}

CORRECTED CODE:
{corrected_code}

Explain the solution in simple English.

Include:

1. What was wrong
2. Why the problem occurred
3. What was changed
4. Why the correction works
5. Important programming concept involved
6. How the student can avoid this mistake

Use beginner-friendly explanations.

Do not claim that the code was executed.

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
            "Code explanation failed: "
            + str(e)
        )


# ============================================================
# 6. CODE IMPROVEMENT TOOL
# ============================================================

@tool
def code_improvement(
    programming_language: str,
    corrected_code: str
) -> str:
    """
    Suggest safe improvements for the corrected
    code without changing its intended behavior.
    """

    prompt = f"""

You are a senior programming mentor.

PROGRAMMING LANGUAGE:
{programming_language}

CORRECTED CODE:
{corrected_code}

Review the corrected code.

Suggest practical improvements related to:

1. Readability
2. Code structure
3. Error handling
4. Naming
5. Performance
6. Maintainability
7. Beginner-friendly coding practices

Only suggest improvements that are relevant.

Do not unnecessarily complicate the code.

Return:

Improvement:
Reason:

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
            "Code improvement failed: "
            + str(e)
        )


# ============================================================
# 7. TOOLS AVAILABLE TO GEMMA
# ============================================================

tools = [
    code_analysis,
    code_fix,
    code_explanation,
    code_improvement
]


# ============================================================
# 8. CODEFIX AI AGENT
# ============================================================

system_prompt = """

You are CodeFix AI, an intelligent
Code Debugging Agent.

Your purpose is to help students understand,
debug and improve their programming code.

The student provides:

1. Programming Language
2. Source Code
3. Error Message
4. Optional Question


YOUR WORKFLOW:

STEP 1:

Use the Code Analysis tool to analyze the
provided source code.

Identify syntax, compilation, runtime and
logical problems.


STEP 2:

Use the Code Fix tool to generate a corrected
version of the code based on the analysis.


STEP 3:

Use the Code Explanation tool to explain:

- What went wrong
- Why it happened
- What was changed
- Why the correction works


STEP 4:

Use the Code Improvement tool to provide
useful recommendations for the corrected code.


STEP 5:

Combine all results and produce the final
CodeFix AI Debugging Report.


IMPORTANT:

YOU ARE THE AGENT CORE.

You must decide when a tool is required.

Use the tools available to you.

Follow the complete workflow.

Do not invent errors.

Do not claim that code was executed,
compiled or tested unless an actual execution
tool was used.

Do not expose internal reasoning.

Do not expose tool calls.

Do not output JSON.

Do not output Python dictionaries.


FINAL RESPONSE FORMAT:

CODEFIX AI - DEBUGGING REPORT

Programming Language:

Code Summary:

Bugs Found:

Root Cause:

Corrected Code:

What Was Changed:

Explanation:

Code Improvement Suggestions:

Prevention Tips:

Final Verdict:


The final response must be understandable
to a college student.

Use clear paragraphs and bullet points.

If no bug is found, clearly state:

"No obvious bug was found in the provided code."

The final response must contain actual information
obtained from the debugging tools.
"""


# ============================================================
# 9. CREATE LANGCHAIN AGENT
# ============================================================

agent = create_agent(

    model=llm_flash,

    tools=tools,

    system_prompt=system_prompt

)


# ============================================================
# 10. INPUT SCHEMA
# ============================================================

class AgentInput(BaseModel):

    programming_language: str = Field(
        description=(
            "Programming language of the code, "
            "for example Java, Python, C, C++ "
            "or JavaScript"
        )
    )

    code: str = Field(
        description=(
            "Source code that needs debugging"
        )
    )

    error_message: str = Field(
        default="",
        description=(
            "Compiler or runtime error message, "
            "if available"
        )
    )

    user_question: str = Field(
        default="",
        description=(
            "Optional question about the code"
        )
    )


# ============================================================
# 11. FORMAT INPUT FOR GEMMA AGENT
# ============================================================

def format_for_agent(x) -> dict:

    if isinstance(
        x,
        dict
    ):

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

        user_question = x.get(
            "user_question",
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

        user_question = getattr(
            x,
            "user_question",
            ""
        )


    user_input = f"""

CODEFIX AI INPUT

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

{error_message if error_message else "No error message provided."}


==================================================
USER QUESTION
==================================================

{user_question if user_question else "No additional question provided."}


==================================================
TASK
==================================================

Debug this code.

You are the Agent Core.

Follow the complete workflow:

Code Analysis
        ↓
Code Fix
        ↓
Code Explanation
        ↓
Code Improvement
        ↓
Final Debugging Report

Return the final CodeFix AI report as readable text.
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
# 12. EXTRACT FINAL TEXT
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


                    # Ignore Gemini reasoning
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
# 13. LANGSERVE CHAIN
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
# 14. FASTAPI APPLICATION
# ============================================================

app = FastAPI(

    title=(
        "CodeFix AI Agent API"
    ),

    version="1.0",

    description=(
        "CodeFix AI Agent using "
        "LangChain, Gemma-4-31B-IT, "
        "FastAPI and LangServe."
    )

)


# ============================================================
# 15. LANGSERVE ROUTER
# ============================================================

add_routes(

    app,

    formatted_agent_chain,

    path="/codefix-agent"

)


# ============================================================
# 16. HOME ENDPOINT
# ============================================================

@app.get("/")
def home():

    return {

        "message":
            "CodeFix AI Agent is running.",

        "agent_core":
            "gemma-4-31b-it",

        "endpoint":
            "/codefix-agent",

        "playground":
            "/codefix-agent/playground/"

    }


# ============================================================
# 17. RUN APPLICATION
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
