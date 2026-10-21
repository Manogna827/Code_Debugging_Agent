import os
import uvicorn

from fastapi import FastAPI
from langserve import add_routes

from langchain_core.runnables import RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI

from pydantic import BaseModel, Field


# ============================================================
# 1. GOOGLE API KEY
# ============================================================

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable is not set.")


# ============================================================
# 2. GEMMA MODEL
# ============================================================

llm = ChatGoogleGenerativeAI(
    model="gemma-4-31b-it",
    google_api_key=GOOGLE_API_KEY,
    temperature=0
)


# ============================================================
# 3. SAFE LLM CALL
# ============================================================

def call_llm(prompt: str) -> str:
    try:
        response = llm.invoke(prompt)
        content = response.content

        if isinstance(content, str):
            return content.strip()

        if isinstance(content, list):
            text_parts = []

            for item in content:
                if isinstance(item, str):
                    text_parts.append(item)

                elif isinstance(item, dict):
                    text = item.get("text", "")

                    if text:
                        text_parts.append(text)

            result = "\n".join(text_parts).strip()

            if result:
                return result

        return str(content)

    except Exception as e:
        return f"Gemma model call failed.\nReason: {str(e)}"


# ============================================================
# 4. CODE ANALYSIS
# ============================================================

def analyze_code(
    programming_language: str,
    code: str,
    error_message: str
) -> str:

    prompt = f"""
You are CodeFix AI, an expert programming debugger.

Analyze the following source code carefully.

PROGRAMMING LANGUAGE:
{programming_language}

SOURCE CODE:
{code}

ERROR MESSAGE:
{error_message if error_message else "No error message was provided."}

Analyze the code for:

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

For every actual problem explain:

- Problem
- Location
- Error type
- Why it happens
- Severity

IMPORTANT RULES:

- Do not invent errors.
- Carefully use the supplied error message.
- If the code is correct, clearly say:
  "No obvious bug was found in the provided code."
- Do not claim that you executed or compiled the code.
- Do not return JSON.
- Return clear normal text.

Return only the code analysis.
"""

    return call_llm(prompt)


# ============================================================
# 5. CODE FIX
# ============================================================

def fix_code(
    programming_language: str,
    code: str,
    analysis: str
) -> str:

    prompt = f"""
You are CodeFix AI, an expert programming mentor.

Create a corrected version of the following program.

PROGRAMMING LANGUAGE:
{programming_language}

ORIGINAL CODE:
{code}

CODE ANALYSIS:
{analysis}

Requirements:

1. Preserve the original purpose.
2. Fix the actual errors identified.
3. Do not unnecessarily rewrite the program.
4. Use valid syntax for the specified language.
5. Keep the code beginner-friendly.
6. Do not add unrelated functionality.

Return exactly in this format:

CORRECTED CODE:

```{programming_language.lower()}
<complete corrected source code>

IMPORTANT:

- Return the complete corrected source code.

- Do not return an incomplete snippet.

- Do not claim that you executed or compiled the code.

- Do not return JSON.
  """
  
  return call_llm(prompt)

============================================================

6. CODE EXPLANATION

============================================================

def explain_code(
programming_language: str,
original_code: str,
analysis: str,
corrected_code: str,
user_question: str
) -> str:

prompt = f"""

You are CodeFix AI, a programming teacher helping a
college student understand programming errors.

PROGRAMMING LANGUAGE:
{programming_language}

ORIGINAL CODE:
{original_code}

BUG ANALYSIS:
{analysis}

CORRECTED CODE:
{corrected_code}

STUDENT QUESTION:
{user_question if user_question else "No additional question was provided."}

Explain the debugging result in simple English.

Include:

1. What was wrong
2. Why the error occurred
3. What was changed
4. Why the correction works
5. Important programming concept involved
6. How the student can avoid this mistake

Answer the student's question directly.

IMPORTANT:

- Use beginner-friendly language.

- Do not claim that the code was executed.

- Do not return JSON.

- Return normal readable text.
  """
  
  return call_llm(prompt)

============================================================

7. CODE IMPROVEMENT

============================================================

def improve_code(
programming_language: str,
corrected_code: str
) -> str:

prompt = f"""

You are a senior programming mentor.

Review the corrected program below.

PROGRAMMING LANGUAGE:
{programming_language}

CORRECTED CODE:
{corrected_code}

Suggest practical improvements related to:

1. Readability
2. Code structure
3. Error handling
4. Variable naming
5. Performance
6. Maintainability
7. Beginner-friendly programming practices

Only suggest improvements that are actually relevant.

Do not unnecessarily complicate the program.

If there are no important improvements, say:

"No major improvement is required for this code."

Return normal readable text.

Do not return JSON.
"""

return call_llm(prompt)

============================================================

8. INPUT SCHEMA

============================================================

class AgentInput(BaseModel):

programming_language: str = Field(
    description=(
        "Programming language such as Java, Python, "
        "C, C++ or JavaScript."
    )
)

code: str = Field(
    description="Source code that needs debugging."
)

error_message: str = Field(
    default="",
    description=(
        "Compiler or runtime error message, if available."
    )
)

user_question: str = Field(
    default="",
    description="Optional question from the student."
)

============================================================

9. MAIN CODEFIX WORKFLOW

============================================================

def codefix_workflow(input_data) -> str:

if isinstance(input_data, dict):

    programming_language = input_data.get(
        "programming_language",
        ""
    )

    code = input_data.get(
        "code",
        ""
    )

    error_message = input_data.get(
        "error_message",
        ""
    )

    user_question = input_data.get(
        "user_question",
        ""
    )

else:

    programming_language = getattr(
        input_data,
        "programming_language",
        ""
    )

    code = getattr(
        input_data,
        "code",
        ""
    )

    error_message = getattr(
        input_data,
        "error_message",
        ""
    )

    user_question = getattr(
        input_data,
        "user_question",
        ""
    )

# Validate input

if not programming_language.strip():
    return "Please provide the programming language."

if not code.strip():
    return "Please provide the source code."

# ========================================================
# STEP 1 - CODE ANALYSIS
# ========================================================

analysis = analyze_code(
    programming_language,
    code,
    error_message
)

# ========================================================
# STEP 2 - CODE FIX
# ========================================================

corrected_code = fix_code(
    programming_language,
    code,
    analysis
)

# ========================================================
# STEP 3 - CODE EXPLANATION
# ========================================================

explanation = explain_code(
    programming_language,
    code,
    analysis,
    corrected_code,
    user_question
)

# ========================================================
# STEP 4 - CODE IMPROVEMENT
# ========================================================

improvement = improve_code(
    programming_language,
    corrected_code
)

# ========================================================
# STEP 5 - FINAL REPORT
# ========================================================

final_report = f"""

CODEFIX AI - DEBUGGING REPORT

Programming Language:
{programming_language}

============================================================
CODE SUMMARY

CodeFix AI analyzed the provided source code and performed
a complete debugging workflow consisting of code analysis,
code correction, explanation and improvement suggestions.

============================================================
BUGS FOUND

{analysis}

============================================================
CORRECTED CODE

{corrected_code}

============================================================
WHAT WAS CHANGED

{explanation}

============================================================
CODE IMPROVEMENT SUGGESTIONS

{improvement}

============================================================
PREVENTION TIPS

- Read compiler and runtime error messages carefully.
- Check the exact line where an error occurs.
- Validate variables before using them.
- Handle possible exceptional situations.
- Test different input values.
- Understand the programming concept behind the error.
- Avoid repeating the same coding mistake.

============================================================
FINAL VERDICT

CodeFix AI completed the debugging analysis and generated
a corrected version with an explanation and improvement
suggestions.

IMPORTANT:
The code was analyzed by AI but was NOT actually executed
or compiled by CodeFix AI.
"""

return final_report.strip()

============================================================

10. LANGSERVE CHAIN

============================================================

codefix_chain = (
RunnableLambda(codefix_workflow)
.with_types(
input_type=AgentInput,
output_type=str
)
)

============================================================

11. FASTAPI APPLICATION

============================================================

app = FastAPI(
title="CodeFix AI Agent API",
version="2.0",
description=(
"AI-powered programming code debugging assistant "
"using Gemma, LangChain, FastAPI and LangServe."
)
)

============================================================

12. LANGSERVE ROUTE

============================================================

add_routes(
app,
codefix_chain,
path="/codefix-agent"
)

============================================================

13. HOME ENDPOINT

============================================================

@app.get("/")
def home():

return {
    "message": "CodeFix AI Agent is running.",
    "agent_core": "gemma-4-31b-it",
    "workflow": (
        "Analysis -> Fix -> Explanation -> Improvement"
    ),
    "endpoint": "/codefix-agent",
    "playground": "/codefix-agent/playground/"
}

============================================================

14. HEALTH CHECK

============================================================

@app.get("/health")
def health():

return {
    "status": "ok",
    "service": "CodeFix AI",
    "model": "gemma-4-31b-it"
}

============================================================

15. START SERVER

============================================================

if name == "main":

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
