"""
Stage 1: MP Indicator Generation (Teacher LLM)
Generates problem-specific Mathematical Proficiency indicators from problem text and unit name.
"""
import asyncio
import json
import os
from openai import AsyncOpenAI
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()

# Initialize OpenAI client
client = AsyncOpenAI(
    api_key=os.getenv("OPENAI_API_KEY", ""),
)

PROMPT = """You are Teacher GPT.
Your task is to analyze a given math Problem and its Unit name, and then generate a step-by-step set of indicators that describe the problem-solving process a student may follow.

You must generate indicators that specify **what kinds of student actions provide evidence of mathematical proficiency**, not a worked solution or a recipe for solving the problem.

Guidelines:
- Use strand assignment rules:
    - CU: which means Conceptual Understanding, comprehension of mathematical concepts, operations, and relations
    - PF: which means Procedural Fluency, skill in carrying out procedures flexibly, accurately, efficiently, and appropriately
    - SC: which means Strategic Competence, ability to formulate, represent, and solve mathematical problems
    - AR: which means Adaptive Reasoning, capacity for logical thought, reflection, explanation, and justification

- Indicator construction rules:
    - Write indicators as **observable actions in student work** (e.g., define, represent, compute, justify, explain, simplify).
    - Indicators must be written at the **action-type level**, not as specific algebraic instantiations.
    - Do NOT introduce concrete substitutions, parameter values, numeric coefficients, symbolic expansions, or canonical tricks (e.g., do NOT write "let a = 3t", "compute 9t²", etc.).
    - Instead, describe the **kind of transformation or representation** being performed (e.g., "express variables using a common parameter", "rewrite the expression in a simpler form", "simplify by canceling common factors").
    - Do NOT include the final answer.
    - Do NOT include a fully specified transformation that trivially solves the problem.
    - Indicators should remain valid across **multiple reasonable solution strategies**; do not assume a single canonical method.

- Structure and ordering:
    - Each indicator must be prefixed with its strand code and an index (e.g., "CU1", "SC2", "PF3", "AR1").
    - The indicators should follow a plausible problem-solving flow: 1) initial understanding, 2) problem representation / planning 3) execution 4) reasoning / justification.
    - Strands may appear multiple times and may be interleaved.
    - If a single moment in problem solving provides evidence for multiple strands, split it into multiple indicators with different strand codes.

- Output format must be a JSON dictionary:
{
    "mathematical_proficiency_indicators": {
        "CU1": "...",
        "SC1": "...",
        "CU2": "...",
        ...
    }
}

---

### One-shot Example

**Input**
Problem: "Solve the differential equation \\( \\frac{dy}{dx} = 2x \\) with initial condition \\( y(0)=1 \\)."
Unit: Differential Equations

**Output**
{
    "mathematical_proficiency_indicators": {
        "CU1": "Determine the type and order of this equation",
        "SC1": "Rewrite the equation in an easier way",
        "CU2": "Write the mathematical idea you need to solve this equation",
        "SC2": "Sort the necessary data and ignore the redundant ones",
        "PF1": "Predict a solution",
        "CU3": "Show the steps for solving the equation using a table, a figure and a diagram",
        "PF2": "Summarize the steps in the solution",
        "PF3": "Write a suitable algorithm to solve this equation",
        "SC3": "Identify any special numerical cases used by this equation to generalize the solution",
        "AR1": "Describe your solution in general",
        "AR2": "Based on your knowledge of differential equations, interpret your solution",
        "AR3": "According to your solution, draw the conclusions"
    }
}
-----------------------------------------------------------------------------------
"""

MATHEMATICAL_PROF_INDI = {}


def read_json_file(file_path: str, encoding: str = "utf-8") -> List[Dict[str, Any]]:
    with open(file_path, 'r', encoding=encoding) as file:
        return json.load(file)


def save_json_file(data: List[Dict[str, Any]], file_path: str, encoding: str = "utf-8"):
    with open(file_path, 'w', encoding=encoding) as file:
        json.dump(data, file, ensure_ascii=False, indent=4)


async def call_gpt_api(item: Dict[str, Any], model_name: str) -> Optional[Dict[str, int]]:
    MAX_TRIAL = 5
    for trial in range(MAX_TRIAL):
        try:
            problem = item.get("problem_text", "")
            curriculum_theme_title = item.get("curriculum_theme_title", "")

            problem_options = item.get("problem_option", "")
            problem_option_string = f"Options: {problem_options}" if problem_options != "" else ""

            user_message = f"{PROMPT}\n\nProblem (in Korean): {problem}\n{problem_option_string} \nUnit (in Korean): {curriculum_theme_title}"

            messages = [
                {"role": "user", "content": user_message}
            ]

            response = await client.chat.completions.create(
                model=model_name,
                messages=messages
            )

            result_text = response.choices[0].message.content

            try:
                result_dict = json.loads(result_text)
                return result_dict["mathematical_proficiency_indicators"]
            except json.JSONDecodeError:
                print("JSON DECODING ERROR")

        except Exception as e:
            print(f"Error processing item: {e}")
            return None


async def process_item(idx: int, item: Dict[str, Any], model_name: str) -> Dict[str, Any]:
    if idx % 50 == 0:
        print("idx:", idx)

    problem_id = item.get("problem_id", "")

    if problem_id in MATHEMATICAL_PROF_INDI.keys():
        result = MATHEMATICAL_PROF_INDI[problem_id]
    else:
        result = await call_gpt_api(item, model_name)
        MATHEMATICAL_PROF_INDI[problem_id] = result

    item_copy = item.copy()
    item_copy["mathematical_proficiency_indicators"] = result

    return item_copy


async def process_all_items_concurrent(data: List[Dict[str, Any]], concurrent_limit: int, model_name: str) -> List[Dict[str, Any]]:
    semaphore = asyncio.Semaphore(concurrent_limit)

    async def process_with_semaphore(idx: int, item: Dict[str, Any]) -> Dict[str, Any]:
        async with semaphore:
            return await process_item(idx, item, model_name)

    tasks = [process_with_semaphore(idx, data[idx]) for idx in range(len(data))]
    results = await asyncio.gather(*tasks)

    return results


async def main():
    # File paths - modify these for your setup
    encoding = "utf-8"
    json_file_path = "data/input_data.json"  # Input: raw interaction data
    output_json_file_path = "data/1_indicator.json"  # Output: data with MP indicators

    # Model name (e.g., "gpt-4o", "gpt-4-turbo")
    model_name = "gpt-4o"

    # Read input data
    data = read_json_file(json_file_path, encoding)
    print(f"Total records: {len(data)}")

    # Filter items that need processing
    failed_items = [item for item in data if item.get("mathematical_proficiency_indicators") is None]
    success_items = [item for item in data if item.get("mathematical_proficiency_indicators") is not None]

    print(f"Already succeeded: {len(success_items)}")
    print(f"Items to process: {len(failed_items)}")

    if len(failed_items) == 0:
        print("No items to process!")
        return

    # Process items
    retried_data = await process_all_items_concurrent(
        data=failed_items,
        concurrent_limit=50,
        model_name=model_name
    )

    # Merge results
    result_map = {item["session_id"]: item for item in success_items}
    for item in retried_data:
        result_map[item["session_id"]] = item

    processed_data = [result_map[item["session_id"]] for item in data]

    # Save results
    save_json_file(processed_data, output_json_file_path, encoding)
    print(f"Completed! Saved {len(processed_data)} processed items to {output_json_file_path}")

    success_count = sum(1 for item in processed_data if item.get("mathematical_proficiency_indicators") is not None)
    fail_count = len(processed_data) - success_count
    print(f"Statistics: {success_count} succeeded, {fail_count} failed")


if __name__ == "__main__":
    asyncio.run(main())
