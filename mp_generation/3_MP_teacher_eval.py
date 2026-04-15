"""
Stage 3: MP Evaluation (Teacher LLM)
Evaluates student responses against MP indicators to produce binary scores.
"""
import asyncio
import json
import os
import time
from openai import AsyncOpenAI
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()

# Initialize OpenAI client
client = AsyncOpenAI(
    api_key=os.getenv("OPENAI_API_KEY", ""),
)

PROMPT = """You are Teacher GPT. Your task is to evaluate a student's responses (answer_indicate) against the reference mathematical proficiency indicators (mathematical_proficiency_indicators).

## Rules:
- Decide silently.
- Output JSON only.
- No explanations.
- No restatement.
- Binary decision only.
- Never skip any indicators

## Evaluation Rules
1. For each indicator:
   - If the student's response is **"I don't know"**, assign 0.
   - If the response does not match or is irrelevant to the indicator, assign 0.
   - If the response matches the indicator's intent and shows correct reasoning/application, assign 1.
2. Output strictly in JSON format, with indicator keys mapped to 0 or 1.
3. Ensure that every indicator is carefully evaluated without skipping or overlooking any of them.

## Input
Problem (in Korean):
{problem_text}

mathematical_proficiency_indicators:
{mathematical_proficiency_indicators JSON}

answer_indicate:
{answer_indicate JSON}

## Output
Provide the evaluation result in the following JSON format:
{
    "CU1": 0 or 1, "SC1": 0 or 1, "PF1": 0 or 1, "AR1": 0 or 1, ...
}

## Input Example
Problem: For the rational function $y=\\dfrac{2x-3}{2x+5}$, how many points on its graph have both $x$- and $y$-coordinates as integers?
Options: [{"index":1,"text":"$1$"}, {"index":2,"text":"$2$"}, {"index":3,"text":"$3$"}, {"index":4,"text":"$4$"}, {"index":5,"text":"$5$"}]

mathematical_proficiency_indicators:
[
    { "CU1": "Interpret what the problem is asking, and recognize that it is about finding points on the graph of the rational function y = (2x - 3) / (2x + 5) whose x- and y-values are both integers (integer lattice points)." },
    { "SC1": "Choose a strategy to rewrite the equation in a form that makes the integer condition more explicit, such as expressing it in terms of y - 1." },
    { "PF1": "Transform y = (2x - 3)/(2x + 5) into y - 1 = [(2x - 3) - (2x + 5)]/(2x + 5) = -8/(2x + 5)." },
    { "AR1": "Use the fact that 2x + 5 is odd to restrict the candidates to the odd divisors of 8." },
    ...
]

answer_indicate:[
    { "CU1": "Although not written explicitly, by rewriting y = (2x - 3)/(2x + 5) as y = -8/(2x + 5) + 1 and listing possible values of 2x + 5 to find integer pairs (x, y), it appears the student recognized the task as identifying integer lattice points." },
    { "SC1": "They rewrote y = (2x - 3)/(2x + 5) as y = -8/(2x + 5) + 1." },
    { "PF1": "Although intermediate algebra steps were omitted, the final expression y - 1 = -8/(2x + 5) (equivalently y = -8/(2x + 5) + 1) was obtained." },
    { "AR1": "I don't know" },
    ...
]

Output:
{
    "CU1": 1,
    "SC1": 1,
    "PF1": 1,
    "AR1": 0,
    ...
}

-----------------------------------------------------------------------------------
"""


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
            mathematical_proficiency_indicators = item.get("mathematical_proficiency_indicators", "")
            if isinstance(mathematical_proficiency_indicators, dict):
                indicator_items = list(mathematical_proficiency_indicators.items())
            else:
                indicator_items = []

            indicator_text = "[\n"
            for k, v in indicator_items:
                indicator_text += f'    {{ "{k}": "{v}" }},\n'
            indicator_text = indicator_text.rstrip(",\n") + "\n]"

            # Answer indicator
            answer_indicator = item.get("answer_indicate", "")
            if isinstance(answer_indicator, dict):
                answer_indicator_items = list(answer_indicator.items())
            else:
                answer_indicator_items = []

            answer_indicator_text = "[\n"
            for k, v in answer_indicator_items:
                answer_indicator_text += f'    {{ "{k}": "{v}" }},\n'
            answer_indicator_text = answer_indicator_text.rstrip(",\n") + "\n]"

            # Problem
            problem = item.get("problem_text", "")
            problem_options = item.get("problem_option", "")
            problem_option_string = f"Options: {problem_options}" if problem_options != "" else ""

            # Prompt
            user_message = f"{PROMPT}\n\nProblem (in Korean): {problem}\n{problem_option_string} \n\n Mathematical Proficiency Indicators: \n{indicator_text}\n\n Answer Indicate {answer_indicator_text}"

            messages = [
                {"role": "user", "content": user_message}
            ]

            response = await client.chat.completions.create(
                model=model_name,
                messages=messages
            )

            result_text = response.choices[0].message.content

            try:
                result_text = result_text.replace("\\,", ",")
                result_text = result_text.replace("\\", "\\\\")
                result_text = result_text.replace("\\\\\\\\,", "\\\\")
                result_dict = json.loads(result_text)
                return result_dict
            except json.JSONDecodeError:
                print("JSON DECODING ERROR")

        except Exception as e:
            print(f"Error processing item: {e}")
            return None


async def process_item(idx: int, item: Dict[str, Any], model_name: str) -> Dict[str, Any]:
    if idx % 100 == 0:
        print("idx:", idx)

    result = await call_gpt_api(item, model_name)

    item_copy = item.copy()
    item_copy["mathematical_proficiency_eval"] = result

    return item_copy


async def process_batch_concurrent(batch: List[Dict[str, Any]], concurrent_limit: int, model_name: str) -> List[Dict[str, Any]]:
    """Process items in batches"""
    semaphore = asyncio.Semaphore(concurrent_limit)

    async def process_with_semaphore(idx: int, item: Dict[str, Any]) -> Dict[str, Any]:
        async with semaphore:
            return await process_item(idx, item, model_name)

    tasks = [process_with_semaphore(idx, item) for idx, item in enumerate(batch)]
    results = await asyncio.gather(*tasks)

    return results


async def main():
    # File paths - modify these for your setup
    encoding = "utf-8"
    json_file_path = "data/2_student_answer.json"  # Input: data with student responses
    output_json_file_path = "data/3_MP_evaluated.json"  # Output: data with MP evaluations

    # Model name
    model_name = "gpt-4o"

    # Batch settings
    BATCH_SIZE = 50
    concurrent_limit = 20

    # Read input data
    data = read_json_file(json_file_path, encoding)
    print(f"Total records: {len(data)}")

    # Load existing results if available
    if os.path.exists(output_json_file_path):
        processed_data = read_json_file(output_json_file_path, encoding)
        print(f"Loaded existing results: {len(processed_data)} records")
    else:
        processed_data = []
        for item in data:
            item_copy = item.copy()
            item_copy["mathematical_proficiency_eval"] = None
            processed_data.append(item_copy)
        save_json_file(processed_data, output_json_file_path, encoding)
        print(f"Initialized output file with {len(processed_data)} records")

    # Find pending items
    pending_indices = [i for i, item in enumerate(processed_data) if item.get("mathematical_proficiency_eval") is None]
    print(f"Pending items: {len(pending_indices)}")

    if len(pending_indices) == 0:
        print("No pending items to process!")
        return

    # Process in batches
    total_batches = (len(pending_indices) + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_num in range(total_batches):
        start_idx = batch_num * BATCH_SIZE
        end_idx = min(start_idx + BATCH_SIZE, len(pending_indices))
        batch_indices = pending_indices[start_idx:end_idx]

        print(f"\nProcessing batch {batch_num + 1}/{total_batches} (items {start_idx + 1}-{end_idx})")

        batch_items = [processed_data[i] for i in batch_indices]

        batch_results = await process_batch_concurrent(
            batch=batch_items,
            concurrent_limit=concurrent_limit,
            model_name=model_name
        )

        for i, result in zip(batch_indices, batch_results):
            processed_data[i] = result

        save_json_file(processed_data, output_json_file_path, encoding)

        success_count = sum(1 for item in processed_data if item.get("mathematical_proficiency_eval") is not None)
        print(f"Saved! Progress: {success_count}/{len(processed_data)} completed")

    print(f"\nAll batches completed! Saved to {output_json_file_path}")

    success_count = sum(1 for item in processed_data if item.get("mathematical_proficiency_eval") is not None)
    fail_count = len(processed_data) - success_count
    print(f"Final Statistics: {success_count} succeeded, {fail_count} failed")


if __name__ == "__main__":
    asyncio.run(main())
