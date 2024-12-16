from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


def prompt_pubmedqa_fn(line, task_name: str = None):
    """
    Convert a dataset line to a Doc object for evaluation.
    We assume line["final_decision"] is one of ["yes", "no", "maybe"].
    """
    # Possible choices
    choices = ["yes", "no", "maybe"]
    gold_idx = choices.index(line["final_decision"])

    # Prepare the instruction
    abstract_text = "\n".join(line["CONTEXTS"])
    question_text = line["QUESTION"]
    query = (
        f"<<Abstract:>> {abstract_text}\n"
        "----\n"
        f"<<Question:>> {question_text}"
    )

    # Return the Doc object
    return Doc(
        task_name=task_name,
        query=query,
        choices=[f" {c}" for c in choices],
        gold_index=gold_idx,
        instruction="",
    )


# Create the task configuration
pubmedqa_task = LightevalTaskConfig(
    name="pubmedqa",
    prompt_function=prompt_pubmedqa_fn,
    suite=["community"],
    hf_repo="HF-Med/pubmedqa",
    hf_subset="",
    hf_avail_splits=["train", "validation", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    metric=[Metrics.loglikelihood_acc],
    generation_size=-1,  # multiple-choice (log-likelihood) scenario
    stop_sequence=None,
)


# Define the prompt function for medqa
def prompt_medqa_fn(line, task_name: str = None):
    """
    Convert a dataset line into a Doc object for evaluation.
    The dataset line is expected to have:
    - 'sent1' for the question
    - 'ending0', 'ending1', 'ending2', 'ending3' for the 4 possible answers
    - 'label' for the correct answer index (0-based)
    """
    choices = ["A", "B", "C", "D"]
    gold_idx = line["label"]  # 0-based index

    # Construct the question and answer choices
    question = line["sent1"]
    option_choices = {
        "A": line["ending0"],
        "B": line["ending1"],
        "C": line["ending2"],
        "D": line["ending3"]
    }

    # Format the query prompt similarly to EleutherAI approach
    # For example:
    # Question: {question}
    # A. {optionA}
    # B. {optionB}
    # C. {optionC}
    # D. {optionD}
    # Answer:
    answers_str = "\n".join([f"{k}. {v}" for k, v in option_choices.items()])
    query = f"Question: {question}\n{answers_str}\nAnswer:"

    return Doc(
        task_name=task_name,
        query=query,
        choices=[f" {o}" for o in choices],  # prepend space as per lighteval convention
        gold_index=gold_idx,
        instruction=""
    )


medqa_task = LightevalTaskConfig(
    name="medqa",
    prompt_function=prompt_medqa_fn,
    suite=["community"],
    hf_repo="HF-Med/medqa-USMLE-4-options",
    hf_subset="",
    hf_avail_splits=["train", "validation", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    metric=[Metrics.loglikelihood_acc],
    generation_size=-1,  # multiple-choice (log-likelihood)
    stop_sequence=None,
)

TASKS_TABLE = [pubmedqa_task, medqa_task]

if __name__ == "__main__":
    print([t.name for t in TASKS_TABLE])
    print(len(TASKS_TABLE))
