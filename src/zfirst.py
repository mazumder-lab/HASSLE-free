import re

def extract_metrics(input_file):
    # Define the task order for LaTeX output
    task_order = [
        "c4_ppl", "wikitext2_ppl", "ptb_ppl", "piqa", "hellaswag", "arc_easy",
        "arc_challenge", "winogrande", "rte", "openbookqa", "boolq", "average_zero_shot"
    ]
    # Initialize a dictionary to store the extracted scores
    scores = {}
    # Read the input text file
    with open(input_file, 'r') as file:
        text = file.read()
    # Extract scalar values for c4_ppl, wikitext2_ppl, ptb_ppl
    scalar_metrics = re.findall(r"(\w+_ppl)\s*=\s*([\d.]+)", text)
    for metric, value in scalar_metrics:
        if metric not in scores:
            scores[metric] = float(value)
    # Extract task metrics from JSON-like sections
    json_metrics = re.findall(r"'(\w+)':\s*([\d.]+)", text)
    for task, value in json_metrics:
        if task not in scores:
            scores[task] = float(value)
    # Extract the average_zero_shot metric
    avg_zero_shot = re.search(r"average_zero_shot\s*=\s*([\d.]+)", text)
    if avg_zero_shot:
        scores["average_zero_shot"] = float(avg_zero_shot.group(1))
    # Prepare the LaTeX-compatible format
    latex_scores = []
    for task in task_order:
        if task in scores:
            value = scores[task]
            # Multiply by 100 for percentages starting from "piqa" onwards
            if task_order.index(task) >= task_order.index("piqa"):
                value *= 100
            latex_scores.append(f"{{{value:.2f}}}")
        else:
            latex_scores.append("--")
        if task in ["ptb_ppl"]:
            latex_scores.append('')
    # Join the scores with " & " and add the LaTeX line ending
    return "&".join(latex_scores) + " \\\\"

# Path to your input text file
input_file = "./direction_to_filename"  # Replace with the actual file path


# Generate the LaTeX-formatted string
latex_string = extract_metrics(input_file)

# Print the result
print(latex_string)