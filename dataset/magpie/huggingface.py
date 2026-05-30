from datasets import load_dataset

raw_data = load_dataset("jaypasnagasai/magpie")

raw_data.push_to_hub("julee0323/magpie")

print("finished")