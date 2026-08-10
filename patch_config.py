import json
with open('config.json', 'r') as f:
    config = json.load(f)

for model in config['models']:
    if model['name'].startswith('gemma-4-26b'):
        for rt in model['runtimes'].values():
            rt['speculative'] = {
                "type": "draft",
                "draft_model": {
                    "repo_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
                    "filename": "mtp-gemma-4-26B-A4B-it.gguf",
                    "revision": "main"
                }
            }
    elif model['name'].startswith('gemma-4-31b'):
        for rt in model['runtimes'].values():
            rt['speculative'] = {
                "type": "draft",
                "draft_model": {
                    "repo_id": "unsloth/gemma-4-31B-it-GGUF",
                    "filename": "mtp-gemma-4-31B-it.gguf",
                    "revision": "main"
                }
            }

with open('config.json', 'w') as f:
    json.dump(config, f, indent=2)
