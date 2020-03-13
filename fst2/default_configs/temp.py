import yaml
import json

with open("ner.yml", "r", encoding="utf-8-sig") as f:
    configs=yaml.load(f.read(), Loader=yaml.FullLoader)


with open("configs.py", "w", encoding="utf-8-sig") as f:
    f.write(json.dumps(configs))