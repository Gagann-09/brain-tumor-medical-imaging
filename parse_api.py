import json
with open('openapi_schema.json', 'r', encoding='utf-16') as f:
    schema = json.load(f)
for path, methods in schema.get('paths', {}).items():
    for method, data in methods.items():
        auth = 'Yes' if data.get('security') else 'No'
        print(method.upper() + ' ' + path + ' --- Summary: ' + data.get('summary', '') + ' --- Auth: ' + auth)
