import re

def handler(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r'[^a-z0-9一-鿿]+', '-', text)
    text = re.sub(r'-+', '-', text).strip('-')
    return text
