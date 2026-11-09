"""Full matched native contexts; the tokenizer never truncates a prompt."""
import json
from pathlib import Path
from particle_sliders.pg09_audio import AudioRow
from .host_source import encode_chat


def encode_prompt(tokenizer,messages,*,max_tokens=4096,enable_thinking=False):
    if not isinstance(messages,list)or not messages:raise ValueError('nonempty native conversation required')
    if any(not isinstance(row,dict)or set(row)-{'role','content'}or row.get('role')not in('system','user','assistant')
        or not isinstance(row.get('content'),str)for row in messages):raise ValueError('text-only native messages required')
    ids=encode_chat(tokenizer,messages,enable_thinking=enable_thinking)
    if not isinstance(ids,list)or not 1<=len(ids)<=max_tokens or any(type(v)is not int or v<0 for v in ids):
        raise ValueError('complete native token context exceeds limit; no truncation')
    return ids


def read_rows(path,tokenizer,*,max_tokens=4096,enable_thinking=False):
    rows=[]
    for line in Path(path).read_text().splitlines():
        if not line.strip():continue
        row=json.loads(line)
        if set(row)=={'neutral_messages','positive_messages'}:
            neutral=encode_prompt(tokenizer,row['neutral_messages'],max_tokens=max_tokens,enable_thinking=enable_thinking)
            positive=encode_prompt(tokenizer,row['positive_messages'],max_tokens=max_tokens,enable_thinking=enable_thinking)
        elif set(row)=={'neutral_ids','positive_ids'}:neutral,positive=row['neutral_ids'],row['positive_ids']
        else:raise ValueError('exact matched neutral/positive full token IDs or messages required')
        for ids in(neutral,positive):
            if not isinstance(ids,list)or not 1<=len(ids)<=max_tokens or any(type(v)is not int or v<0 for v in ids):
                raise ValueError('complete nonempty native token IDs required; no truncation')
        if neutral==positive:raise ValueError('distinct teacher/student native contexts required')
        rows.append(AudioRow(tuple(neutral),tuple(positive)))
    if len(rows)<2:raise ValueError('at least two actual matched rows required')
    return rows
