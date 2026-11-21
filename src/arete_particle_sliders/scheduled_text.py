"""Apply explicit token gains to native cached Arete decoding."""

from contextlib import ExitStack
from math import isfinite
from sys import float_info

from .backend import generate_raw


def generate(model, tokenizer, slider, messages, *, step_gains, strength=1.0,
             max_new_tokens=128, enable_thinking=False, do_sample=False):
    """Use the first gain on prompt prefill, then one gain per cached step."""
    if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= 4096:
        raise ValueError("max_new_tokens must be in [1, 4096]")
    if not isinstance(step_gains, (list, tuple)) or len(step_gains) != max_new_tokens:
        raise ValueError("Arete requires one gain per possible generated token")
    if type(strength) not in (int, float) or not -float_info.max <= strength <= float_info.max:
        raise ValueError("Arete nominal strength must be finite")
    if any(type(gain) not in (int, float) or not 0 <= gain <= float_info.max for gain in step_gains):
        raise ValueError("Arete token gains must be finite and nonnegative")
    strengths = [strength * gain for gain in step_gains]
    if any(not isfinite(value) for value in strengths):
        raise ValueError("Arete effective strengths must be finite")
    if (model.generation_config.num_beams not in (None, 1)
            or model.generation_config.num_return_sequences not in (None, 1)):
        raise ValueError("Scheduled Arete generation requires one native cached sequence")

    applied = []
    prefix_tokens = None
    scope = ExitStack()

    def before(_model, _args, arguments):
        nonlocal prefix_tokens
        tokens = arguments.get("input_ids")
        mask = arguments.get("attention_mask")
        if tokens is None or mask is None or tokens.ndim != 2 or mask.ndim != 2:
            raise RuntimeError("Arete scheduled decoding requires native token IDs and an attention mask")
        index = len(applied)
        if index == 0:
            prefix_tokens = tokens.shape[1]
        width = 1 if index else prefix_tokens
        if (index >= max_new_tokens or tokens.shape != (1, width)
                or mask.shape != (1, prefix_tokens + index)):
            raise RuntimeError("Arete native cached token position differs from the requested schedule")
        scope.enter_context(slider.scaled(strengths[index]))
        applied.append(strengths[index])

    def after(_model, _args, _output):
        scope.close()

    with ExitStack() as handles:
        handles.callback(model.register_forward_pre_hook(before, with_kwargs=True).remove)
        handles.callback(model.register_forward_hook(after, always_call=True).remove)
        handles.callback(scope.close)
        text = generate_raw(model, tokenizer, slider, messages, strength=strength,
                            enable_thinking=enable_thinking, max_new_tokens=max_new_tokens,
                            do_sample=do_sample)
    if not applied:
        raise RuntimeError("Arete did not evaluate the scheduled prompt")
    return text, {"prefix_tokens": prefix_tokens, "effective_strengths": applied}
