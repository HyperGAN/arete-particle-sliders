"""Schedule the saved RAW adapter through Arete's native generation owner."""

from contextlib import ExitStack
from sys import float_info

from arete_native.native import NativeResponse

from .adaptive import CachedGate, CachedPromptSteering, render_adaptive
from .steering_core import load_steering_core


def generate(session, *, step_gains=None, steering=None, strength=1.0, max_new_tokens=128, **request):
    """Render native text with fixed gains or calibrated cached steering."""
    session._open()
    binding = session.binding
    native = getattr(binding.close, "__self__", None)
    if (session.selection != "raw" or not isinstance(native, NativeResponse)
            or native.network is not binding.network or binding.render is None):
        raise ValueError("Arete scheduling requires an open native RAW render session")
    if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= 4096:
        raise ValueError("Arete requires a token budget in [1,4096]")
    if steering is not None:
        if type(strength) not in (int, float) or not -float_info.max <= strength <= float_info.max:
            raise ValueError("Arete adaptive strength must be finite")
        if step_gains is not None:
            raise ValueError("Use fixed gains or adaptive steering for this request")
        return render_adaptive(session, native, steering=steering, strength=float(strength),
                               max_new_tokens=max_new_tokens, **request)
    if type(strength) not in (int, float) or not 0 <= strength <= 1:
        raise ValueError("Arete nominal strength must be in [0,1]")
    if (not isinstance(step_gains, (list, tuple)) or len(step_gains) != max_new_tokens
            or any(type(gain) not in (float, int) or not 0 <= gain <= float_info.max for gain in step_gains)):
        raise ValueError("Arete requires one finite nonnegative gain per possible output token")
    strengths = [strength * gain for gain in step_gains]
    if any(value > 1 for value in strengths):
        raise ValueError("Arete effective strengths must be in [0,1]")

    applied = []
    prefix = None
    step = ExitStack()

    def before(_model, _args, inputs):
        nonlocal prefix
        ids = inputs.get("input_ids")
        mask = inputs.get("attention_mask")
        if ids is None or mask is None or ids.ndim != 2 or mask.ndim != 2:
            raise RuntimeError("Arete scheduled decoding requires native IDs and a mask")
        position = len(applied)
        if position == 0:
            prefix = ids.shape[1]
        width = 1 if position else prefix
        if (position >= len(strengths) or ids.shape != (1, width)
                or mask.shape != (1, prefix + position)):
            raise RuntimeError("Arete evaluated an unexpected cached token position")
        scale = strengths[position]
        step.enter_context(native.network.scaled(scale))
        applied.append(scale)

    def after(_model, _args, _output):
        step.close()

    model = native.native_model
    with ExitStack() as handles:
        handles.callback(model.register_forward_pre_hook(before, with_kwargs=True).remove)
        handles.callback(model.register_forward_hook(after, always_call=True).remove)
        handles.callback(step.close)
        output = session.render(strength=strength, max_new_tokens=max_new_tokens, **request)
    if not applied or len(applied) != len(output.native["new_token_ids"]):
        raise RuntimeError("Arete produced an unexpected number of scheduled tokens")
    output.native["strength_schedule"] = dict(prefix_tokens=prefix, effective_strengths=applied)
    return output
