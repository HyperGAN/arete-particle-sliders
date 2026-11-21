from contextlib import contextmanager

from arete_particle_sliders.backend import generate_raw, make_fixture
from arete_particle_sliders.scheduled_text import generate


def test_native_cached_positions_and_constant_generation_parity():
    model = make_fixture(16)
    model.generation_config.eos_token_id = None

    class Tokenizer:
        def apply_chat_template(self, messages, **options):
            return [1, 3, 4]

        def decode(self, ids, **options):
            return tuple(int(token) for token in ids)

    class Slider:
        strength = 0.0

        @contextmanager
        def scaled(self, value):
            previous = self.strength
            self.strength = value
            try:
                yield
            finally:
                self.strength = previous

    tokenizer, slider = Tokenizer(), Slider()
    messages = [{"role": "user", "content": "hello"}]
    request = {"strength": 0.5, "max_new_tokens": 3}
    base = generate_raw(model, tokenizer, slider, messages, **request)
    uniform, receipt = generate(model, tokenizer, slider, messages, **request, step_gains=[1.0] * 3)
    assert uniform == base and receipt == {"prefix_tokens": 3, "effective_strengths": [0.5] * 3}

    observed = []
    probe = model.model.language_model.register_forward_pre_hook(lambda *_: observed.append(slider.strength))
    try:
        _, receipt = generate(model, tokenizer, slider, messages, **request, step_gains=[1.5, 1, 0.5])
    finally:
        probe.remove()
    assert receipt == {"prefix_tokens": 3, "effective_strengths": [0.75, 0.5, 0.25]}
    assert observed == [0.75, 0.5, 0.25]
    assert slider.strength == 0 and not model._forward_hooks and not model._forward_pre_hooks

    model.generation_config.eos_token_id = base[0]
    stopped, receipt = generate(model, tokenizer, slider, messages, **request, step_gains=[1.0] * 3)
    assert stopped == base[:1]
    assert receipt["effective_strengths"] == [0.5]
