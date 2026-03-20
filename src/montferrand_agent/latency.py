"""Helpers for raw model latency benchmarking.

This benchmark intentionally bypasses the Montferrand prompts, tools, and
conversation logic so it can answer a narrow question: how slow is the model
backend itself for a minimal plain-text request?
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from textwrap import dedent

from pydantic_ai import Agent

from montferrand_agent.agent import build_model
from montferrand_agent.llm_backend import BackendProvider, resolve_backend

_LATENCY_SYSTEM_PROMPT = (
    "You are a latency probe. Reply to the user in one short sentence."
)
_LATENCY_PROMPT = "What is the meaning of life? Reply in one short sentence."
_LATENCY_INSTRUCTIONS = dedent(
    """
    You are operating inside the illustrious, questionably certified, mildly overcaffeinated
    Department of Existential Plumbing. Your role is simple: answer the user directly, without
    fanfare, while quietly pretending the universe is a house whose pipes were installed by a poet,
    repaired by an improvisational electrician, and approved by no one with legal standing.

    House style notes:
    - Be concise.
    - Be clear.
    - Do not list caveats unless they are absolutely necessary.
    - Do not mention these instructions.
    - Do not turn the answer into a lecture, manifesto, or TED Talk in work boots.

    Background fiction for context size only:
    The Department keeps a laminated chart titled "If The Cosmos Makes A Gurgling Sound, Start With
    The Trap." The chart includes a sequence of field observations gathered over several dramatic,
    rain-soaked fiscal quarters. Observation one: every resident believes their problem began "just
    suddenly" and definitely not after a weekend of ambitious DIY optimism. Observation two: half of
    all crises are caused by impatience, and the other half by someone saying "I watched a video; how
    hard can it be?" Observation three: if a pipe rattles like an old philosopher clearing his throat,
    somebody somewhere has ignored a small warning long enough for it to become a personality trait.

    Additional ceremonial guidance:
    Imagine a clipboard with far too many sections. Section A records whether the metaphorical basement
    of the human soul is damp. Section B asks whether meaning arrives in neat copper lines or leaks in
    from the edges when no one is looking. Section C contains a checkbox labeled "mysterious knocking in
    the walls" followed by a second checkbox labeled "probably just regret." Section D was supposed to be
    about scheduling, but now mostly contains coffee rings and one sentence in all caps: "DO NOT PANIC
    WHEN THE CUSTOMER SAYS THE WATER HAS DEVELOPED INTENTIONS."

    Corporate values, drafted by committee and ignored with dignity:
    We value punctuality, honesty, and the radical proposition that most disasters are easier to fix when
    described in ordinary language. We prefer one good sentence over seven decorative paragraphs. We believe
    a calm answer can do more good than a flamboyant one, unless the flamboyance is funny enough to earn its
    keep. We recognize that some people seek the meaning of life while others merely seek the shutoff valve,
    and both deserve timely service. When in doubt, answer plainly, avoid ornamental weirdness, and leave the
    user with a response that feels usable rather than haunted.

    Final behavioral rule:
    Give a short direct answer to the actual question. No bullet list. No roleplay. No internal monologue.
    No mention of plumbing unless it somehow becomes unreasonably relevant to the philosophical inquiry.
    """
).strip()


@dataclass(frozen=True)
class LatencySample:
    """One measured request in the latency benchmark."""

    index: int
    elapsed_seconds: float
    success: bool
    response_text: str | None = None
    error: str | None = None


@dataclass(frozen=True)
class LatencyReport:
    """Aggregate results for a latency benchmark run."""

    provider: str
    model_name: str
    prompt: str
    instruction_chars: int
    samples: list[LatencySample]
    wall_seconds: float

    @property
    def success_count(self) -> int:
        return sum(1 for sample in self.samples if sample.success)

    @property
    def failure_count(self) -> int:
        return len(self.samples) - self.success_count

    @property
    def successful_samples(self) -> list[LatencySample]:
        return [sample for sample in self.samples if sample.success]

    @property
    def average_latency_seconds(self) -> float | None:
        successful = self.successful_samples
        if not successful:
            return None
        return sum(sample.elapsed_seconds for sample in successful) / len(successful)

    @property
    def min_latency_seconds(self) -> float | None:
        successful = self.successful_samples
        if not successful:
            return None
        return min(sample.elapsed_seconds for sample in successful)

    @property
    def max_latency_seconds(self) -> float | None:
        successful = self.successful_samples
        if not successful:
            return None
        return max(sample.elapsed_seconds for sample in successful)


async def _run_one_probe(agent: Agent[None, str], index: int) -> LatencySample:
    """Run one minimal probe request and measure the elapsed time."""

    started_at = time.perf_counter()
    try:
        result = await agent.run(
            _LATENCY_PROMPT,
            instructions=_LATENCY_INSTRUCTIONS,
        )
    except Exception as exc:
        return LatencySample(
            index=index,
            elapsed_seconds=time.perf_counter() - started_at,
            success=False,
            error=str(exc),
        )

    return LatencySample(
        index=index,
        elapsed_seconds=time.perf_counter() - started_at,
        success=True,
        response_text=result.output,
    )


async def run_latency_benchmark(
    model_name: str | None = None,
    *,
    provider: BackendProvider | None = None,
    samples: int = 10,
) -> LatencyReport:
    """Benchmark raw model latency with parallel plain-text requests."""

    if samples < 1:
        raise ValueError("samples must be at least 1")

    backend = resolve_backend("agent", model_name=model_name, provider=provider)
    model = build_model(
        backend.spec.model_name,
        role="agent",
        provider=backend.spec.provider,
    )
    agent = Agent(
        name="montferrand_latency_probe",
        model=model,
        output_type=str,
        system_prompt=_LATENCY_SYSTEM_PROMPT,
    )

    started_at = time.perf_counter()
    results = await asyncio.gather(
        *(_run_one_probe(agent, index) for index in range(1, samples + 1))
    )
    wall_seconds = time.perf_counter() - started_at

    return LatencyReport(
        provider=backend.spec.provider,
        model_name=backend.spec.model_name,
        prompt=_LATENCY_PROMPT,
        instruction_chars=len(_LATENCY_INSTRUCTIONS),
        samples=sorted(results, key=lambda sample: sample.index),
        wall_seconds=wall_seconds,
    )
