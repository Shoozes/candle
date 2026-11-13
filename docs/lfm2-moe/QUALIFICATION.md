# Held Q4 production qualification

This protocol is inert. It authorizes no execution, installation, activation,
package assembly, publication or consumer repin. A fresh exact owner grant and
quiet/resource admission are required before its one finite attempt.

`tests/fixtures/lfm2_moe_tiny/qualification_inputs.json` freezes two authored
prompts and eight teacher-forced positions each before any production run.
Its token IDs come from the locked official tokenizer. Fixed logit probe IDs
allow finite difference reporting alongside exact next-token selections.
No generated captions or loading-only result can satisfy acceptance.

The first packet covers Candle parity only, in this order: pinned llama.cpp CPU,
Candle CPU, then Candle RTX 4090. Each lane uses both prompts, batch/chunk 128,
context 12,288 and the same existing Q4_K_M artifact. There are six finite child
loads and 48 teacher-forced observations total, eight per child. Each child has
a 180-second ceiling and a 32 GiB Windows Job cap; the packet has a 900-second
wall-clock ceiling. CPU uses two threads. CUDA graphs stay disabled in the
reference launcher. No timeout or application default is changed.

Before every child, require no foreign large-model or build workload, at least
40 GiB available host physical memory and commit headroom, and (for CUDA) at
least 18 GiB dedicated GPU free. These are initial conservative admission bounds;
they can refuse this packet and cannot be relaxed within its attempt. Measure
actual peaks and post-exit memory, not just PID absence. The existing bounded
Windows launcher creates children suspended, assigns the kill-on-close Job,
then resumes them; retain terminal and cleanup evidence for each child.

Stop the entire packet on admission failure, nonzero exit, timeout, cleanup
failure, nonfinite/incorrect logit shape or any teacher-forced top-1 mismatch.
Do not retry, change prompts/positions, relax limits or add probes in that
attempt. `tools/lfm2_moe/compare_probes.py` requires `--inputs` pointing to the
frozen qualification manifest and `--case` naming one exact case. It checks the
locked reference revision, exact prompt/probe IDs, every frozen position and
the complete frozen continuation reported by both engines before comparing
selections and recording maximum absolute/RMS logit differences. It scans
combined logs as bytes and decodes JSON lines as strict UTF-8 so clipped native
diagnostic text cannot hide a numerical failure. A top-1 mismatch reports the
first divergent position and both token IDs. The comparator rejects shortened
runs and wrong inputs. New packets include all four comparison commands
explicitly and retain binary/input identities and process receipts. Record any
numerical disagreement as a qualification failure for investigation.

Before review, bind the inert packet to the current dirty-source inventory and
SHA-256 of both binaries. The inventory includes the MoE overlay and local
CPU/CUDA build sources, manifests, lockfile and toolchain configuration.
It includes the example package's build script, its download module and the
CUDA sources compiled by that script.
It hashes working-tree bytes and records deleted source paths with a null
hash. The task-local reference probe is built from locked
llama.cpp `582eab71` and the continuation-reporting probe source, CPU-only,
without modifying `C:/llamacpp`. Packet `00ab6c5e` predates the reference
continuation fields. Its archived position-2 mismatch is replayed with the
comparator at `42deff05`. The current comparator rejects that older record.
The installed bundle at `9df0b73c` has a separate runtime identity.
[SOURCES](SOURCES.md) records the existing model's size, hash and path.

Edge's six model-free profile/template/output tests and CLI/API compilation are
software proof only. After successful Candle parity, select the next bounded
Edge qualification step explicitly: admit the exact candidate through its held
profile gate, then prove CLI/API plain chat, strict final JSON, cancellation,
unload/reload and a successful recovery request on CPU and CUDA. Preserve current
90-second API deadlines and all existing permission/tool gates. Source publication
must precede normal reviewed Edge pin/lock adoption. This first packet does not
grant that separate consumer acceptance, publication or package work.

Full native-checkpoint/Q8 production, Metal and 131,072-token context acceptance
remain outside the selected delivery. Native and other implemented storage
types retain their deterministic fixture scope.

---
AI-edited: 2026-11-13 | agent=Codex/root | task=lfm2-moe-prefill-review | change=document example package build inputs in the packet inventory
