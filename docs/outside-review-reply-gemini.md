Measured both ways. **Half-pel is not the problem.**

First, forcing *every* block inter, because at 1080p only 0.9% of blocks choose
inter and the mode decision hides the reference quality entirely. `fast` preset,
one slice, coded file sizes in bytes:

| clip | half-pel | integer-pel | zero vector | all intra |
|---|---:|---:|---:|---:|
| Sintel 1080p x16 | 5,702,222 | 5,757,490 | 5,536,009 | **3,667,721** |
| bus CIF x24 | **1,652,007** | 1,723,821 | 2,358,846 | 1,752,169 |
| mobile CIF x24 | **1,812,417** | 1,882,729 | 2,058,829 | 2,136,737 |
| container CIF x24 | **1,098,385** | 1,099,949 | 1,099,742 | 1,512,122 |
| akiyo CIF x16 | **323,807** | 324,534 | 325,231 | 776,272 |

Half-pel beats integer-pel on *every* clip - 0.2% on the near-static ones, 4.3%
on bus. The low-pass worry behind your question (interpolating the reference
destroys grain the current frame still has, so the residual carries both) does
not show up at all.

What does show up: intra beats every temporal option by **34%** at 1080p, and
the searched vectors come out *worse than not moving* (5,702,222 against
5,536,009). That second part is mostly MV coding overhead - about 122,000
vectors at ~11 bits is close to the whole 166KB gap - so it is our known
rate-blind-search problem rather than a new effect.

Second, zeroth-order residual entropy, which removes vector cost from the
comparison entirely (bits per sample, no model, no MV bits):

| clip | spatial MED | temporal, zero vector | temporal, motion compensated |
|---|---:|---:|---:|
| Sintel 1080p | **1.062** | 2.191 | 2.184 |
| bus CIF | 5.069 | 6.658 | 5.545 |
| akiyo CIF | 3.257 | **1.353** | 1.353 |

**The spatial residual has half the entropy of the temporal one at 1080p**, and
motion compensation improves on a zero vector by 0.3% there against 17% on bus.
The search is not broken - there is nothing left to find once MED has taken the
residual to 1.06 bits per sample. The temporal residual sums two *independent*
grain fields, the reference's and the current frame's, and no vector subtracts
noise the other frame does not have.

So the mode decision rejecting inter on 99.1% of 1080p blocks is the encoder
being right, and question 2 in the brief is closed - it was never a bug.

One caveat on that entropy table: on bus it prefers spatial (5.069) while the
actual *coded sizes* prefer half-pel inter (1,652,007 against intra's
1,752,169). Zeroth-order entropy ignores context and our model has a lot of it,
so that table should not be used on its own to choose a predictor.

Follow-up questions, given that:

1. Does this change your view on rate-aware motion search? It is clearly not
   the explanation for the 1080p behaviour, but does it still look worth
   building for the CIF-scale case where motion genuinely pays 17%?

2. Given spatial prediction is already at 1.06 bits/sample on smooth 1080p
   content, is there anything that would make temporal prediction competitive
   there for *lossless* - or is the right conclusion that a lossless codec
   should simply detect this and go intra, as ours does?

3. The other four questions in the brief are still open, particularly the first
   one: where the 6.86% JPEG XL lossless gap most likely lives, given we
   already have self-correcting weighted prediction and already built and
   rejected a learned MA/MANIAC context tree.
