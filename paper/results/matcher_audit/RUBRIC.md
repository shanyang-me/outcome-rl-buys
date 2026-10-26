# Is this physics answer right?

Each item has a physics `problem`, the reference `gold` answer (a list; a multi-part problem lists its parts in order), the model's
final boxed `answer` (parts separated by " ; "; repeated copies of the same answer are common and harmless), and `tail`, the last
part of the model's derivation, for context (use it to see which option a letter stands for, or what a bare number refers to).

An automatic matcher already scored this answer below full marks. Decide whether the matcher was too strict. Judge agreement with the
gold, not your own solution:
- `right`: every part the gold asks for is there and gives the same physical result: equal values within about 2% (units may be
  missing or different but equivalent), algebraically equivalent expressions, equivalent notation, or a multiple-choice letter whose
  option is the gold value. Extra harmless text or repetition is fine.
- `partial`: some parts right, others wrong or missing.
- `wrong`: the result differs from the gold, or no usable final answer is given.

Output: one JSON object per line, one line per item, exactly:
{"uid": "<uid verbatim>", "verdict": "right"|"partial"|"wrong", "why": "<= 10 words"}
