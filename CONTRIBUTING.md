# Contributing

Contributions are welcome — bug reports, sharper predicates, better
ranking ideas, and tests.

## Development setup

```bash
git clone https://github.com/habib-analyst/slicesniff.git
cd slicesniff
python -m pip install -e ".[dev]"
python -m pytest
```

## Ground rules

- **NumPy is the only runtime dependency.** That is a product feature
  (prediction-only, framework-free); please do not add others.
- **Every behaviour change needs a test.** The planted-slice test in
  `tests/test_slicesniff.py` is the contract: a deliberately planted
  confidently-wrong slice must be found and ranked first. If you change
  the search or the ranking, that test and the example must still pass.
- **Honesty in numbers.** Examples print what they measure. If a change
  makes the demo numbers worse, the README and report say so — they are
  never tuned quietly.
- **Readable slices only.** Any new predicate type must render as plain
  text a practitioner can check by hand (`feature == 'x'`,
  `0.4 <= feature <= 0.8`). No opaque slice definitions.

## Reporting issues

Open a GitHub issue with a minimal reproduction: the shapes of your
`probs` / `y_true` / metadata, the SliceSniff version, and what you
expected the top card to be.
