"""kernel-lab: a fixed-contract GPU kernel optimisation lab.

The package separates three things that must never be mixed:

* task contracts (``tasks/``) - frozen mathematics, shapes and tolerances;
* the evaluator (``queue``/``evaluator``/``correctness``/``timing``) - the only
  component allowed to touch a GPU and to declare a candidate correct or fast;
* the agents (``dsh_adapter``) - they only ever submit source and read feedback.
"""
