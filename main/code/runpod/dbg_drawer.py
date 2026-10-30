import numpy as np
from curewm.backends.libero import LiberoBackend, load_libero_demos
from curewm.perturbations import annotate_phases
demos = load_libero_demos("libero_goal", 3, task_ids=[0])
be = LiberoBackend("libero_goal", render=False)
for d in demos:
    be.reset_to(d.init_state); nom = be.rollout(d.actions); d.contacts = nom["contacts"]; ph = annotate_phases(d)
    g = d.actions[:, 6]
    print("T", len(g), "succ", nom["success"], "phase counts", np.bincount(ph, minlength=5).tolist(),
          "grip<0.5 steps", int((g < 0.5).sum()), "first close", int(np.argmax(g < 0.5)) if (g < 0.5).any() else None,
          "contacts", int(nom["contacts"].sum()))
be.close()
