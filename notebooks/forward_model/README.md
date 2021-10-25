### Toy model for forward modeling of SEDs

First of all, let's start with a very simple example, where $\vec{\theta}$ only has two parameters: stellar mass $M_{\star}$ and simple $\tau$-model SFH. The galaxy redshift is sampled from a redshift distribution.

We add some simple Gaussian noise based on a given `SNR = 20`.

We use https://github.com/bd-j/exspect/blob/main/fitting/specphot_demo.py as a reference.

---

Software requirements:

`fsps`: https://github.com/bd-j/python-fsps

`sedpy`: https://github.com/bd-j/sedpy

`prospector`: https://github.com/bd-j/prospector

`exspect`: https://github.com/bd-j/exspect