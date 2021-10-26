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

---

When generating mock SEDs, it is better to reuse `sps` instead of incorporate it into `mock_obs` function (the way in https://github.com/bd-j/exspect/blob/main/fitting/specphot_demo.py). In that way, `sps` is built each time calling `mock_obs()`, which is very time consuming. Actually in each run, FSPS caches the SSPs for future use (unless you delete or re-initialize`sps`), and it should be very fast to generate spectrum with different `tage` and stellar mass. 


---
For this simple $\tau$ model:

$$ \mathrm{SFR}(t_l) \propto \exp(-(t_{age}(z) - t_l) / \tau)$$
we have (assuming all stellar mass is formed in this way)
$$ \mathrm{SFR}(t_0) = \frac{M}{\tau (e^{t_0/\tau} - 1)}$$

$$ \mathrm{sSFR}(t_0) = \frac{1}{\tau (e^{t_0/\tau} - 1)}$$