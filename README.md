# Resource usage indication example: S2

On an ordinary desktop computer (Intel Core i5-4590S, 8GB RAM), state
generation is very much feasible (<6h / state) at Lambda = 16,
i.e. with matrices of size 612 and an operator space PAP consisting of
the spherical harmonics up to l = 33.

Distance calculation with the 'SCS' algorithm uses significant amounts
of memory, however: a single thread in dimension 312 (Lambda = 12)
uses around 7 GB, which limits feasibility of higher-precision graph
generation on typical desktop systems.
