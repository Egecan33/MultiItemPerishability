import math


def list_blocks_single_item(d, c, s, h, expiry, pi, mu):
    T = len(d)

    def block_costs(t, e):
        q = sum(d[u] for u in range(t, e + 1))
        if q <= 0:
            return None
        if expiry[t] < e:
            return None

        prod = s[t] + c[t] * q
        hold = 0.0
        for u in range(t, e + 1):
            hold += d[u] * sum(h[t:u])
        phys = prod + hold
        red = phys - pi[t] * q - mu
        return q, phys, red

    blocks = []
    for t in range(T):
        for e in range(t, T):
            res = block_costs(t, e)
            if res is None:
                continue
            q, phys, red = res
            blocks.append((t, e, q, phys, red))

    return blocks


if __name__ == "__main__":
    d = [6, 0, 5, 0, 3, 4, 0, 5]
    c = [3.0, 3.0, 3.3, 3.3, 3.6, 3.6, 3.8, 3.8]
    s = [22.0] * 8
    h = [1.0] * 8
    expiry = [t + l for t, l in enumerate([3, 3, 2, 2, 2, 2, 2, 2])]
    pi = [0.5] * 8
    mu = 1.0

    blocks = list_blocks_single_item(d, c, s, h, expiry, pi, mu)
    blocks.sort(key=lambda x: (x[0], x[1]))

    for t, e, q, phys, red in blocks:
        print(f"block ({t+1},{e+1}): Q={q:.1f}, phys={phys:.2f}, red={red:.2f}")
