#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main.py

Bin Packing – zestaw algorytmów z prostym CLI, wywołanie:
  python main.py <algorytm> [opcje]

Dostępne algorytmy:
  full       – pełny przegląd (n≤8)
  hill_det   – Hill‐Climbing deterministyczne
  hill_rand  – Hill‐Climbing losowe
  tabu       – Tabu Search
  sa         – Simulated Annealing
  ga         – Genetic Algorithm
  gp         – Genetic Programming demo
  es         – Evolutionary Strategy demo
  compare    – porównanie HC det vs SA

Parametry wspólne:
  --items     lista wag (domyślnie stałe)
  --capacity  pojemność kosza (domyślnie 10)

Przykłady:
  python main.py full --items 5 7 6 2 4 3 --capacity 10
  python main.py hill_det --iters 500 --items 5 7 6 2 4 3
  python main.py hill_rand --iters 500
  python main.py tabu --tabu-size 5 --iters 200 --allow-return
  python main.py sa --iters 500 --neighbor-dist normal
  python main.py ga --pop 30 --gens 100 --crossover uniform \
                    --mutation reassign --termination noimprove \
                    --no_improve 20 --elite
  python main.py gp --gens 20 --pop 10
  python main.py es
  python main.py compare --iters 200 --runs 5
"""

import argparse, random, copy, itertools, math
from collections import deque
import matplotlib.pyplot as plt

STATIC_ITEMS = [5, 7, 6, 2, 4, 3, 10, 1, 2, 3, 4]
DEFAULT_CAPACITY = 10


# 1. FUNKCJA CELU
def objective(sol, items, capacity):
    """ liczba użytych koszy + kara za przepełnienie (proporcjonalna) """
    used, overflow = 0, 0
    for b in sol:
        if not b:                 # pomijamy puste
            continue
        total = sum(items[i] for i in b)
        if total > capacity:      # zbierz nadmiar
            overflow += total - capacity
        used += 1
    return used + overflow        # im mniej, tym lepiej


# 2. LOSOWE ROZWIĄZANIE (First‐Fit)
#def random_solution(items, capacity):
#    idx = list(range(len(items)))
#    random.shuffle(idx)
#    bins = []
#    for i in idx:
#        for b in bins:
#            if sum(items[j] for j in b) + items[i] <= capacity:
#                b.append(i)
#                break
#        else:
#            bins.append([i])
#    return bins

#def random_solution(items, capacity=None):          # capacity ignorowane
#    idx = list(range(len(items)))
#    random.shuffle(idx)
#    return [[i] for i in idx]                       # 1 przedmiot → 1 kosz
def random_solution(items, capacity):
    n = len(items)
    assign = [random.randint(0, n - 1) for _ in range(n)] # generujemy losowe rozkład
    bins_dict = {}
    for i, b in enumerate(assign):
        if b not in bins_dict:
            bins_dict[b] = []
        bins_dict[b].append(i)
    return [bins_dict[k] for k in sorted(bins_dict)]

# 3. SĄSIEDZTWO: przeniesienie jednego przedmiotu 7
def get_neighbors(sol, items, capacity=None):
    neighbors = []

    for bin_idx in range(len(sol)):

        for item in sol[bin_idx]:
            # Move to existing bins
            for bin_dx_double in range(len(sol)):
                if bin_dx_double == bin_idx:
                    continue
                new = [bin.copy() for bin in sol]
                new[bin_idx].remove(item)
                new[bin_dx_double].append(item)
                new = [b for b in new if b]
                neighbors.append(new)

            new = [bin.copy() for bin in sol]
            new[bin_idx].remove(item)
            new = [b for b in new if b]
            new.append([item])
            neighbors.append(new)

    return neighbors


# 4. PEŁNY PRZEGLĄD 6
def full_enumeration(items, capacity, max_n=8):
    if len(items) > max_n:
        return None, None
    n = len(items)
    best_count = n + 1
    best_sol = None
    for assign in itertools.product(range(n), repeat=n):
        bins = {}
        valid = True
        for i, b in enumerate(assign):
            bins.setdefault(b, []).append(i) # dodaje i albo tworzy pustą liste z i
        for b in bins.values():
            if sum(items[i] for i in b) > capacity:
                valid = False
                break
        if not valid:
            continue

        used = len(bins)
        if used < best_count:
            best_count = used
            best_sol = [bins[k] for k in sorted(bins)]
    return best_sol, best_count


# 5. HILL‐CLIMBING deterministyczne 7
def hill_climbing_det(items, capacity, max_iters=1000):
    cur = random_solution(items, capacity)
    cur_obj = objective(cur, items, capacity)
    best, best_obj = cur, cur_obj
    history = [best_obj]

    for _ in range(max_iters):
        nbrs = get_neighbors(cur, items, capacity)
        print("hilldet",len(nbrs))
        if not nbrs:
            break

        objs = [objective(n, items, capacity) for n in nbrs]
        m = min(objs)
        history.append(m)

        if m < cur_obj:
            i = objs.index(m)
            cur, cur_obj = nbrs[i], m
            if m < best_obj:
                best, best_obj = cur, m
        else:
            break
    history += [history[-1]] * (max_iters - len(history))

    return best, best_obj,history


# 6. HILL‐CLIMBING losowe 7
def hill_climbing_rand(items, capacity, max_iters=1000):
    cur = random_solution(items, capacity)
    cur_obj = objective(cur, items, capacity)
    best, best_obj = cur, cur_obj
    history = [best_obj]

    for _ in range(max_iters):
        nbrs = get_neighbors(cur, items, capacity)
        print("hillrand",len(nbrs))

        if not nbrs:
            break
        cand = random.choice(nbrs)
        co = objective(cand, items, capacity)
        history.append(co)

        if co < cur_obj:
            cur, cur_obj = cand, co
            if co < best_obj:
                best, best_obj = cand, co

    return best, best_obj,history


# 7. TABU SEARCH 1
def tabu_search(items, capacity,
                tabu_size=10,
                max_iters=1000,
                allow_return=False):
    def sol_key(sol):
        sb = sorted([sorted(b) for b in sol])
        return "|".join(",".join(map(str, b)) for b in sb)

    cur = random_solution(items, capacity)
    cur_obj = objective(cur, items, capacity)
    best, best_obj = cur, cur_obj

    tabu = deque([sol_key(cur)], maxlen=tabu_size)
    tabu_set = set(tabu)
    last_valid = cur
    history = [cur_obj]

    for _ in range(max_iters):
        nbrs = get_neighbors(cur, items, capacity)
        print("tabu",len(nbrs))

        next_sol = None
        for n in nbrs:
            k = sol_key(n)
            if k not in tabu_set:
                next_sol = n
                break
        if next_sol is None:
            if allow_return:
                cur = last_valid
                cur_obj = objective(cur, items, capacity)
                continue
            else:
                break
        cur = next_sol
        cur_obj = objective(cur, items, capacity)
        history.append(cur_obj)

        if cur_obj < best_obj:
            best, best_obj = cur, cur_obj
        k = sol_key(cur)
        tabu.append(k)
        tabu_set.add(k)
        last_valid = cur

    return best, best_obj,history


# 8. SIMULATED ANNEALING
def simulated_annealing(items, capacity,
                        T0=10.0, alpha=0.95, stop_T=0.1,
                        max_iters=1000):
    cur = random_solution(items, capacity)
    cur_obj = objective(cur, items, capacity)
    best, best_obj = cur, cur_obj
    T, it = T0, 0
    history = [best_obj]

    while T > stop_T and it < max_iters:
        nbrs = get_neighbors(cur, items, capacity)
        print("sa",len(nbrs))

        if not nbrs:
            break
        cand = random.choice(nbrs)
        co = objective(cand, items, capacity)
        history.append(co)

        if co < cur_obj or random.random() < math.exp((cur_obj - co) / T):
            cur, cur_obj = cand, co
            if co < best_obj:
                best, best_obj = cand, co
        T *= alpha
        it += 1

    history += [history[-1]] * (max_iters - len(history))
    return best, best_obj,history


# 9. GENETIC ALGORITHM
def encode(sol, n):
    chrom = [0] * n #[0,3,4,0,2,4,1]
    for bi, b in enumerate(sol): # przechodzimy przez biny i ich numery
        for i in b: # przechodzimy przez itemy w danym binie
            chrom[i] = bi # na miejscu danego itema w chromosomie przypisujemy jego numer bina
    return chrom

def decode(chrom, items=None, capacity=None):
    bins_dict = {}
    for i, bin_id in enumerate(chrom):
        if bin_id not in bins_dict:
            bins_dict[bin_id] = []
        bins_dict[bin_id].append(i)
    return [bins_dict[k] for k in sorted(bins_dict)]

def fitness(chrom, items, capacity):
    return objective(decode(chrom, items, capacity), items, capacity)


def tournament_selection(pop, fits, k=2):
    best_index = random.choice(range(len(pop)))
    for _ in range(k):
        challenger_index = random.choice(range(len(pop)))
        if fits[challenger_index] < fits[best_index]:
            best_index = challenger_index
    return pop[best_index]

def select_parents(pop, fits):
    return tournament_selection(pop, fits), tournament_selection(pop, fits)


def crossover_one(ch1, ch2):
    p = random.randint(1, len(ch1) - 1)
    return ch1[:p] + ch2[p:]

def crossover_uniform(ch1, ch2):
    return [ch1[i] if random.random() < 0.5 else ch2[i] for i in range(len(ch1))]

def mutate_swap(ch):
    i, j = random.sample(range(len(ch)), 2)
    ch[i], ch[j] = ch[j], ch[i]

def mutate_reassign(ch, max_bins):
    i = random.randrange(len(ch))
    ch[i] = random.randrange(max_bins)


def genetic_algorithm(items, capacity,pop_size=50, gens=100,
                      crossover_method='one', mutation_method='swap',
                      termination='gen', no_improve_limit=20,
                      elite=False):
    n = len(items)
    pop = [encode(random_solution(items, capacity), n) for _ in range(pop_size)]
    fits = [fitness(c, items, capacity) for c in pop]

    best_idx, best_val = min(enumerate(fits), key=lambda i: i[1])

    best_ch = pop[best_idx].copy()
    no_imp, gen = 0, 0
    history = [best_val]

    while True: # kończy się gdy no impr limit ( 20 )  limit ilosć iteracji
        gen += 1
        new = [best_ch.copy()] if elite else []
        while len(new) < pop_size:
            p1, p2 = select_parents(pop, fits)
            ch = crossover_one(p1, p2) if crossover_method == 'one' else crossover_uniform(p1, p2)

            if random.random() < 0.1:
                if mutation_method == 'swap':
                    mutate_swap(ch)
                else:
                    mutate_reassign(ch, pop_size)
            new.append(ch)

        pop = new
        fits = [fitness(c, items, capacity) for c in pop]

        idx, val = min(enumerate(fits), key=lambda i: i[1])

        if val < best_val:
            best_val, best_ch, no_imp = val, pop[idx].copy(), 0
        else:
            no_imp += 1

        if (termination == 'gen' and gen >= gens) or (termination == 'noimprove' and no_imp >= no_improve_limit):
            break
        history.append(val)

    history += [history[-1]] * (gens - len(history))
    return decode(best_ch, items, capacity), best_val,history


# 11. EVOLUTIONARY STRATEGY
def evolutionary_strategy(func, dim, bounds, mu, lam, sigma, gens):
    pop = [[random.uniform(*bounds) for _ in range(dim)] for _ in range(mu)]
    sig = [sigma] * mu
    fits = [func(ind) for ind in pop]
    history = []
    for _ in range(gens):
        offs, off_s, off_f = [], [], []
        for _ in range(lam):
            i, j = random.sample(range(mu), 2)
            parent = pop[i] if fits[i] < fits[j] else pop[j]
            s = sig[i]
            child = [min(max(x + random.gauss(0, s), bounds[0]), bounds[1]) for x in parent]
            tau = 1 / math.sqrt(dim)
            s_new = abs(s * math.exp(random.gauss(0, tau)))
            offs.append(child);
            off_s.append(s_new);
            off_f.append(func(child))
        idxs = sorted(range(lam), key=lambda i: off_f[i])[:mu]
        pop = [offs[i] for i in idxs]
        sig = [off_s[i] for i in idxs]
        fits = [off_f[i] for i in idxs]
        history.append(min(fits))
    best_i = fits.index(min(fits))
    return pop[best_i], fits[best_i], history

# 12. PORÓWNANIE – WSZYSTKIE ALGORYTMY
def compare_methods_2(seed=42, runs=5, iters=200):
    """
    Porównuje HC_det, HC_rand, Tabu, SA, GA.
    Tworzy jedną figurę zawierającą:
    - wykres słupkowy średnich wyników
    - time series każdego algorytmu w każdym runie
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import random

    def gen_instance(rng):
        n = rng.randint(12, 24)
        items = [rng.randint(1, 10) for _ in range(n)]

        low = max(items)  # musi zmieścić największy
        high = int(sum(items) / n * 1.8)  # „luźny” kosz

        if high < low:  # zabezpieczenie
            high = low  # range = [low, low]

        capacity = rng.randint(low, high)
        return items, capacity

    random.seed(seed)
    rng = random.Random(seed)

    algs = {
        "HC_det": lambda i, c: hill_climbing_det(i, c, max_iters=iters)[2],
        "HC_rand": lambda i, c: hill_climbing_rand(i, c, max_iters=iters)[2],
        "Tabu": lambda i, c: tabu_search(i, c, tabu_size=10, max_iters=iters)[2],
        "SA": lambda i, c: simulated_annealing(i, c, max_iters=iters)[2],
        "GA": lambda i, c: genetic_algorithm(i, c, pop_size=50, gens=iters)[2],
    }

    final_scores = {name: [] for name in algs}
    histories_per_run = []

    w = 8  # szerokość (cale)
    h = 2.6 * (runs + 1)  # wysokość: 1,6 cal/wykres
    fig = plt.figure(figsize=(w, h), layout="constrained")
    spec = gridspec.GridSpec(nrows=runs + 1, ncols=1, figure=fig)

    # Run-y i ich przebiegi
    for r in range(runs):
        items, capacity = gen_instance(rng)
        print(f"Run {r + 1}: capacity={capacity}, items={items}")

        ax_ts = fig.add_subplot(spec[r + 1, 0])
        ax_ts.set_title(f"Run {r + 1} – Time Series (#bins per iter)")

        for name, solve in algs.items():
            print(name)

            hist = solve(items, capacity)
            print(hist)
            if len(hist) < iters:
                hist += [hist[-1]] * (iters - len(hist))
            final_scores[name].append(hist[-1])
            ax_ts.plot(hist, label=name)

        ax_ts.set_ylabel("Liczba koszy")
        ax_ts.set_xlabel("Iteracja")
        ax_ts.legend(fontsize="x-small", loc="upper right")

    # Średnie końcowe wyniki
    avgs = {name: sum(vals) / len(vals) for name, vals in final_scores.items()}

    print(f"\nŚrednia liczba koszy (seed={seed}, {runs} instancji):")
    print("-------------------------------------")
    for name, val in avgs.items():
        print(f"{name:8s}: {val:6.2f}")
    print()

    # Wykres słupkowy u góry
    ax_bar = fig.add_subplot(spec[0, 0])
    ax_bar.bar(avgs.keys(), avgs.values())
    ax_bar.set_ylabel("Średnia liczba koszy")
    ax_bar.set_title(f"Średnie wyniki końcowe ({runs} instancji)")

    plt.tight_layout()
    plt.show()

# 13. CLI
def parse_args():
    p = argparse.ArgumentParser(description="Bin Packing CLI")
    sub = p.add_subparsers(dest="alg", required=True)

    base = argparse.ArgumentParser(add_help=False)
    base.add_argument("-c","--capacity", type=int, default=10)
    base.add_argument("-i","--items", type=int, nargs="+")

    def add(name, **kw):
        return sub.add_parser(name, parents=[base], **kw)

    f = add("full", help="full enumeration")
    f.add_argument("--max-n", type=int, default=8)

    hd = add("hill_det", help="hill climbing det")
    hd.add_argument("-t","--iters", type=int, default=1000)

    hr = add("hill_rand", help="hill climbing rand")
    hr.add_argument("-t","--iters", type=int, default=1000)

    ts = add("tabu", help="tabu search")
    ts.add_argument("--tabu-size", type=int, default=10)
    ts.add_argument("-t","--iters", type=int, default=1000)
    ts.add_argument("--allow-return", action="store_true")

    sa = add("sa", help="simulated annealing")
    sa.add_argument("--T0", type=float, default=10.0)
    sa.add_argument("--alpha", type=float, default=0.95)
    sa.add_argument("--stop-T", type=float, default=0.1)
    sa.add_argument("-t","--iters", type=int, default=1000)
    sa.add_argument("--neighbor-dist", choices=["uniform","normal"], default="uniform")

    ga = add("ga", help="genetic algorithm")
    ga.add_argument("--pop", type=int, default=50)
    ga.add_argument("--gens", type=int, default=100)
    ga.add_argument("--crossover", choices=["one","uniform"], default="one")
    ga.add_argument("--mutation", choices=["swap","reassign"], default="swap")
    ga.add_argument("--termination", choices=["gen","noimprove"], default="gen")
    ga.add_argument("--no_improve", type=int, default=20)
    ga.add_argument("--elite", action="store_true")

    gp = add("gp", help="genetic programming demo")
    gp.add_argument("--gens", type=int, default=30)
    gp.add_argument("--pop", type=int, default=20)

    add("es", help="evolutionary strategy demo")

    cmp = add("compare", help="compare HC vs SA")
    cmp.add_argument("--runs", type=int, default=5)
    cmp.add_argument("-t","--iters", type=int, default=200)

    cmp2 = add("compare2", help="compare all algorithms on random problems")
    cmp2.add_argument("--runs", type=int, default=5)
    cmp2.add_argument("-t", "--iters", type=int, default=200)
    cmp2.add_argument("--seed", type=int, default=42)

    return p.parse_args()

def main():
    args = parse_args()
    items = args.items if args.items else STATIC_ITEMS
    cap   = args.capacity

    def show(sol, cnt):
        # przetłumacz indeksy na wagi
        bins = [[items[i] for i in b] for b in sol]
        print(f"{args.alg} bins: {cnt}, solution: {bins}")

    if args.alg == "full":
        sol, cnt = full_enumeration(items, cap, args.max_n)
        show(sol, cnt)

    elif args.alg == "hill_det":
        sol, cnt = hill_climbing_det(items, cap, args.iters)
        show(sol, cnt)

    elif args.alg == "hill_rand":
        sol, cnt = hill_climbing_rand(items, cap, args.iters)
        show(sol, cnt)

    elif args.alg == "tabu":
        sol, cnt = tabu_search(items, cap,
                               tabu_size=args.tabu_size,
                               max_iters=args.iters,
                               allow_return=args.allow_return)
        show(sol, cnt)

    elif args.alg == "sa":
        sol, cnt = simulated_annealing(items, cap,
                                       T0=args.T0, alpha=args.alpha,
                                       stop_T=args.stop_T,
                                       max_iters=args.iters,
                                       neighbor_dist=args.neighbor_dist)
        show(sol, cnt)

    elif args.alg == "ga":
        sol, cnt = genetic_algorithm(items, cap,
                   pop_size=args.pop,
                   gens=args.gens,
                   crossover_method=args.crossover,
                   mutation_method=args.mutation,
                   termination=args.termination,
                   no_improve_limit=args.no_improve,
                   elite=args.elite)

        show(sol, cnt)

    elif args.alg == "es":
        run_es_demo()

    elif args.alg == "compare":
        compare_methods(items, cap, runs=args.runs, iters=args.iters)

    elif args.alg == "compare2":
        compare_methods_2(seed=args.seed, runs=args.runs, iters=args.iters)

    else:
        print("Nieznany alg")

if __name__=="__main__":
    main()

