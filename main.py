#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main.py

Bin Packing – zestaw algorytmów z prostym CLI, wywołanie:
  python main.py <algorytm>

Dostępne algorytmy:
  full       – pełny przegląd (n≤8)
  hill_det   – Hill‐Climbing deterministyczne
  hill_rand  – Hill‐Climbing losowe
  tabu       – Tabu Search (bez dodatkowych opcji)
  sa         – Simulated Annealing (ustawienia w kodzie)
  ga         – Genetic Algorithm (parametry w kodzie)
  gp         – Genetic Programming demo
  es         – Evolutionary Strategy demo
  compare    – porównanie HC det vs SA

Domyślne dane:
  items    = [5,7,6,2,4,3,10,1,2,3,4]
  capacity = 10
"""

import argparse, random, copy, itertools, math
from collections import deque
import matplotlib.pyplot as plt

STATIC_ITEMS = [5, 7, 6, 2, 4, 3, 10, 1, 2, 3, 4]
DEFAULT_CAPACITY = 10

# 1. FUNKCJA CELU
def objective(sol, items, capacity):
    used = 0
    for b in sol:
        if not b:
            continue
        total = sum(items[i] for i in b)
        if total > capacity:
            return len(items) + 1
        used += 1
    return used

# 2. LOSOWE ROZWIĄZANIE (First‐Fit)
def random_solution(items, capacity):
    idx = list(range(len(items)))
    random.shuffle(idx)
    bins = []
    for i in idx:
        placed = False
        for b in bins:
            if sum(items[j] for j in b) + items[i] <= capacity:
                b.append(i)
                placed = True
                break
        if not placed:
            bins.append([i])
    return bins

# 3. SĄSIEDZTWO: przeniesienie jednego przedmiotu
def get_neighbors(sol, items, capacity):
    neigh = []
    for a in range(len(sol)):
        for i in sol[a]:
            # do istniejących binów
            for b in range(len(sol)):
                if b == a:
                    continue
                if sum(items[j] for j in sol[b]) + items[i] <= capacity:
                    new = copy.deepcopy(sol)
                    new[a].remove(i)
                    new[b].append(i)
                    if not new[a]:
                        new.pop(a)
                    neigh.append(new)
            # do nowego binu
            if items[i] <= capacity:
                new = copy.deepcopy(sol)
                new[a].remove(i)
                if not new[a]:
                    new.pop(a)
                new.append([i])
                neigh.append(new)
    return neigh

# 4. PEŁNY PRZEGLĄD
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
            bins.setdefault(b, []).append(i)
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

# 5. HILL‐CLIMBING deterministyczne
def hill_climbing_det(items, capacity, max_iters=1000):
    cur = random_solution(items, capacity)
    cur_obj = objective(cur, items, capacity)
    best, best_obj = cur, cur_obj
    for _ in range(max_iters):
        nbrs = get_neighbors(cur, items, capacity)
        if not nbrs:
            break
        objs = [objective(n, items, capacity) for n in nbrs]
        m = min(objs)
        if m < cur_obj:
            i = objs.index(m)
            cur, cur_obj = nbrs[i], m
            if m < best_obj:
                best, best_obj = cur, m
        else:
            break
    return best, best_obj

# 6. HILL‐CLIMBING losowe
def hill_climbing_rand(items, capacity, max_iters=1000):
    cur = random_solution(items, capacity)
    cur_obj = objective(cur, items, capacity)
    best, best_obj = cur, cur_obj
    for _ in range(max_iters):
        nbrs = get_neighbors(cur, items, capacity)
        if not nbrs:
            break
        cand = random.choice(nbrs)
        co = objective(cand, items, capacity)
        if co < cur_obj:
            cur, cur_obj = cand, co
            if co < best_obj:
                best, best_obj = cand, co
    return best, best_obj

# 7. TABU SEARCH
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

    for _ in range(max_iters):
        nbrs = get_neighbors(cur, items, capacity)
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
        if cur_obj < best_obj:
            best, best_obj = cur, cur_obj
        k = sol_key(cur)
        tabu.append(k)
        tabu_set.add(k)
        last_valid = cur

    return best, best_obj

# 8. SIMULATED ANNEALING
def simulated_annealing(items, capacity,
                        T0=10.0, alpha=0.95, stop_T=0.1,
                        max_iters=1000, neighbor_dist='uniform'):
    cur = random_solution(items, capacity)
    cur_obj = objective(cur, items, capacity)
    best, best_obj = cur, cur_obj
    T, it = T0, 0
    while T > stop_T and it < max_iters:
        nbrs = get_neighbors(cur, items, capacity)
        if not nbrs:
            break
        if neighbor_dist == 'normal':
            sigma = max(1, len(nbrs)//3)
            idx = int(abs(random.gauss(0, sigma))) % len(nbrs)
            cand = nbrs[idx]
        else:
            cand = random.choice(nbrs)
        co = objective(cand, items, capacity)
        if co < cur_obj or random.random() < math.exp((cur_obj - co) / T):
            cur, cur_obj = cand, co
            if co < best_obj:
                best, best_obj = cand, co
        T *= alpha
        it += 1
    return best, best_obj

# 9. GENETIC ALGORITHM
def encode(sol, n):
    chrom = [0]*n
    for bi, b in enumerate(sol):
        for i in b:
            chrom[i] = bi
    return chrom

def decode(chrom, items, capacity):
    paired = list(enumerate(chrom))
    paired.sort(key=lambda x: x[1])
    bins = []
    for i, _ in paired:
        placed = False
        for b in bins:
            if sum(items[j] for j in b) + items[i] <= capacity:
                b.append(i)
                placed = True
                break
        if not placed:
            bins.append([i])
    return bins

def fitness(chrom, items, capacity):
    return objective(decode(chrom, items, capacity), items, capacity)

def select_parents(pop, fits):
    def tourney():
        i, j = random.sample(range(len(pop)), 2)
        return pop[i] if fits[i] < fits[j] else pop[j]
    return tourney(), tourney()

def crossover_one(ch1, ch2):
    p = random.randint(1, len(ch1)-1)
    return ch1[:p] + ch2[p:]

def crossover_uniform(ch1, ch2):
    return [ch1[i] if random.random()<0.5 else ch2[i] for i in range(len(ch1))]

def mutate_swap(ch):
    i, j = random.sample(range(len(ch)), 2)
    ch[i], ch[j] = ch[j], ch[i]

def mutate_reassign(ch, max_bins):
    i = random.randrange(len(ch))
    ch[i] = random.randrange(max_bins)

def genetic_algorithm(items, capacity,
                      pop_size=50, gens=100,
                      crossover_method='one', mutation_method='swap',
                      termination='gen', no_improve_limit=20,
                      elite=False):
    n = len(items)
    pop = [encode(random_solution(items, capacity), n) for _ in range(pop_size)]
    fits = [fitness(c, items, capacity) for c in pop]
    best_idx, best_val = min(range(pop_size), key=lambda i: fits[i]), min(fits)
    best_ch = pop[best_idx].copy()
    no_imp = 0
    gen = 0

    while True:
        gen += 1
        new = [best_ch.copy()] if elite else []
        while len(new) < pop_size:
            p1, p2 = select_parents(pop, fits)
            ch = crossover_one(p1, p2) if crossover_method=='one' else crossover_uniform(p1, p2)
            if mutation_method=='swap' and random.random()<0.1:
                mutate_swap(ch)
            if mutation_method=='reassign' and random.random()<0.1:
                mutate_reassign(ch, pop_size)
            new.append(ch)
        pop = new
        fits = [fitness(c, items, capacity) for c in pop]
        idx, val = min(range(pop_size), key=lambda i: fits[i]), min(fits)
        if val < best_val:
            best_val = val
            best_ch = pop[idx].copy()
            no_imp = 0
        else:
            no_imp += 1
        if termination=='gen' and gen>=gens:
            break
        if termination=='noimprove' and no_imp>=no_improve_limit:
            break

    return decode(best_ch, items, capacity), best_val

# 10. GP DEMO
class GPNode:
    def __init__(self, val, left=None, right=None):
        self.val, self.left, self.right = val, left, right
    def eval(self, x):
        if not self.left:
            return x if self.val=='x' else float(self.val)
        a, b = self.left.eval(x), self.right.eval(x)
        return {'+':a+b,'-':a-b,'*':a*b,'/':a/b if abs(b)>1e-6 else 1}[self.val]
    def clone(self):
        return GPNode(self.val,
                      self.left.clone() if self.left else None,
                      self.right.clone() if self.right else None)
    def collect(self):
        nodes = [self]
        if self.left:  nodes += self.left.collect()
        if self.right: nodes += self.right.collect()
        return nodes
    def __str__(self):
        if not self.left: return str(self.val)
        return f"({self.val} {self.left} {self.right})"

def gen_gp_tree(depth, funcs, terms):
    if depth==0 or (depth>1 and random.random()<0.3):
        return GPNode(random.choice(terms))
    f = random.choice(funcs)
    return GPNode(f,
                  gen_gp_tree(depth-1, funcs, terms),
                  gen_gp_tree(depth-1, funcs, terms))

def gp_crossover(p1, p2):
    c1, c2 = p1.clone(), p2.clone()
    n1, n2 = random.choice(c1.collect()), random.choice(c2.collect())
    n1.val, n1.left, n1.right, n2.val, n2.left, n2.right = \
        n2.val, n2.left, n2.right, n1.val, n1.left, n1.right
    return c1

def gp_mutation(tree, funcs, terms, depth):
    m = tree.clone()
    target = random.choice(m.collect())
    new = gen_gp_tree(depth, funcs, terms)
    target.val, target.left, target.right = new.val, new.left, new.right
    return m

def gp_demo(gens=30, pop_size=20):
    funcs = ['+','-','*','/']; terms = ['x','1','2','3','5']
    pop = [gen_gp_tree(3, funcs, terms) for _ in range(pop_size)]
    xs = [i/10 for i in range(-10,11)]
    def fit(t): return sum((t.eval(x)-math.sin(x))**2 for x in xs)
    fits = [fit(t) for t in pop]
    best, best_err = pop[fits.index(min(fits))], min(fits)
    for _ in range(gens):
        new = [best.clone()]
        while len(new) < pop_size:
            if random.random() < 0.8:
                i, j = random.sample(range(pop_size), 2)
                child = gp_crossover(pop[i], pop[j])
            else:
                child = pop[random.randrange(pop_size)].clone()
            if random.random() < 0.1:
                child = gp_mutation(child, funcs, terms, 3)
            new.append(child)
        pop = new
        fits = [fit(t) for t in pop]
        ce = min(fits)
        if ce < best_err:
            best_err = ce
            best = pop[fits.index(ce)]
    print("GP best expr:", best)
    print("Error sin(x):", best_err)

# 11. EVOLUTIONARY STRATEGY
def evolutionary_strategy(func, dim, bounds, mu, lam, sigma, gens):
    pop = [[random.uniform(*bounds) for _ in range(dim)] for _ in range(mu)]
    sig = [sigma]*mu
    fits = [func(ind) for ind in pop]
    history = []
    for _ in range(gens):
        offs, off_s, off_f = [], [], []
        for _ in range(lam):
            i, j = random.sample(range(mu), 2)
            parent = pop[i] if fits[i]<fits[j] else pop[j]
            s = sig[i]
            child = [min(max(x+random.gauss(0,s), bounds[0]), bounds[1]) for x in parent]
            tau = 1/math.sqrt(dim)
            s_new = abs(s*math.exp(random.gauss(0,tau)))
            offs.append(child); off_s.append(s_new); off_f.append(func(child))
        idxs = sorted(range(lam), key=lambda i: off_f[i])[:mu]
        pop = [offs[i] for i in idxs]
        sig = [off_s[i] for i in idxs]
        fits = [off_f[i] for i in idxs]
        history.append(min(fits))
    best_i = fits.index(min(fits))
    return pop[best_i], fits[best_i], history

def run_es_demo():
    def sphere(x): return sum(xi*xi for xi in x)
    def rastr(x): return 10*len(x) + sum(xi*xi-10*math.cos(2*math.pi*xi) for xi in x)
    def rosen(x): return sum(100*(x[i+1]-x[i]**2)**2 + (x[i]-1)**2 for i in range(len(x)-1))
    for name, f in [('Sphere',sphere),('Rastrigin',rastr),('Rosenbrock',rosen)]:
        sol, val, _ = evolutionary_strategy(f, 2, (-5.12,5.12), 5, 20, 1.0, 50)
        print(f"ES {name}: sol={sol}, val={val:.4f}")

# 12. PORÓWNANIE HC vs SA
def compare_methods(items, capacity, runs=5, iters=200):
    hc_hist, sa_hist = [], []
    for _ in range(runs):
        _, best = hill_climbing_det(items, capacity, iters)
        hist = [best]
        cur = random_solution(items, capacity)
        best = objective(cur, items, capacity)
        for _ in range(iters):
            nbrs = get_neighbors(cur, items, capacity)
            objs = [objective(n, items, capacity) for n in nbrs] or [best]
            m = min(objs)
            if m < best:
                cur, best = nbrs[objs.index(m)], m
            hist.append(best)
        hc_hist.append(hist)

        cur = random_solution(items, capacity)
        best = objective(cur, items, capacity)
        hist2 = [best]
        T = 10
        for _ in range(iters):
            nbrs = get_neighbors(cur, items, capacity)
            cand = random.choice(nbrs) if nbrs else cur
            co = objective(cand, items, capacity)
            if co < best or random.random() < math.exp((best - co)/T):
                cur, best = cand, co
            hist2.append(best)
            T *= 0.95
        sa_hist.append(hist2)

    avg_hc = [sum(col)/runs for col in zip(*[h + [h[-1]]*(iters+1-len(h)) for h in hc_hist])]
    avg_sa = [sum(col)/runs for col in zip(*[h + [h[-1]]*(iters+1-len(h)) for h in sa_hist])]

    print("\nIter | HC_det | SA")
    print("-------------------")
    for i, (h, s) in enumerate(zip(avg_hc, avg_sa)):
        print(f"{i:4d} | {h:6.2f} | {s:4.2f}")
    print()

    plt.plot(avg_hc, label='HC det')
    plt.plot(avg_sa, label='SA')
    plt.xlabel('Iteracja')
    plt.ylabel('Średnia liczba koszy')
    plt.title('Porównanie HC det vs SA')
    plt.legend()
    plt.tight_layout()
    plt.show()

# 13. CLI
def parse_args():
    p = argparse.ArgumentParser(description="Bin Packing CLI")
    sub = p.add_subparsers(dest="alg", required=True)

    sub.add_parser("full",      help="full enumeration")
    sub.add_parser("hill_det",  help="hill climbing det")
    sub.add_parser("hill_rand", help="hill climbing rand")
    sub.add_parser("tabu",      help="tabu search")
    sub.add_parser("sa",        help="simulated annealing")
    sub.add_parser("ga",        help="genetic algorithm")
    sub.add_parser("gp",        help="genetic programming demo")
    sub.add_parser("es",        help="evolutionary strategy demo")
    sub.add_parser("compare",   help="compare HC det vs SA")

    return p.parse_args()

def main():
    args   = parse_args()
    items  = STATIC_ITEMS
    cap    = DEFAULT_CAPACITY

    def show(sol, cnt):
        bins = [[items[i] for i in b] for b in sol]
        print(f"{args.alg}: użyto {cnt} koszy, rozwiązanie = {bins}")

    if args.alg == "full":
        sol, cnt = full_enumeration(items, cap)
        show(sol, cnt)

    elif args.alg == "hill_det":
        sol, cnt = hill_climbing_det(items, cap)
        show(sol, cnt)

    elif args.alg == "hill_rand":
        sol, cnt = hill_climbing_rand(items, cap)
        show(sol, cnt)

    elif args.alg == "tabu":
        sol, cnt = tabu_search(items, cap)
        show(sol, cnt)

    elif args.alg == "sa":
        sol, cnt = simulated_annealing(items, cap)
        show(sol, cnt)

    elif args.alg == "ga":
        sol, cnt = genetic_algorithm(items, cap)
        show(sol, cnt)

    elif args.alg == "gp":
        gp_demo()

    elif args.alg == "es":
        run_es_demo()

    elif args.alg == "compare":
        compare_methods(items, cap)

    else:
        print("Nieznany alg")

if __name__ == "__main__":
    main()
