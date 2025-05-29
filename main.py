import itertools
from graphviz import Digraph

# === AST Node Definitions ===
class Symbol:
    def __init__(self, value):
        self.value = value

class Epsilon:
    pass

class Star:
    def __init__(self, child):
        self.child = child

class Concat:
    def __init__(self, left, right):
        self.left = left
        self.right = right

class Union:
    def __init__(self, left, right):
        self.left = left
        self.right = right

# === Parser ===
class Parser:
    def __init__(self, expr):
        self.tokens = list(expr)
        self.pos = 0

    def parse(self):
        return self._parse_union()

    def _parse_union(self):
        left = self._parse_concat()
        while self._peek() == 'U':
            self._next()
            right = self._parse_concat()
            left = Union(left, right)
        return left

    def _parse_concat(self):
        nodes = []
        while self._peek() not in (None, ')', 'U'):
            nodes.append(self._parse_star())
        if not nodes:
            return Epsilon()
        node = nodes[0]
        for nxt in nodes[1:]:
            node = Concat(node, nxt)
        return node

    def _parse_star(self):
        node = self._parse_atom()
        if self._peek() == '*':
            self._next()
            node = Star(node)
        return node

    def _parse_atom(self):
        ch = self._peek()
        if ch == '(':
            self._next()
            node = self._parse_union()
            if self._peek() != ')':
                raise SyntaxError("Missing closing parenthesis")
            self._next()
            return node
        if ch == 'ε':
            self._next()
            return Epsilon()
        if ch in ('0', '1'):
            self._next()
            return Symbol(ch)
        raise SyntaxError(f"Unexpected character: {ch}")

    def _peek(self):
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def _next(self):
        ch = self._peek()
        self.pos += 1
        return ch

# === Thompson Construction ===
class NFA:
    def __init__(self, start, accept, transitions):
        self.start = start
        self.accept = accept
        self.transitions = transitions

state_counter = itertools.count()

def new_state():
    return f"s{next(state_counter)}"

def thompson(node):
    if isinstance(node, Symbol):
        s, f = new_state(), new_state()
        return NFA(s, f, {s: [(node.value, f)], f: []})
    if isinstance(node, Epsilon):
        s, f = new_state(), new_state()
        return NFA(s, f, {s: [(None, f)], f: []})
    if isinstance(node, Concat):
        n1 = thompson(node.left)
        n2 = thompson(node.right)
        n1.transitions.setdefault(n1.accept, []).append((None, n2.start))
        merged = {**n1.transitions, **n2.transitions}
        return NFA(n1.start, n2.accept, merged)
    if isinstance(node, Union):
        n1 = thompson(node.left)
        n2 = thompson(node.right)
        s, f = new_state(), new_state()
        trans = {s: [(None, n1.start), (None, n2.start)], f: []}
        for st, edges in itertools.chain(n1.transitions.items(), n2.transitions.items()):
            trans.setdefault(st, []).extend(edges)
        trans.setdefault(n1.accept, []).append((None, f))
        trans.setdefault(n2.accept, []).append((None, f))
        return NFA(s, f, trans)
    if isinstance(node, Star):
        n = thompson(node.child)
        s, f = new_state(), new_state()
        trans = {s: [(None, n.start), (None, f)], f: []}
        for st, edges in n.transitions.items():
            trans.setdefault(st, []).extend(edges)
        trans.setdefault(n.accept, []).extend([(None, n.start), (None, f)])
        return NFA(s, f, trans)
    raise ValueError("Unknown AST node")

# === Subset Construction (NFA -> DFA) ===
class DFA:
    def __init__(self, start, accepts, transitions, state_names):
        self.start = start
        self.accepts = accepts
        self.transitions = transitions
        self.state_names = state_names

def epsilon_closure(states, transitions):
    closure = set(states)
    stack = list(states)
    while stack:
        st = stack.pop()
        for sym, nxt in transitions.get(st, []):
            if sym is None and nxt not in closure:
                closure.add(nxt)
                stack.append(nxt)
    return closure

def move(states, symbol, transitions):
    return {nxt for st in states for sym, nxt in transitions.get(st, []) if sym == symbol}

def subset_construction(nfa):
    start = frozenset(epsilon_closure({nfa.start}, nfa.transitions))
    
    # Use BFS for more predictable state ordering
    d_states = set()
    ordered_states = []
    queue = [start]
    d_states.add(start)
    d_trans = {}
    
    alpha = sorted({sym for edges in nfa.transitions.values() for sym, _ in edges if sym is not None})
    
    while queue:
        T = queue.pop(0)  # BFS queue
        ordered_states.append(T)
        d_trans[T] = {}
        
        for sym in sorted(alpha):  # Process symbols in sorted order
            U = frozenset(epsilon_closure(move(T, sym, nfa.transitions), nfa.transitions))
            d_trans[T][sym] = U
            
            if U not in d_states:
                d_states.add(U)
                queue.append(U)
    
    # Add trap state if needed
    trap = frozenset()
    if trap not in d_states:
        d_states.add(trap)
        ordered_states.append(trap)
    
    # Ensure all transitions are defined
    for T in list(d_states):
        for sym in alpha:
            d_trans.setdefault(T, {})
            d_trans[T].setdefault(sym, trap)
    
    accepts = {T for T in d_states if nfa.accept in T}
    
    # Create names dictionary with more predictable ordering
    names = {}
    # Ensure start state gets q0
    names[start] = "q0"
    # Assign remaining states q1, q2, etc. based on BFS order
    counter = 1
    for state in ordered_states:
        if state != start:
            names[state] = f"q{counter}"
            counter += 1
            
    return DFA(start, accepts, d_trans, names)

# === Hopcroft Minimization ===
def minimize_dfa(dfa):
    accept_states = set(dfa.accepts)
    non_accept = set(dfa.transitions.keys()) - accept_states
    
    # Initialize partition with accepting and non-accepting states
    P = []
    if accept_states:
        P.append(frozenset(accept_states))
    if non_accept:
        P.append(frozenset(non_accept))
    W = P.copy()
    
    alphabet = []
    if dfa.transitions:
        for trans in dfa.transitions.values():
            if trans:  # Make sure the transitions dict is not empty
                alphabet = list(sorted(trans.keys()))
                break
    
    # Hopcroft's algorithm
    while W:
        A = W.pop()
        for c in alphabet:
            X = frozenset(q for q, trans in dfa.transitions.items() 
                         if c in trans and trans[c] in A)
            newP = []
            for Y in P:
                inter = Y & X
                diff = Y - X
                if inter and diff:
                    newP.extend([inter, diff])
                    if Y in W:
                        W.remove(Y)
                        W.extend([inter, diff])
                    else:
                        W.append(inter)
                else:
                    newP.append(Y)
            P = newP
    
    # Map states to their equivalence classes
    state_map = {q: block for block in P for q in block}
    new_states = set(P)
    new_start = state_map[dfa.start]
    new_accepts = {block for block in new_states if block & accept_states}
    
    # Create transitions for the minimized DFA
    new_trans = {}
    for block in new_states:
        rep = next(iter(block))
        if rep in dfa.transitions:
            new_trans[block] = {c: state_map[dfa.transitions[rep][c]] 
                              for c in alphabet if c in dfa.transitions[rep]}
    
    # Create a BFS ordering of states starting from the initial state
    ordered_states = []
    visited = set()
    queue = [new_start]
    visited.add(new_start)
    
    while queue:
        state = queue.pop(0)
        ordered_states.append(state)
        
        # Process outgoing transitions in alphabetical order
        if state in new_trans:
            for symbol in sorted(new_trans[state].keys()):
                next_state = new_trans[state][symbol]
                if next_state not in visited:
                    visited.add(next_state)
                    queue.append(next_state)
    
    # Add any remaining states that weren't reached in BFS
    for state in new_states:
        if state not in visited:
            ordered_states.append(state)
            visited.add(state)
    
    # Create new names with q0 as start state, following BFS order
    new_names = {}
    for i, state in enumerate(ordered_states):
        new_names[state] = f"q{i}"
            
    return DFA(new_start, new_accepts, new_trans, new_names)

# === Prune Unreachable States ===
def prune_unreachable(dfa):
    """
    Remove states with no incoming transitions (unreachable) before visualization.
    Also reorders states to ensure logical naming based on BFS traversal from start state.
    """
    # Find reachable states using BFS for better ordering
    reachable = []
    visited = set()
    queue = [dfa.start]
    visited.add(dfa.start)
    
    while queue:
        state = queue.pop(0)  # BFS using queue
        reachable.append(state)
        
        # Explore transitions in sorted order by symbols for more predictable naming
        sorted_transitions = sorted(dfa.transitions.get(state, {}).items())
        for _, next_state in sorted_transitions:
            if next_state not in visited:
                visited.add(next_state)
                queue.append(next_state)
                
    # Filter transitions and accepts
    pruned_transitions = {s: {c:t for c,t in dfa.transitions[s].items() if t in visited} 
                         for s in reachable if s in dfa.transitions}
    pruned_accepts = {s for s in dfa.accepts if s in visited}
    
    # Create new state names with q0 as start state, following BFS order
    new_names = {}
    for i, state in enumerate(reachable):
        new_names[state] = f"q{i}"
    
    # Update DFA with pruned information and new names
    dfa.transitions = pruned_transitions
    dfa.accepts = pruned_accepts
    dfa.state_names = new_names
    
    return dfa

# === Visualization ===
def visualize_nfa(nfa, filename='nfa'):
    # Renumber states: initial is q0, then BFS order q1..qn
    reachable = []
    visited = {nfa.start}
    queue = [nfa.start]
    while queue:
        st = queue.pop(0)
        reachable.append(st)
        for sym, nxt in nfa.transitions.get(st, []):
            if nxt not in visited:
                visited.add(nxt)
                queue.append(nxt)
    mapping = {old: f"q{i}" for i, old in enumerate(reachable)}

    dot = Digraph(comment='NFA')
    dot.attr(rankdir='LR')
    # invisible initial arrow to q0
    dot.node('start', shape='none', label='')
    dot.edge('start', mapping[nfa.start], label='')

    # draw states and transitions
    for old in reachable:
        name = mapping[old]
        shape = 'doublecircle' if old == nfa.accept else 'circle'
        dot.node(name, shape=shape)
        for sym, nxt in nfa.transitions.get(old, []):
            label = 'ε' if sym is None else sym
            dot.edge(name, mapping[nxt], label=label)

    dot.render(filename, format='png', cleanup=True)



def visualize_dfa(dfa, filename='dfa'):
    # prune unreachable states before visualization and ensure logical state naming
    dfa = prune_unreachable(dfa)
    dot = Digraph(comment='DFA')
    dot.attr(rankdir='LR')
    dot.node('start', shape='none', label='')
    dot.edge('start', dfa.state_names[dfa.start], label='')
    
    # First add all nodes to ensure consistent node order in visualization
    for s in sorted(dfa.transitions.keys(), key=lambda x: dfa.state_names[x]):
        dot.node(dfa.state_names[s], 
                shape='doublecircle' if s in dfa.accepts else 'circle', 
                label=dfa.state_names[s])
    
    # Then add all edges
    for s in sorted(dfa.transitions.keys(), key=lambda x: dfa.state_names[x]):
        # Sort transitions by symbol for consistent visualization
        for c, t in sorted(dfa.transitions[s].items()):
            dot.edge(dfa.state_names[s], dfa.state_names[t], label=c)
            
    dot.render(filename, format='png', cleanup=True)

# === Main ===
if __name__ == '__main__':
    expr = input("Enter regular expression: ")
    ast = Parser(expr).parse()
    nfa = thompson(ast)
    dfa = subset_construction(nfa)
    dfa = minimize_dfa(dfa)
    visualize_nfa(nfa, 'output_nfa')
    visualize_dfa(dfa, 'output_dfa')
    print("Generated output_nfa.png and output_dfa.png")