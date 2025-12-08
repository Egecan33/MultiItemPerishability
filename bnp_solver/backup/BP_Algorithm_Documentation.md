# Branch-and-Price for Perishable Lot-Sizing
## Algorithm Documentation and Mathematical Formulation
### Dantzig-Wolfe Decomposition with LEFO Consumption Policy

---

## 1. Problem Overview

This implementation solves the **Capacitated Lot-Sizing Problem (CLSP)** with perishable items featuring heterogeneous shelf lives under the **Last-Expired-First-Out (LEFO)** consumption policy. The algorithm employs **Dantzig-Wolfe decomposition**, generating only **Zero-Inventory-Ordering (ZIO)** production plans via dynamic programming.

**Key Features:**
- Best-First Search strategy (lowest LP bound first)
- Arc branching (z variables) and Setup branching (y variables)
- RMP linking constraints with dual extraction (τ, σ)
- Column inheritance between parent and child nodes
- Duplicate column detection via signatures

---

## 2. Mathematical Notation

### 2.1 Index Sets and Parameters

**Sets:**
```
I = set of items (indexed by i)
T = {0, 1, ..., T−1} = planning horizon periods (indexed by t, u)
Ωᵢ = set of feasible production plans (columns) for item i
```

**Parameters:**
```
dᵢₜ = demand for item i in period t           →  item.demand_quantity_by_period[t]
sᵢₜ = setup cost for item i in period t       →  item.setup_cost_by_period[t]
cᵢₜ = unit production cost for item i at t    →  item.production_unit_cost_by_period[t]
hᵢₜ = unit holding cost for item i at t       →  item.holding_unit_cost_by_period[t]
Cₜ = production capacity in period t          →  capacity[t]
mᵢₜ = shelf life for item i produced at t     →  items[i]['shelf_seq'][t]
```

**Shelf-Life Structures (Code Mapping):**
```
vᵢₜ = t + mᵢₜ = expiry period (inclusive)     →  item.perishability_horizon_by_start_period[t]
Γᵢₜ = {t, t+1, ..., min(T−1, vᵢₜ)}           →  valid ends from get_valid_ends(t)
```

The set Γᵢₜ defines which demand periods can be satisfied by production in period t. Under LEFO, items expiring first are consumed first, which the ZIO structure enforces implicitly through contiguous arc coverage.

**Code: ProductionItem dataclass**
```python
@dataclass
class ProductionItem:
    """
    Represents a single item in the lot-sizing problem.
    
    Attributes:
        item_id: Unique identifier i
        number_of_periods: T (planning horizon length)
        demand_quantity_by_period: dᵢₜ for t ∈ {0,...,T-1}
        production_unit_cost_by_period: cᵢₜ
        setup_cost_by_period: sᵢₜ
        holding_unit_cost_by_period: hᵢₜ
        perishability_horizon_by_start_period: vᵢₜ = t + mᵢₜ (absolute expiry, inclusive)
        lost_sales_penalty_per_unit: penalty for unmet demand
    """
    item_id: int
    number_of_periods: int
    demand_quantity_by_period: List[float]
    production_unit_cost_by_period: List[float]
    setup_cost_by_period: List[float]
    holding_unit_cost_by_period: List[float]
    perishability_horizon_by_start_period: List[int]  # vᵢₜ = t + mᵢₜ
    lost_sales_penalty_per_unit: float
```

### 2.2 Decision Variables

**Original (Compact) Formulation Variables:**
```
yᵢₜ ∈ {0,1} = 1 if setup occurs for item i in period t    →  setup_by_period[t]
xᵢₜ ≥ 0 = production quantity for item i in period t      →  capacity_usage_by_period[t]
zᵢₜᵤ ∈ {0,1} = 1 if demand at u satisfied from prod at t  →  arc_usage[(t, u)]
```

**Master Problem Variables:**
```
λᵢₖ ≥ 0 = weight of column k for item i                   →  lambdas[(item_id, idx)]
```

---

## 3. Dantzig-Wolfe Decomposition Structure

The problem decomposes by item: the **capacity constraints** couple all items (complicating constraints), while demand satisfaction and shelf-life constraints are item-specific (block-diagonal structure).

### 3.1 Column Definition

Each column k ∈ Ωᵢ for item i represents a complete ZIO production plan encoded as:

```
Column k for item i:
  cₖ = Σₜ[sᵢₜ·yₖₜ + Σᵤ(cᵢₜ + H(t,u))·dᵢᵤ·zₖₜᵤ]    (total plan cost)
  aₖₜ = Σᵤ dᵢᵤ·zₖₜᵤ                                (capacity usage at t)
  yₖₜ ∈ {0,1}                                       (setup indicator)
  zₖₜᵤ ∈ {0,1}                                      (arc indicator)
```

Where H(t,u) = Σₛ₌ₜ^{u-1} hᵢₛ is the cumulative holding cost from production period t to consumption period u.

**Code: ProductionPlanColumn dataclass**
```python
@dataclass
class ProductionPlanColumn:
    """
    Represents a single ZIO column k ∈ Ωᵢ for item i.
    
    Attributes:
        item_id: Which item this column belongs to
        total_plan_cost: cₖ = Σₜ[sᵢₜ·yₖₜ + Σᵤ(cᵢₜ + H(t,u))·dᵢᵤ·zₖₜᵤ]
        capacity_usage_by_period: aₖₜ = Σᵤ dᵢᵤ·zₖₜᵤ (production quantity at t)
        setup_by_period: yₖₜ ∈ {0,1} (setup indicator)
        arc_usage: zₖₜᵤ ∈ {0,1} (demand u satisfied from production t)
    """
    item_id: int
    total_plan_cost: float
    capacity_usage_by_period: List[float]  # aₖₜ
    setup_by_period: List[float]  # yₖₜ
    arc_usage: Dict[Tuple[int, int], float]  # zₖₜᵤ

    def get_signature(self) -> str:
        """Unique signature for duplicate detection."""
        arcs = sorted(self.arc_usage.keys())
        setups = [t for t, y in enumerate(self.setup_by_period) if y > 0.5]
        return f"I{self.item_id}_Y{setups}_Z{arcs}"

    def violates_branching(
        self,
        theta_0: Set[Tuple[int, int]],
        theta_1: Set[Tuple[int, int]],
        upsilon_0: Set[int],
        upsilon_1: Set[int],
    ) -> bool:
        """Check if column violates any branching constraints."""
        # Check forbidden arcs (Θ⁰)
        for (t, u) in theta_0:
            if self.arc_usage.get((t, u), 0.0) > 0.5:
                return True
        
        # Check forced arcs (Θ¹) - must have them
        for (t, u) in theta_1:
            if self.arc_usage.get((t, u), 0.0) < 0.5:
                return True
        
        # Check forbidden setups (Υ⁰)
        for t in upsilon_0:
            if self.setup_by_period[t] > 0.5:
                return True
        
        # Check forced setups (Υ¹) - must have them
        for t in upsilon_1:
            if self.setup_by_period[t] < 0.5:
                return True
        
        return False
```

**Holding Cost Precomputation:**
```python
# Hpref[k] = Σⱼ₌₀^{k-1} hⱼ
Hpref = [0.0] * (T + 1)
for k in range(1, T + 1):
    Hpref[k] = Hpref[k - 1] + holding_cost[k - 1]

# H(t,u) = Hpref[u] - Hpref[t]
def block_holding_cost(t: int, s: int) -> float:
    """Cumulative holding cost from t to s."""
    if s <= t:
        return 0.0
    return (DHpref[s + 1] - DHpref[t + 1]) - Hpref[t] * (Dpref[s + 1] - Dpref[t + 1])
```

### 3.2 Restricted Master Problem (RMP)

The RMP selects a convex combination of columns:

```
minimize   Σᵢ Σₖ cₖ · λᵢₖ

subject to:
  Σᵢ Σₖ aₖₜ · λᵢₖ ≤ Cₜ              ∀t        (capacity)      → dual: πₜ
  Σₖ λᵢₖ = 1                        ∀i        (convexity)     → dual: μᵢ
  Σₖ zₖₜᵤ · λᵢₖ = 0                 ∀(t,u)∈Θ⁰ᵢ (arc link)     → dual: τᵢₜᵤ
  Σₖ yₖₜ · λᵢₖ = 0                  ∀t∈Υ⁰ᵢ    (setup link)    → dual: σᵢₜ
  λᵢₖ ≥ 0                           ∀i,k
```

**Note:** The linking constraints for Θ⁰ and Υ⁰ are explicit in the RMP. The constraints for Θ¹ and Υ¹ are handled implicitly in the pricing subproblem (columns that don't satisfy forced arcs/setups are simply not generated).

### 3.3 Branching Linking Constraints

During branch-and-bound, we track four sets of branching decisions per item:

**Arc Branching (on z variables):**
```
Θ⁰ᵢ = {(t,u) : zᵢₜᵤ fixed to 0}  →  theta_0_by_item[i]
Θ¹ᵢ = {(t,u) : zᵢₜᵤ fixed to 1}  →  theta_1_by_item[i]
```

**Setup Branching (on y variables):**
```
Υ⁰ᵢ = {t : yᵢₜ fixed to 0}       →  upsilon_0_by_item[i]
Υ¹ᵢ = {t : yᵢₜ fixed to 1}       →  upsilon_1_by_item[i]
```

**Code: BranchNode dataclass**
```python
@dataclass
class BranchNode:
    """
    Represents a node in the branch-and-bound tree.
    
    Branching constraints:
        theta_0_by_item[i] = Θ⁰ᵢ: arcs (t,u) fixed to 0
        theta_1_by_item[i] = Θ¹ᵢ: arcs (t,u) fixed to 1
        upsilon_0_by_item[i] = Υ⁰ᵢ: setups t fixed to 0
        upsilon_1_by_item[i] = Υ¹ᵢ: setups t fixed to 1
    """
    node_id: int
    parent_id: Optional[int]
    depth: int
    
    # Arc branching: Θ⁰ᵢ and Θ¹ᵢ
    theta_0_by_item: Dict[int, Set[Tuple[int, int]]] = field(default_factory=dict)
    theta_1_by_item: Dict[int, Set[Tuple[int, int]]] = field(default_factory=dict)
    
    # Setup branching: Υ⁰ᵢ and Υ¹ᵢ
    upsilon_0_by_item: Dict[int, Set[int]] = field(default_factory=dict)
    upsilon_1_by_item: Dict[int, Set[int]] = field(default_factory=dict)
    
    # Node state
    lp_bound: float = math.inf
    is_integer: bool = False
    is_pruned: bool = False
    prune_reason: Optional[str] = None
    
    # Branching decision that created this node
    branch_variable: Optional[Tuple] = None  # (item_id, var_type, indices, value)
    branch_direction: Optional[str] = None
```

---

## 4. Restricted Master Problem Implementation

### 4.1 RMP Class Structure

**Code: RestrictedMasterProblem class**
```python
class RestrictedMasterProblem:
    """
    Restricted Master Problem for Dantzig-Wolfe decomposition.
    
    minimize   Σᵢ Σₖ cₖ · λᵢₖ
    subject to:
        Σᵢ Σₖ aₖₜ · λᵢₖ ≤ Cₜ           ∀t    (capacity)      → dual: πₜ
        Σₖ λᵢₖ = 1                     ∀i    (convexity)     → dual: μᵢ
        Σₖ zₖₜᵤ · λᵢₖ = 0             ∀(t,u) ∈ Θ⁰ᵢ          → dual: τᵢₜᵤ
        Σₖ yₖₜ · λᵢₖ = 0              ∀t ∈ Υ⁰ᵢ              → dual: σᵢₜ
        λᵢₖ ≥ 0
    """
    
    def __init__(
        self,
        items: List[ProductionItem],
        capacity: List[float],
        theta_0_by_item: Dict[int, Set[Tuple[int, int]]],
        upsilon_0_by_item: Dict[int, Set[int]],
    ):
```

### 4.2 RMP Initialization

```python
def __init__(self, items, capacity, theta_0_by_item, upsilon_0_by_item):
    self.items = items
    self.T = len(capacity)
    self.capacity = capacity
    self.theta_0_by_item = theta_0_by_item
    self.upsilon_0_by_item = upsilon_0_by_item
    
    self.model = gurobi.Model("RMP")
    self.model.Params.OutputFlag = 0
    self.model.Params.Method = 1  # Dual Simplex

    # Column storage
    self.columns: Dict[int, List[ProductionPlanColumn]] = {i.item_id: [] for i in items}
    self.lambdas: Dict[Tuple[int, int], gurobi.Var] = {}
    self.column_signatures: Set[str] = set()  # For duplicate detection
    
    # Build constraint expressions
    self.convex_expr = {i.item_id: gurobi.LinExpr(0.0) for i in items}
    self.cap_expr = [gurobi.LinExpr(0.0) for _ in range(self.T)]
    
    # Arc linking expressions: Σₖ zₖₜᵤ · λᵢₖ for each (i, t, u) ∈ Θ⁰
    self.arc_link_expr: Dict[Tuple[int, int, int], gurobi.LinExpr] = {}
    for i in items:
        for (t, u) in theta_0_by_item.get(i.item_id, set()):
            self.arc_link_expr[(i.item_id, t, u)] = gurobi.LinExpr(0.0)
    
    # Setup linking expressions: Σₖ yₖₜ · λᵢₖ for each (i, t) ∈ Υ⁰
    self.setup_link_expr: Dict[Tuple[int, int], gurobi.LinExpr] = {}
    for i in items:
        for t in upsilon_0_by_item.get(i.item_id, set()):
            self.setup_link_expr[(i.item_id, t)] = gurobi.LinExpr(0.0)

    # Add constraints
    # Convexity: Σₖ λᵢₖ = 1
    for item in items:
        self.convex_con[item.item_id] = self.model.addConstr(
            self.convex_expr[item.item_id] == 1.0, name=f"conv_{item.item_id}"
        )
    
    # Capacity: Σᵢ Σₖ aₖₜ · λᵢₖ ≤ Cₜ
    for t in range(self.T):
        self.cap_con.append(
            self.model.addConstr(self.cap_expr[t] <= self.capacity[t], name=f"cap_{t}")
        )
    
    # Arc linking: Σₖ zₖₜᵤ · λᵢₖ = 0 for (t,u) ∈ Θ⁰ᵢ
    for (item_id, t, u), expr in self.arc_link_expr.items():
        self.arc_link_con[(item_id, t, u)] = self.model.addConstr(
            expr == 0.0, name=f"arclink_{item_id}_{t}_{u}"
        )
    
    # Setup linking: Σₖ yₖₜ · λᵢₖ = 0 for t ∈ Υ⁰ᵢ
    for (item_id, t), expr in self.setup_link_expr.items():
        self.setup_link_con[(item_id, t)] = self.model.addConstr(
            expr == 0.0, name=f"setuplink_{item_id}_{t}"
        )
```

### 4.3 Adding Columns to RMP

When a column is added, all relevant expressions are updated:

```python
def add_column(self, col: ProductionPlanColumn) -> bool:
    """Add a column to the RMP. Returns True if added (not duplicate)."""
    sig = col.get_signature()
    if sig in self.column_signatures:
        return False  # Duplicate detection
    self.column_signatures.add(sig)
    
    item_id = col.item_id
    idx = len(self.columns[item_id])
    
    # Create λᵢₖ variable with objective coefficient cₖ
    lam = self.model.addVar(
        lb=0.0, vtype=GRB.CONTINUOUS, obj=col.total_plan_cost,
        name=f"lam_{item_id}_{idx}"
    )
    self.lambdas[(item_id, idx)] = lam
    self.columns[item_id].append(col)

    # Update convexity expression: Σₖ λᵢₖ
    self.convex_expr[item_id] += lam
    
    # Update capacity expressions: Σₖ aₖₜ · λᵢₖ
    for t in range(self.T):
        if col.capacity_usage_by_period[t] > 0:
            self.cap_expr[t] += col.capacity_usage_by_period[t] * lam
    
    # Update arc linking expressions: Σₖ zₖₜᵤ · λᵢₖ
    for (t, u) in col.arc_usage:
        if (item_id, t, u) in self.arc_link_expr:
            self.arc_link_expr[(item_id, t, u)] += lam
    
    # Update setup linking expressions: Σₖ yₖₜ · λᵢₖ
    for t in range(self.T):
        if col.setup_by_period[t] > 0.5:
            if (item_id, t) in self.setup_link_expr:
                self.setup_link_expr[(item_id, t)] += lam

    self._rebuild_constraints()
    return True
```

### 4.4 Solving RMP and Extracting Duals

```python
def solve(self) -> Tuple[float, Dict[int, float], List[float], 
                         Dict[int, Dict[Tuple[int, int], float]], 
                         Dict[int, Dict[int, float]]]:
    """
    Solve the RMP and return objective and duals.
    
    Returns:
        (obj, mu, pi, tau_by_item, sigma_by_item) where:
        - obj: objective value
        - mu[i]: convexity dual μᵢ
        - pi[t]: capacity dual πₜ
        - tau_by_item[i][(t,u)]: arc linking dual τᵢₜᵤ
        - sigma_by_item[i][t]: setup linking dual σᵢₜ
    """
    self.model.optimize()
    
    if self.model.status != GRB.OPTIMAL:
        return math.inf, {}, [], {}, {}
    
    # Extract convexity duals μᵢ
    mu = {i.item_id: self.convex_con[i.item_id].Pi for i in self.items}
    
    # Extract capacity duals πₜ
    pi = [self.cap_con[t].Pi for t in range(self.T)]
    
    # Extract arc linking duals τᵢₜᵤ
    tau_by_item: Dict[int, Dict[Tuple[int, int], float]] = {i.item_id: {} for i in self.items}
    for (item_id, t, u), con in self.arc_link_con.items():
        tau_by_item[item_id][(t, u)] = con.Pi
    
    # Extract setup linking duals σᵢₜ
    sigma_by_item: Dict[int, Dict[int, float]] = {i.item_id: {} for i in self.items}
    for (item_id, t), con in self.setup_link_con.items():
        sigma_by_item[item_id][t] = con.Pi
    
    return self.model.ObjVal, mu, pi, tau_by_item, sigma_by_item
```

### 4.5 Dummy Column Detection

```python
def has_active_dummy(self, eps: float = 1e-6) -> bool:
    """Check if any dummy column has positive λ value."""
    for item in self.items:
        if self.columns[item.item_id]:
            # Dummy is always column index 0
            lam = self.lambdas.get((item.item_id, 0))
            if lam is not None and lam.X > eps:
                col = self.columns[item.item_id][0]
                # Dummy has empty arc_usage
                if not col.arc_usage:
                    return True
    return False
```

---

## 5. Pricing Subproblem: Dynamic Programming Formulation

The pricing subproblem finds a column with negative reduced cost. For ZIO columns, this is solved via **backward dynamic programming**.

### 5.1 Reduced Cost Structure

For a column k of item i, the reduced cost is:

```
r̄ₖ = cₖ - μᵢ - Σₜ πₜ·aₖₜ - Σₜᵤ τᵢₜᵤ·zₖₜᵤ - Σₜ σᵢₜ·yₖₜ
```

Decomposing by production run, the reduced cost of a run from t covering demands {t, ..., s} is:

```
r̄(t,s) = sᵢₜ - σᵢₜ + Σᵤ₌ₜ^s [(cᵢₜ + H(t,u) - πₜ)·dᵤ - τᵢₜᵤ]
```

**Code: run_reduced_cost function**
```python
def run_reduced_cost(t: int, s: int) -> float:
    """
    Reduced cost of production run from t covering demands [t, s].
    
    r̄(t,s) = sᵢₜ - σᵢₜ + Σᵤ₌ₜ^s [(cᵢₜ + H(t,u) - πₜ)·dᵤ - τᵢₜᵤ]
    """
    cost = setup_cost[t]
    cost -= sigma.get(t, 0.0)  # Subtract setup dual -σᵢₜ
    
    q = block_demand(t, s)
    if q > 0:
        hold = block_holding_cost(t, s)
        cost += prod_cost[t] * q + hold
        cost -= capacity_duals[t] * q  # Subtract capacity dual -πₜ·aₖₜ
    
    # Subtract arc duals -τᵢₜᵤ
    for u in range(t, s + 1):
        if demand[u] > 0:
            cost -= tau.get((t, u), 0.0)
    
    return cost
```

### 5.2 DP State Definition and Recursion

**State Variable:**
```
f(t) = minimum reduced cost to satisfy demands {t, t+1, ..., T−1}
```

**Boundary Condition:**
```
f(T) = 0
```

**Backward Recursion:**
```
f(t) = min {
    f(t+1)                           if dₜ = 0 and t ∉ Υ¹ᵢ     (skip)
    sᵢₜ - σᵢₜ + f(t+1)               if dₜ = 0 and t ∈ Υ¹ᵢ     (setup only)
    r̄(t,s) + f(s+1)                  for s ∈ valid_ends(t)     (produce)
}
```

**Decision Tracking:**
```
decision[t] = (action, s) where:
  action = -1: skip period
  action =  0: setup only (no production)
  action =  1: produce from t to s
```

### 5.3 Branching Constraint Handling in DP

The pricing DP must respect all branching constraints through feasibility checks and forced decisions.

**Pre-processing (Infeasibility Detection):**
```python
# Check for conflicting setup constraints: yₜ=1 and yₜ=0
for t in upsilon_1:
    if t in upsilon_0:
        return INF, None  # Conflict

# Check for conflicting arc constraints: demand u forced from multiple sources
forced_source: Dict[int, int] = {}  # u -> t
must_produce_at: Set[int] = set()
min_end_for_forced: Dict[int, int] = {}

for (t, u) in theta_1:
    if u in forced_source and forced_source[u] != t:
        return INF, None  # Demand u forced from multiple sources
    forced_source[u] = t
    must_produce_at.add(t)
    min_end_for_forced[t] = max(min_end_for_forced.get(t, t), u)

# If forced to produce at t but setup forbidden at t -> infeasible
for t in must_produce_at:
    if t in upsilon_0:
        return INF, None
```

**Valid End Periods (get_valid_ends):**
```python
def get_valid_ends(t: int) -> List[int]:
    """
    Get valid end periods s for a production run starting at t.
    Must satisfy:
    1. Shelf-life: s ≤ vᵢₜ (can satisfy demand up to expiry)
    2. Forced arcs: s ≥ min_end_for_forced[t]
    3. No forbidden arcs in [t, s]
    4. No conflicting forced arcs (demand u forced from different t')
    """
    v_t = expiry_abs[t]  # Expiry period (inclusive last usable)
    s_max = min(T - 1, v_t)
    
    min_s = min_end_for_forced.get(t, t)
    
    valid = []
    for s in range(t, s_max + 1):
        if s < min_s:
            continue
        
        # Check no forbidden arcs in [t, s]
        forbidden = False
        for u in range(t, s + 1):
            if demand[u] > 0 and (t, u) in theta_0:
                forbidden = True
                break
        if forbidden:
            continue
        
        # Check forced arcs consistency
        conflict = False
        for u in range(t, s + 1):
            if demand[u] > 0 and u in forced_source and forced_source[u] != t:
                conflict = True
                break
        if conflict:
            continue
        
        valid.append(s)
    
    return valid
```

### 5.4 Complete DP Pricing Function

```python
def dp_pricing_for_single_item(
    item: ProductionItem,
    capacity_duals: List[float],  # πₜ
    convexity_dual: float,  # μᵢ
    theta_0: Set[Tuple[int, int]],  # Θ⁰ᵢ: forbidden arcs
    theta_1: Set[Tuple[int, int]],  # Θ¹ᵢ: forced arcs
    upsilon_0: Set[int],  # Υ⁰ᵢ: forbidden setups
    upsilon_1: Set[int],  # Υ¹ᵢ: forced setups
    tau: Dict[Tuple[int, int], float],  # τᵢₜᵤ: arc linking duals
    sigma: Dict[int, float],  # σᵢₜ: setup linking duals
    epsilon: float = 1e-9,
) -> Tuple[float, Optional[ProductionPlanColumn]]:
    """
    DP-based pricing for ZIO lot-sizing with perishability.
    
    Returns:
        (reduced_cost, column) or (INF, None) if infeasible
    """
    # ... [Pre-processing for conflicts as shown above] ...
    
    # === Backward DP: f(t) = min reduced cost to satisfy {t, ..., T-1} ===
    f = [INF] * (T + 1)
    f[T] = 0.0
    decision: List[Tuple[int, int]] = [(-1, -1)] * (T + 1)

    for t in range(T - 1, -1, -1):
        must_setup = t in upsilon_1 or t in must_produce_at
        can_setup = t not in upsilon_0
        
        # Option 1: Skip period (no setup, no production)
        if demand[t] == 0 and not must_setup:
            if f[t + 1] < f[t]:
                f[t] = f[t + 1]
                decision[t] = (-1, -1)
        
        # Option 2: Setup only (no production)
        if demand[t] == 0 and can_setup and must_setup and t not in must_produce_at:
            setup_only_cost = setup_cost[t] - sigma.get(t, 0.0)
            if setup_only_cost + f[t + 1] < f[t]:
                f[t] = setup_only_cost + f[t + 1]
                decision[t] = (0, -1)
        
        # Option 3: Produce from t to some valid end s
        if can_setup:
            valid_ends = get_valid_ends(t)
            for s in valid_ends:
                q = block_demand(t, s)
                if q <= 0 and t not in must_produce_at:
                    continue
                
                run_cost = run_reduced_cost(t, s)
                next_t = s + 1
                
                if run_cost + f[next_t] < f[t]:
                    f[t] = run_cost + f[next_t]
                    decision[t] = (1, s)

    if f[0] >= INF:
        return INF, None

    # === Reconstruct solution ===
    # ... [Build column from decisions] ...
    
    # Compute final reduced cost
    rc = total_cost - convexity_dual
    for t in range(T):
        if cap_usage[t] > 0:
            rc -= capacity_duals[t] * cap_usage[t]
        if setup_usage[t] > 0.5:
            rc -= sigma.get(t, 0.0)
    for (t, u) in arc_usage:
        rc -= tau.get((t, u), 0.0)

    if rc < -epsilon:
        return rc, ProductionPlanColumn(...)
    else:
        return rc, None
```

---

## 6. Column Generation Algorithm

### 6.1 Main Loop

```python
def solve_node_with_column_generation(
    items: List[ProductionItem],
    capacity: List[float],
    node: BranchNode,
    inherited_columns: Optional[Dict[int, List[ProductionPlanColumn]]] = None,
    max_iter: int = 500,
    eps: float = 1e-9,
    verbose: bool = False,
    stats: Optional[SearchStatistics] = None,
) -> Tuple[float, RestrictedMasterProblem, bool, Dict]:
    """Solve a single B&B node via column generation."""
    
    # Initialize RMP with branching constraints
    rmp = RestrictedMasterProblem(
        items, capacity,
        node.theta_0_by_item,
        node.upsilon_0_by_item,
    )

    # Add dummy columns for initial feasibility
    for item in items:
        dummy_col = ProductionPlanColumn(
            item_id=item.item_id,
            total_plan_cost=10000.0 * (sum(item.demand_quantity_by_period) + 1),
            capacity_usage_by_period=[0.0] * T,
            setup_by_period=[0.0] * T,
            arc_usage={},  # Empty = no demand covered
        )
        rmp.add_column(dummy_col)

    # Add inherited columns (filtered for feasibility)
    if inherited_columns:
        for item in items:
            item_id = item.item_id
            theta_0 = node.theta_0_by_item.get(item_id, set())
            theta_1 = node.theta_1_by_item.get(item_id, set())
            upsilon_0 = node.upsilon_0_by_item.get(item_id, set())
            upsilon_1 = node.upsilon_1_by_item.get(item_id, set())
            
            for col in inherited_columns.get(item_id, []):
                if not col.violates_branching(theta_0, theta_1, upsilon_0, upsilon_1):
                    rmp.add_column(col)

    # Column generation iterations
    for iteration in range(1, max_iter + 1):
        lb, mu, pi, tau_by_item, sigma_by_item = rmp.solve()
        
        if not math.isfinite(lb):
            return math.inf, rmp, False, {}

        any_added = False
        for item in items:
            item_id = item.item_id
            theta_0 = node.theta_0_by_item.get(item_id, set())
            theta_1 = node.theta_1_by_item.get(item_id, set())
            upsilon_0 = node.upsilon_0_by_item.get(item_id, set())
            upsilon_1 = node.upsilon_1_by_item.get(item_id, set())
            tau = tau_by_item.get(item_id, {})
            sigma = sigma_by_item.get(item_id, {})

            rc, col = dp_pricing_for_single_item(
                item,
                capacity_duals=pi,
                convexity_dual=mu[item_id],
                theta_0=theta_0,
                theta_1=theta_1,
                upsilon_0=upsilon_0,
                upsilon_1=upsilon_1,
                tau=tau,
                sigma=sigma,
                epsilon=eps,
            )

            if col is not None and rc < -eps:
                if rmp.add_column(col):
                    any_added = True

        if not any_added:
            z_vals = extract_z_values(rmp, items, eps)
            return lb, rmp, True, z_vals  # Converged

    # Max iterations reached
    lb, _, _, _, _ = rmp.solve()
    z_vals = extract_z_values(rmp, items, eps)
    return lb, rmp, False, z_vals
```

### 6.2 Termination Criterion

Column generation terminates when no pricing subproblem finds a column with r̄ₖ < −ε (where ε = 10⁻⁶).

### 6.3 Dummy Columns

Each item starts with a dummy column to ensure initial RMP feasibility:

```python
dummy_cost = 10000 * (total_demand + 1)
capacity_usage = [0] * T
setup_by_period = [0] * T
arc_usage = {}  # Empty - no demand covered
```

**A solution is valid only if no dummy column has positive λ value.**

### 6.4 Extracting Solution Values

```python
def extract_z_values(rmp, items, eps) -> Dict[int, Dict[Tuple[int, int], float]]:
    """Extract z[i][(t,u)] = Σₖ zₖₜᵤ · λᵢₖ values from RMP solution."""
    z = {i.item_id: {} for i in items}
    for (item_id, idx), lam in rmp.lambdas.items():
        lam_val = lam.X
        if lam_val < eps:
            continue
        col = rmp.columns[item_id][idx]
        for (t, u) in col.arc_usage:
            z[item_id][(t, u)] = z[item_id].get((t, u), 0.0) + lam_val
    return z

def extract_y_values(rmp, items, eps) -> Dict[int, Dict[int, float]]:
    """Extract y[i][t] = Σₖ yₖₜ · λᵢₖ values from RMP solution."""
    T = rmp.T
    y = {i.item_id: {t: 0.0 for t in range(T)} for i in items}
    for (item_id, idx), lam in rmp.lambdas.items():
        lam_val = lam.X
        if lam_val < eps:
            continue
        col = rmp.columns[item_id][idx]
        for t in range(T):
            if col.setup_by_period[t] > 0.5:
                y[item_id][t] += lam_val
    return y
```

---

## 7. Branch-and-Bound Tree Search

### 7.1 Search Strategy

The algorithm uses **Best-First Search** with a priority queue ordered by:
1. LP bound (lowest first) - primary key
2. Depth (deepest first among ties) - secondary key for "diving" behavior

```python
# Priority queue: (lp_bound, -depth, node_id, node, inherited_columns)
heap: List[Tuple[float, int, int, BranchNode, Dict]] = []
heapq.heappush(heap, (node.lp_bound, -node.depth, node.node_id, node, columns))
```

This ensures:
- Monotonically improving global lower bound
- Early detection of optimal solutions
- Efficient pruning via incumbent bound
- "Dive" behavior among nodes with equal bounds

### 7.2 Variable Selection

Branching selects the **most fractional** variable with priority:

**Priority 1: Arc variables z**
```python
def find_most_fractional_z(z_vals, node, eps) -> Optional[Tuple[int, int, int, float]]:
    """Find most fractional arc variable z[i,(t,u)]."""
    best_frac = 0.0
    best = None
    for item_id, arcs in z_vals.items():
        theta_0 = node.theta_0_by_item.get(item_id, set())
        theta_1 = node.theta_1_by_item.get(item_id, set())
        for (t, u), val in arcs.items():
            if (t, u) in theta_0 or (t, u) in theta_1:
                continue  # Already fixed
            frac = min(val, 1.0 - val)
            if frac > eps and frac > best_frac:
                best_frac = frac
                best = (item_id, t, u, val)
    return best
```

**Priority 2: Setup variables y (fallback)**
```python
def find_most_fractional_y(y_vals, node, eps) -> Optional[Tuple[int, int, float]]:
    """Find most fractional setup variable y[i,t]."""
    best_frac = 0.0
    best = None
    for item_id, setups in y_vals.items():
        upsilon_0 = node.upsilon_0_by_item.get(item_id, set())
        upsilon_1 = node.upsilon_1_by_item.get(item_id, set())
        for t, val in setups.items():
            if t in upsilon_0 or t in upsilon_1:
                continue  # Already fixed
            frac = min(val, 1.0 - val)
            if frac > eps and frac > best_frac:
                best_frac = frac
                best = (item_id, t, val)
    return best
```

### 7.3 Column Inheritance

Child nodes inherit columns from parent if they remain feasible under the new branching constraints:

```python
if inherited_columns:
    for item in items:
        item_id = item.item_id
        theta_0 = node.theta_0_by_item.get(item_id, set())
        theta_1 = node.theta_1_by_item.get(item_id, set())
        upsilon_0 = node.upsilon_0_by_item.get(item_id, set())
        upsilon_1 = node.upsilon_1_by_item.get(item_id, set())
        
        for col in inherited_columns.get(item_id, []):
            if not col.violates_branching(theta_0, theta_1, upsilon_0, upsilon_1):
                rmp.add_column(col)
```

### 7.4 Fathoming Conditions

A node is fathomed (pruned) when:

| Condition | Code | Description |
|-----------|------|-------------|
| **Infeasible** | `prune_reason = "infeasible"` | RMP has no feasible solution (LP bound = ∞) |
| **Bound** | `prune_reason = "bound"` | LP bound ≥ incumbent upper bound |
| **Integer** | `prune_reason = "integer"` | Solution is integer and valid |
| **Dummy** | `prune_reason = "dummy"` | Optimal uses dummy columns (original problem infeasible) |
| **Duplicate** | `prune_reason = "duplicate"` | Node signature already explored |

```python
# Fathoming checks in main loop
if not math.isfinite(lb):
    node.is_pruned = True
    node.prune_reason = "infeasible"
    continue

if best_ub is not None and lb >= best_ub - eps:
    node.is_pruned = True
    node.prune_reason = "bound"
    continue

if rmp.has_active_dummy(eps):
    node.is_pruned = True
    node.prune_reason = "dummy"
    continue

if is_integer(z_vals, y_vals, eps):
    node.is_integer = True
    if best_ub is None or lb < best_ub - eps:
        best_ub = lb  # New incumbent
        fathom_heap_by_incumbent(heap, best_ub, eps)
    continue
```

### 7.5 Creating Child Nodes

**Arc Branching (Z=0 and Z=1):**
```python
branch_var = find_most_fractional_z(z_vals, node, eps)
if branch_var is not None:
    item_id, t_br, u_br, z_val = branch_var
    
    # Z=0 branch: add (t_br, u_br) to Θ⁰ᵢ
    left = BranchNode(...)
    left.theta_0_by_item[item_id].add((t_br, u_br))
    
    # Z=1 branch: add (t_br, u_br) to Θ¹ᵢ
    right = BranchNode(...)
    right.theta_1_by_item[item_id].add((t_br, u_br))
```

**Setup Branching (Y=0 and Y=1):**
```python
branch_var_y = find_most_fractional_y(y_vals, node, eps)
if branch_var_y is not None:
    item_id, t_br, y_val = branch_var_y
    
    # Y=0 branch: add t_br to Υ⁰ᵢ
    left = BranchNode(...)
    left.upsilon_0_by_item[item_id].add(t_br)
    
    # Y=1 branch: add t_br to Υ¹ᵢ
    right = BranchNode(...)
    right.upsilon_1_by_item[item_id].add(t_br)
```

---

## 8. LEFO Compliance Through ZIO Structure

The **Last-Expired-First-Out** policy is enforced implicitly through the **Zero-Inventory-Ordering (ZIO)** column structure:

1. **Contiguous Coverage:** Each production run at t covers a contiguous range of demand periods [t, s] where s ∈ Γᵢₜ. Production can start from a zero-demand period (unlike Wagner-Whitin).

2. **No Inventory Overlap (ZIO):** Between production runs, inventory reaches zero. This prevents mixing batches with different expiry dates.

3. **Shelf-Life Feasibility:** The set Γᵢₜ restricts coverage to periods before expiry (vᵢₜ).

Together, these properties ensure that when a demand period u is satisfied by production at t, no newer production (with later expiry) exists in inventory—satisfying LEFO.

**Important:** In the master LP, convex combinations of ZIO columns still satisfy LEFO because:
- Every column in Ωᵢ is LEFO-feasible by construction
- LEFO constraints are linear
- Any convex combination of LEFO-feasible points remains LEFO-feasible

---

## 9. Notation Summary: Code ↔ Mathematics

| Code Variable | Math Notation | Description |
|---------------|---------------|-------------|
| `item.perishability_horizon_by_start_period[t]` | vᵢₜ = t + mᵢₜ | Expiry period |
| `get_valid_ends(t)` | Γᵢₜ | Valid demand coverage set |
| `capacity_duals` / `pi[t]` | πₜ | Capacity constraint dual |
| `convexity_dual` / `mu[i]` | μᵢ | Convexity constraint dual |
| `tau[(t,u)}` | τᵢₜᵤ | Arc linking dual |
| `sigma[t]` | σᵢₜ | Setup linking dual |
| `theta_0_by_item[i]` | Θ⁰ᵢ | Arcs fixed to 0 |
| `theta_1_by_item[i]` | Θ¹ᵢ | Arcs fixed to 1 |
| `upsilon_0_by_item[i]` | Υ⁰ᵢ | Setups fixed to 0 |
| `upsilon_1_by_item[i]` | Υ¹ᵢ | Setups fixed to 1 |
| `f[t]` | f(t) | DP cost-to-go |
| `decision[t]` | (action, s) | DP decision at t |
| `lambdas[(i,k)]` | λᵢₖ | Column weight |
| `columns[i][k]` | Column k ∈ Ωᵢ | Production plan |
| `arc_usage[(t,u)]` | zₖₜᵤ | Arc indicator in column |
| `setup_by_period[t]` | yₖₜ | Setup indicator in column |
| `capacity_usage_by_period[t]` | aₖₜ | Capacity usage in column |
| `total_plan_cost` | cₖ | Column cost |
| `convex_con[i]` | Σₖ λᵢₖ = 1 | Convexity constraint |
| `cap_con[t]` | Σᵢₖ aₖₜλᵢₖ ≤ Cₜ | Capacity constraint |
| `arc_link_con[(i,t,u)]` | Σₖ zₖₜᵤλᵢₖ = 0 | Arc linking constraint |
| `setup_link_con[(i,t)]` | Σₖ yₖₜλᵢₖ = 0 | Setup linking constraint |

---

## 10. Statistics and Monitoring

### 10.1 SearchStatistics Class

```python
class SearchStatistics:
    def __init__(self):
        self.nodes_created = 0
        self.nodes_explored = 0
        self.nodes_integer = 0
        self.nodes_fathomed_by_bound = 0
        self.nodes_fathomed_by_infeasible = 0
        self.nodes_fathomed_integer = 0
        self.nodes_fathomed_on_incumbent = 0
        self.nodes_fathomed_duplicate = 0
        self.nodes_fathomed_dummy = 0
        self.max_depth = 0
        self.incumbent_history = []
        self.start_time = time.time()
        self.total_columns_generated = 0
        self.total_cg_iterations = 0
```

### 10.2 Node Signature for Duplicate Detection

```python
def node_signature(node: BranchNode) -> str:
    """Create unique signature for duplicate node detection."""
    sig_parts = []
    for item_id in sorted(node.theta_0_by_item.keys()):
        arcs = sorted(node.theta_0_by_item[item_id])
        if arcs:
            sig_parts.append(f"I{item_id}_Z0:{','.join(f'{t}-{u}' for t,u in arcs)}")
    for item_id in sorted(node.theta_1_by_item.keys()):
        arcs = sorted(node.theta_1_by_item[item_id])
        if arcs:
            sig_parts.append(f"I{item_id}_Z1:{','.join(f'{t}-{u}' for t,u in arcs)}")
    for item_id in sorted(node.upsilon_0_by_item.keys()):
        periods = sorted(node.upsilon_0_by_item[item_id])
        if periods:
            sig_parts.append(f"I{item_id}_Y0:{','.join(str(t) for t in periods)}")
    for item_id in sorted(node.upsilon_1_by_item.keys()):
        periods = sorted(node.upsilon_1_by_item[item_id])
        if periods:
            sig_parts.append(f"I{item_id}_Y1:{','.join(str(t) for t in periods)}")
    return "|".join(sig_parts)
```

---

## 11. JSON I/O Interface

### 11.1 Instance Format

```json
{
    "period": 8,
    "production_capacity": [15, 11, 15, 8, 8, 8, 9, 9],
    "items": {
        "1": {
            "demand": [6, 0, 5, 0, 3, 4, 0, 5],
            "c_var": [3.0, 3.0, 3.3, 3.3, 3.6, 3.6, 3.8, 3.8],
            "setup": [22.0, 22.0, 22.0, 22.0, 22.0, 22.0, 22.0, 22.0],
            "h": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "shelf_seq": [3, 3, 2, 2, 2, 2, 2, 2]
        }
    }
}
```

### 11.2 Main Entry Point

```python
def solve_instance(
    instance_path: str | Path = "last_instance.json",
    time_limit: int = 0,
    mip_gap: float = 0.0,
    out_dir: str | Path = "bp_results",
    max_nodes: int = 10000,
    verbose: bool = True,
) -> Tuple[dict, List[str]]:
    """
    Solve using Branch-and-Price (compatible with MIP solver interface).
    
    Returns:
        (summary_dict, orders_list)
    """
```

### 11.3 Output Format

**summary.json:**
```json
{
    "status": 2,
    "objective": 2371.53,
    "best_bound": 2371.53,
    "gap": 0.0,
    "runtime_sec": 1.23,
    "solver_version": "bp_best_first_v2",
    "n_items": 3,
    "T": 8,
    "nodes_explored": 15,
    "columns_generated": 42,
    "cg_iterations": 78
}
```

**orders.txt:**
```
Item 1 — orders (t → qty)
  0 →   11.000
  4 →    7.000
  7 →    5.000

Item 2 — orders (t → qty)
  1 →    7.000
  3 →    6.000
  5 →    9.000
```

---

## 12. Example Run Output

```
╔════════════════════════════════════════════════════════════════════╗
║       CAPACITATED LOT SIZING WITH PERISHABILITY (B&P)              ║
╠════════════════════════════════════════════════════════════════════╣
║  Items:    3                                                       ║
║  Periods:  8                                                       ║
╚════════════════════════════════════════════════════════════════════╝

======================================================================
            BRANCH-AND-PRICE WITH DP PRICING (BEST-FIRST)
======================================================================

>>> ROOT NODE <<<
  └─ CG: LB=2371.53
  Root LB:  2371.5309
  Integer?  False

======================================================================
BEST-FIRST SEARCH (Best LP Bound)
======================================================================

N   1 D 1  └─ CG: LB=2531.93  FATHOMED: 2531.93 ≥ 2500.00
N   2 D 1  └─ CG: LB=2384.49  Branch Z[1,0,2]=0.455
...
N  15 D 5  └─ CG: LB=2371.53  INTEGER: 2371.53 ★ NEW INCUMBENT!

======================================================================
                          FINAL RESULTS
======================================================================
  Time elapsed:       0.85 seconds
  Nodes created:      16
  Nodes explored:     15
  Integer solutions:  1
  Max depth:          5
  Total CG iters:     78
  Total cols gen:     42

  Fathomed nodes:     15
    By bound:         8
    Infeasible:       2
    Integer:          1
    On incumbent:     3
    Duplicate:        0
    Dummy active:     1

  Best lower bound:   2371.5309
  Best upper bound:   2371.5309
  Gap:                0.0000 (0.00%)

  ★★★ PROVEN OPTIMAL! ★★★
======================================================================
```

---

## 13. Algorithm Flow Summary

```
1. INITIALIZE
   └─ Create root node with empty branching sets
   └─ Solve root via column generation
   └─ If integer and no dummy → OPTIMAL

2. MAIN LOOP (while heap not empty and nodes < max)
   └─ Pop node with lowest LP bound
   └─ Solve node via column generation with inherited columns
   └─ Check fathoming conditions:
       ├─ Infeasible → prune
       ├─ Bound ≥ incumbent → prune
       ├─ Dummy active → prune
       └─ Integer → update incumbent, prune
   └─ Find most fractional variable (z first, then y)
   └─ Create two child nodes:
       ├─ Variable = 0 branch
       └─ Variable = 1 branch
   └─ Add children to heap (if not duplicate)

3. TERMINATE
   └─ Report best solution found
   └─ Write output files
```

---

*End of Documentation*
