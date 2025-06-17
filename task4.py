from pulp import *

# Model
model = LpProblem("ProfitMaximization", LpMaximize)

x = LpVariable("Product_A", lowBound=0, cat='Integer')
y = LpVariable("Product_B", lowBound=0, cat='Integer')

model += 40*x + 30*y  # Objective
model += 2*x + 1*y <= 100  # Machine 1 constraint
model += 1*x + 3*y <= 90   # Machine 2 constraint

model.solve()
print(f"Product A: {x.varValue}, Product B: {y.varValue}")
print(f"Profit: ₹{value(model.objective)}")
