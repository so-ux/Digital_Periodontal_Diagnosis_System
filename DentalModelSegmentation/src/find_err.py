import math

total = 50
teeth_per_inst = 14
crash = 2

avg_tla = 0.015
avg_tsa = 0.98
avg_tir = 0.95

print(
    math.exp(-(avg_tla * (total - crash) * teeth_per_inst + 5 * crash * teeth_per_inst) / (total * teeth_per_inst)),
    avg_tir * (total - crash) / total,
    avg_tsa * (total - crash) / total
)
