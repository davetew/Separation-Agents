from sep_agents.properties.ree_databases import build_ree_system
import reaktoro as rkt

system = build_ree_system("light_ree")
print("Species in light_ree system:")
for sp in system.species():
    print(sp.name())
