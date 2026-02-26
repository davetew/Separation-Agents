"""Catalog REE species in Reaktoro databases."""
import reaktoro as rkt
import re

db = rkt.SupcrtDatabase("supcrtbl")
ree_elements = ["La", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Y", "Sc"]

by_element = {}
minerals = []
aqueous = []

for sp in db.species():
    name = sp.name()
    formula = sp.formula().str() if hasattr(sp.formula(), "str") else str(sp.formula())
    
    matched_elems = []
    for elem in ree_elements:
        if re.search(rf"{elem}(?![a-z])", formula) or re.search(rf"{elem}(?![a-z])", name):
            matched_elems.append(elem)
    
    if matched_elems:
        for elem in matched_elems:
            by_element.setdefault(elem, []).append(name)
        
        if "(aq)" in name or "+" in name or name.endswith("-"):
            aqueous.append(name)
        else:
            minerals.append(name)

print("=== REE species count per element ===")
for elem in ree_elements:
    count = len(by_element.get(elem, []))
    print(f"  {elem:3s}: {count:3d} species")

print(f"\n=== REE Mineral phases ({len(set(minerals))}) ===")
for m in sorted(set(minerals)):
    sp = db.species().get(m)
    formula = sp.formula().str() if hasattr(sp.formula(), "str") else str(sp.formula())
    print(f"  {m:35s} {formula}")

print(f"\n=== Sample aqueous species (Ce) ===")
for s in sorted(by_element.get("Ce", [])):
    if "(aq)" in s or "+" in s or s.endswith("-"):
        sp = db.species().get(s)
        formula = sp.formula().str() if hasattr(sp.formula(), "str") else str(sp.formula())
        print(f"  {s:35s} {formula}")
