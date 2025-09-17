def administer_meds(delta_t, tstop):
    t = 0
    while t < tstop: 
        print(f"Administering meds at t={t}")
        t += delta_t

# -----2b. A first test case-----
administer_meds(0.25, 1)

# -----2c. A second test case-----
administer_meds(0.1, 1)

# -----2f. A safer implementation-----
def administer_meds_rev(delta_t, tstop):
    num = int(tstop / delta_t)
    t = 0
    for _ in range(num): 
        print(f"Administering meds at t={t:.4f}")
        t += delta_t

administer_meds_rev(0.1, 1)