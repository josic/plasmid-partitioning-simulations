import glob, re, random, statistics, math

files = sorted(glob.glob("loss_run_*.txt"))
vals = []

for f in files:
    txt = open(f).read()
    m = re.search(r"Loss prob\s+([0-9.eE+-]+)", txt)
    if m:
        vals.append(float(m.group(1)))

if not vals:
    raise SystemExit("No Loss prob values found.")

mean = statistics.mean(vals)

# Bootstrap over run-level estimates
B = 10000
boots = []
for _ in range(B):
    sample = [random.choice(vals) for _ in vals]
    boots.append(statistics.mean(sample))

boot_var = statistics.variance(boots)
boot_sd = math.sqrt(boot_var)
ci = sorted(boots)
lo = ci[int(0.025 * B)]
hi = ci[int(0.975 * B)]

print(f"Number of files used: {len(vals)}")
print(f"Estimated loss probability: {mean:.12g}")
print(f"Bootstrap variance: {boot_var:.12g}")
print(f"Bootstrap standard error: {boot_sd:.12g}")
print(f"95% bootstrap CI: [{lo:.12g}, {hi:.12g}]")
