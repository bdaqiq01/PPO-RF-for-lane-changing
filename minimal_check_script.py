import traci, os

SUMO = "sumo"  # use "sumo-gui" if you want to see it
cfg = os.path.join("SUMO_sim", "base_compl", "base.sumocfg")
dt = 0.1

traci.start([SUMO, "-c", cfg, "--step-length", str(dt)])
try:
    t_prev_ms = traci.simulation.getCurrentTime()  # in milliseconds
    print("t0(ms) =", t_prev_ms)
    for k in range(5):
        traci.simulationStep()  # advance one step
        t_now_ms = traci.simulation.getCurrentTime()
        advanced = (t_now_ms - t_prev_ms) / 1000.0
        print(f"step {k+1}: advanced by {advanced} sec")
        assert abs(advanced - dt) < 1e-9, "Mismatch between SUMO step and dt!"
        t_prev_ms = t_now_ms
finally:
    traci.close()
