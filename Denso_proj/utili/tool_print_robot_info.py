"""
Standalone Isaac Sim script to load a USD file and query each link's COM and inertia.
"""

import os
import time

# =============================================================================
# 1. Start the simulation app.
#
# IMPORTANT: The SimulationApp must be created before any other Isaac Sim modules
# are imported.
# =============================================================================
from omni.isaac.lab.app import AppLauncher


# Create the SimulationApp. Set headless to True if you don't need the GUI.
simulation_app = SimulationApp({"headless": False})
print("SimulationApp started.")

# =============================================================================
# 2. Open the USD stage.
#
# Use omni.usd to open your USD file. Update the usd_file_path variable to point to
# your file.
# =============================================================================
import omni.usd

usd_path = r"C:\Users\13306\OneDrive\Coding_Proj\issaaclab140\HEcode\Denso_proj"
# Replace with the actual path to your USD scene file.
usd_file_path = os.path.join(os.getcwd(), usd_path)
usd_context = omni.usd.get_context()
stage = usd_context.open_stage(usd_file_path)
print(f"Opened USD stage from: {usd_file_path}")

# Give the stage a moment to load.
simulation_app.update()
time.sleep(1.0)

# =============================================================================
# 3. Create an ArticulationView for the articulation.
#
# The ArticulationView provides access to dynamic properties for all links in an
# articulation. Specify the prim path of your articulation here.
# =============================================================================
from omni.isaac.core.articulations import ArticulationView

# Update this prim path to match your articulation in the stage.
articulation_prim_path = "/World/YourArticulationPrim"

# Create the view. The prim_paths_expr can be a single prim path or a pattern.
articulation_view = ArticulationView(
    prim_paths_expr=articulation_prim_path,
    name="articulation_view"
)
print(f"Created ArticulationView for prim: {articulation_prim_path}")

# It is a good idea to let the simulation update once more so that the articulation
# (and its dynamic properties) are properly initialized.
simulation_app.update()
time.sleep(1.0)

# =============================================================================
# 4. Query and print link properties.
#
# Retrieve the center-of-mass (COM) positions and inertias for each link in the
# articulation. The methods return (typically) a list of vectors (or arrays) with one
# element per link.
# =============================================================================
body_coms = articulation_view.get_body_coms()
body_inertias = articulation_view.get_body_inertias()

print("Link properties:")
for idx, (com, inertia) in enumerate(zip(body_coms, body_inertias)):
    print(f" Link {idx}:")
    print(f"   COM     = {com}")
    print(f"   Inertia = {inertia}")

# =============================================================================
# 5. (Optional) Run the simulation loop.
#
# You can run a short simulation loop if desired.
# =============================================================================
for _ in range(50):
    simulation_app.update()
    time.sleep(0.01)

# =============================================================================
# 6. Shut down the simulation.
# =============================================================================
simulation_app.close()
print("SimulationApp closed.")
