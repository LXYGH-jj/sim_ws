"""
@file env.py
@package mjcsim
@author Xinyuan Liu (liuxinyuan872@gmail.com)
@license License BSD-3-Clause
@Copyright (c) 2026, Harbin Institute of Technology.
@date 2026-01
"""
from pathlib import Path
import time
import typing

import mujoco
from mujoco import viewer as mujoco_viewer
import numpy as np
import glfw

class MujocoEnv:
    """
    This class manages a Mujoco simulation environment and provides utility 
    functions.
    """
    def __init__(self, gui: bool=True, dt=0.002):
        self.dt = dt
        # Create model and data
        xml = """
        <mujoco model="empty_world">
        <worldbody>
            <light diffuse="0.8 0.8 0.8" pos="0 0 3" />
        </worldbody>
        </mujoco>
        """.format(t=self.dt)
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)

        self.step_counter = 0
        self.model.opt.timestep = float(self.dt)

        self.objects = []
        self.robots = []
        self.robot_tracking = False
        self.robot_to_track = 0

        # viewer
        self.gui = gui
        self._viewer = None
        self._video_path = None
        self._recorder = None

    def _create_viewer(self):
        """Create a MuJoCo viewer instance without blocking execution."""
        if mujoco_viewer is not None:
            max_attempts = 5
            attempt = 0
            while attempt < max_attempts:
                try:
                    # Create a non-blocking viewer that does not enter the main loop
                    self._viewer = mujoco_viewer.launch_passive(self.model, self.data)
                    print("Viewer created successfully")
                    return self._viewer
                except Exception as e:
                    attempt += 1
                    print(f"Viewer launch attempt {attempt} failed: {e}")
                    if attempt >= max_attempts:
                        print(f"Failed to launch viewer after {max_attempts} attempts")
                        break
                    time.sleep(0.5)
        return None
    
    def update_viewer(self):
        """Update the viewer display without blocking."""
        if self._viewer is not None:
            try:
                # Synchronize simulated data to the viewer
                self._viewer.sync()
            except Exception as e:
                print(f"Viewer update failed: {e}")

    def close_viewer(self):
        """Close the viewer properly."""
        if self._viewer is not None:
            try:
                self._viewer.close()
            except Exception:
                pass
            self._viewer = None

    def add_robot(self, robot):
        """Add a RobotWrapper-like object.
        Otherwise, if robot provides model_xml and environment is empty (only default world),
        replace current model/sim with robot model.
        Returns:
            The robot object.
        """
        if hasattr(robot, "xml_filename"):
            if getattr(self, "robots", []) == [] and len(self.objects) == 0:
                model_xml = robot.xml_filename
                model_path = Path(model_xml)
                if model_path.exists():
                    self.model = mujoco.MjModel.from_xml_path(str(model_path))
                else:
                    self.model = mujoco.MjModel.from_xml_string(model_xml)
                try:
                    self.model.opt.timestep = float(self.dt)
                except Exception:
                    pass
                self.data = mujoco.MjData(self.model)
                if self.gui:
                    self._create_viewer()
                if hasattr(robot, "attach_to_sim"):
                    robot.attach_to_sim(self.model, self.data)
            self.robots.append(robot)
            return robot
        else:
            raise NotImplementedError(
                "Merging a standalone robot XML into an existing MuJoCo model "
                    "is not implemented. Please provide a single combined XML or "
                    "implement robot.attach_to_sim(sim) to perform a merge."
                )

        

    def step(self, sleep: bool = False):
        """Advance the simulation by one environment step."""
        if sleep:
            time.sleep(self.dt)
        # Determine substeps: if model.opt.timestep differs from desired dt, step multiple times.
        model_timestep = getattr(self.model.opt, "timestep", None)
        if model_timestep is None or model_timestep == 0:
            # fallback to directly calling mujoco.mj_step once
            try:
                mujoco.mj_step(self.model, self.data)
            except Exception:
                # final fallback: forward() then step()
                try:
                    mujoco.mj_forward(self.model, self.data)
                    mujoco.mj_step(self.model, self.data)
                except Exception:
                    pass
        else:
            # compute number of internal steps to reach dt
            n_substeps = max(1, int(round(self.dt / float(model_timestep))))
            for _ in range(n_substeps):
                mujoco.mj_step(self.model, self.data)
        self.step_counter += 1
        # Allow robots to compute numerical quantities 
        for robot in self.robots:
            if hasattr(robot, "compute_numerical_quantities"):
                try:
                    robot.compute_numerical_quantities(self.dt)
                except Exception:
                    # ignore robot errors to keep env robust
                    pass
        # update viewer
        if self._viewer is not None:
            self.update_viewer()

        # camera tracking
        if self.robot_tracking and self._viewer is not None:
            try:
                robot_idx = max(0, int(self.robot_to_track) - 1)
                if 0 <= robot_idx < len(self.robots):
                    robot = self.robots[robot_idx]
                    if hasattr(robot, "get_base_position"):
                        pos = robot.get_base_position()
                        # update viewer camera lookat
                        cam = getattr(self._viewer, "cam", None)
                        if cam is not None and hasattr(cam, "lookat"):
                            cam.lookat[:] = pos
            except Exception:
                pass

    def get_time_since_start(self):
        """Return simulation time in seconds since environment creation."""
        return self.step_counter * self.dt


class MujocoEnvWithGround(MujocoEnv):
    """This class provides a shortcut to construct a MuJoCo simulation 
       environment with a flat ground.
    """
    def __init__(self, dt=0.002):
        super().__init__(dt)
        print("Environment creation successful")