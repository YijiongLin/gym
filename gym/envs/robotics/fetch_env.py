import numpy as np

from gym.envs.robotics import rotations, robot_env, utils
from ipdb import set_trace

############################### 5 parts in tatol are added by ray in this file ###########################################

def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)


class FetchEnv(robot_env.RobotEnv):
    """Superclass for all Fetch environments.
    """

    def __init__(
        self, model_path, n_substeps, gripper_extra_height, block_gripper,
        has_object, target_in_the_air, target_offset, obj_range, target_range,
        distance_threshold, initial_qpos, reward_type, 
        obstacle_num = None,obstacle_added = False, env_type = None
    ):
        """Initializes a new Fetch environment.

        Args:
            model_path (string): path to the environments XML file
            n_substeps (int): number of substeps the simulation runs on every call to step
            gripper_extra_height (float): additional height above the table when positioning the gripper
            block_gripper (boolean): whether or not the gripper is blocked (i.e. not movable) or not
            has_object (boolean): whether or not the environment has an object
            target_in_the_air (boolean): whether or not the target should be in the air above the table or on the table surface
            target_offset (float or array with 3 elements): offset of the target
            obj_range (float): range of a uniform distribution for sampling initial object positions
            target_range (float): range of a uniform distribution for sampling a target
            distance_threshold (float): the threshold after which a goal is considered achieved
            initial_qpos (dict): a dictionary of joint names and values that define the initial configuration
            reward_type ('sparse' or 'dense'): the reward type, i.e. sparse or dense
        """
        self.gripper_extra_height = gripper_extra_height
        self.block_gripper = block_gripper
        self.has_object = has_object
        self.target_in_the_air = target_in_the_air
        self.target_offset = target_offset
        self.obj_range = obj_range
        self.target_range = target_range
        self.distance_threshold = distance_threshold
        self.reward_type = reward_type
        self.env_type=env_type
        # obstacle set
        self.obstacle_num = obstacle_num
        self.obstacle_added = obstacle_added   # added by ray 1
        self.obstacle_0_added_tem_pos = 0
        self.obstacle_test_mode = False
        self.obstacle_bid_0 = 0  # added by ray 2_necessary
        self.obstacle_bid_1 = 0  
        self.first_time_in_loop = True
        self.used_obstacle_bid_list = []
        super(FetchEnv, self).__init__(
            model_path=model_path, n_substeps=n_substeps, n_actions=4,
            initial_qpos=initial_qpos)

    # GoalEnv methods
    # ----------------------------

    def compute_reward(self, achieved_goal, goal, info):
        # Compute distance between goal and the achieved goal.
        d = goal_distance(achieved_goal, goal)
        if self.reward_type == 'sparse':
            return -(d > self.distance_threshold).astype(np.float32)
        else:
            return -d

    # RobotEnv methods
    # ----------------------------

    def _step_callback(self):
        if self.block_gripper:
            self.sim.data.set_joint_qpos('robot0:l_gripper_finger_joint', 0.)
            self.sim.data.set_joint_qpos('robot0:r_gripper_finger_joint', 0.)
            self.sim.forward()

    def _set_action(self, action):
        assert action.shape == (4,)
        action = action.copy()  # ensure that we don't change the action outside of this scope
        pos_ctrl, gripper_ctrl = action[:3], action[3]

        pos_ctrl *= 0.05  # limit maximum change in position
        rot_ctrl = [1., 0., 1., 0.]  # fixed rotation of the end effector, expressed as a quaternion
        gripper_ctrl = np.array([gripper_ctrl, gripper_ctrl])
        assert gripper_ctrl.shape == (2,)
        if self.block_gripper:
            gripper_ctrl = np.zeros_like(gripper_ctrl)
        action = np.concatenate([pos_ctrl, rot_ctrl, gripper_ctrl])

        # Apply action to simulation.
        utils.ctrl_set_action(self.sim, action)
        utils.mocap_set_action(self.sim, action)

    def _get_obs(self):
        # positions
        grip_pos = self.sim.data.get_site_xpos('robot0:grip')
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_site_xvelp('robot0:grip') * dt
        robot_qpos, robot_qvel = utils.robot_get_obs(self.sim)
        if self.has_object:
            object_pos = self.sim.data.get_site_xpos('object0')
            # rotations
            object_rot = rotations.mat2euler(self.sim.data.get_site_xmat('object0'))
            # velocities
            object_velp = self.sim.data.get_site_xvelp('object0') * dt
            object_velr = self.sim.data.get_site_xvelr('object0') * dt
            # gripper state
            object_rel_pos = object_pos - grip_pos
            object_velp -= grip_velp
        else:
            object_pos = object_rot = object_velp = object_velr = object_rel_pos = np.zeros(0)
        gripper_state = robot_qpos[-2:]
        gripper_vel = robot_qvel[-2:] * dt  # change to a scalar if the gripper is made symmetric

        if not self.has_object:
            achieved_goal = grip_pos.copy()
        else:
            achieved_goal = np.squeeze(object_pos.copy())
        obs = np.concatenate([
            grip_pos, object_pos.ravel(), object_rel_pos.ravel(), gripper_state, object_rot.ravel(),
            object_velp.ravel(), object_velr.ravel(), grip_velp, gripper_vel,
        ])

        # obstacle state
            
        if self.obstacle_added:     # added by ray 4
            
            if self.first_time_in_loop:
                # First time goes in to init network dimention.
                self.first_time_in_loop = False
                for i in range(self.obstacle_num):
                    obstacle_id = "obstacle_" + str(i)
                    self.used_obstacle_bid_list.append(self.model.body_name2id(obstacle_id))

            all_obstacles_states = np.array([])
            for obstacle_bid in self.used_obstacle_bid_list:
                obstacle_pos = self.model.body_pos[obstacle_bid].ravel().copy()
                obstacle_quat = self.model.body_quat[obstacle_bid].ravel().copy()
                obstacle_rot = rotations.quat2euler(obstacle_quat).ravel().copy()
                temp_all_obstacles_states = np.concatenate([obstacle_pos, obstacle_rot])
                all_obstacles_states = np.concatenate([all_obstacles_states,temp_all_obstacles_states])

            obs = np.concatenate([
                grip_pos, object_pos.ravel(), object_rel_pos.ravel(), gripper_state, object_rot.ravel(),
                object_velp.ravel(), object_velr.ravel(), grip_velp, gripper_vel, 
                all_obstacles_states.ravel(),
            ])

        return {
            'observation': obs.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self.goal.copy(),
        }

    def _viewer_setup(self):
        body_id = self.sim.model.body_name2id('robot0:gripper_link')
        lookat = self.sim.data.body_xpos[body_id]
        for idx, value in enumerate(lookat):
            self.viewer.cam.lookat[idx] = value
        self.viewer.cam.distance = 2.5
        self.viewer.cam.azimuth = 132.
        self.viewer.cam.elevation = -14.

    def _render_callback(self):
        # Visualize target.
        sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
        site_id = self.sim.model.site_name2id('target0')
        self.sim.model.site_pos[site_id] = self.goal - sites_offset[0]
        self.sim.forward()

    def _reset_sim(self):
        self.sim.set_state(self.initial_state)

        # Randomize start position of object.
        if self.has_object:

            if self.obstacle_test_mode:
                # object initial pos
                object_xpos=self.initial_gripper_xpos[:2]+np.array([0,-0.1])
                object_qpos = self.sim.data.get_joint_qpos('object0:joint')
                assert object_qpos.shape == (7,)
                object_qpos[:2] = object_xpos
                self.sim.data.set_joint_qpos('object0:joint', object_qpos)
                # obstacle initial pos
                self.model.body_pos[self.obstacle_bid_0]=self.initial_gripper_xpos[:3]
                self.sim.forward()
                return True, 0


            #the obstacle force end effector intial pos out of the middle of desk, so need to offset the object initial pos
            object_xpos = self.initial_gripper_xpos[:2]
            while np.linalg.norm(object_xpos - self.initial_gripper_xpos[:2]) < 0.1:
                object_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.obj_range, self.obj_range, size=2)
            object_qpos = self.sim.data.get_joint_qpos('object0:joint')
            assert object_qpos.shape == (7,)
            object_qpos[:2] = object_xpos
            self.sim.data.set_joint_qpos('object0:joint', object_qpos)

            if self.obstacle_added: # added by ray 5
                if self.env_type =="push":
                    for obstacle_bid in self.used_obstacle_bid_list:
                        self.model.body_pos[obstacle_bid,0] = self.np_random.uniform(low=1.15, high=1.45)  
                        self.model.body_pos[obstacle_bid,1] = self.np_random.uniform(low=0.6, high=0.9) 
                        self.model.body_pos[obstacle_bid,2] = 0.425 
                        self.model.body_quat[obstacle_bid,3] = self.np_random.uniform(low=0, high=1) 
                elif self.env_type == "pickandplace":
                    for obstacle_bid in self.used_obstacle_bid_list:
                        self.model.body_pos[obstacle_bid,0] = self.np_random.uniform(low=1.05, high=1.35)  
                        self.model.body_pos[obstacle_bid,1] = self.np_random.uniform(low=0.6, high=0.9) 
                        self.model.body_pos[obstacle_bid,2] = 0.425 + self.np_random.uniform(low=0, high=0.25)
                        # All rotation
                        self.model.body_quat[obstacle_bid,1] = self.np_random.uniform(low=0, high=1) 
                        self.model.body_quat[obstacle_bid,2] = self.np_random.uniform(low=0, high=1) 
                        self.model.body_quat[obstacle_bid,3] = self.np_random.uniform(low=0, high=1) 
                elif self.env_type=="slide":
                    for obstacle_bid in self.used_obstacle_bid_list:
                        self.model.body_pos[obstacle_bid,0] = self.np_random.uniform(low=0.95, high=1.15)  
                        self.model.body_pos[obstacle_bid,1] = self.np_random.uniform(low=0.6, high=0.9) 
                        self.model.body_pos[obstacle_bid,2] = 0.425 
                        self.model.body_quat[obstacle_bid,3] = self.np_random.uniform(low=0, high=1) 
                    
        obstacle_0_added_tem_pos = self.model.body_pos[self.used_obstacle_bid_list[0]]
        self.sim.forward()
        return True,obstacle_0_added_tem_pos

    def _sample_goal(self,obstacle_0_added_tem_pos=np.array([0,0,0])):
        if self.obstacle_test_mode:
            goal = self.initial_gripper_xpos[:3]+np.array([0,0.1,0])
            return goal.copy()

        goal = obstacle_0_added_tem_pos.copy()
        if self.has_object:
            while (goal_distance(goal, obstacle_0_added_tem_pos)<0.08):
                goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(-self.target_range, self.target_range, size=3)
                goal += self.target_offset
                goal[2] = self.height_offset
                if self.target_in_the_air and self.np_random.uniform() < 0.5:
                    goal[2] += self.np_random.uniform(0, 0.45)
        else:
            goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(-0.15, 0.15, size=3)
        return goal.copy()

    def _is_success(self, achieved_goal, desired_goal):
        d = goal_distance(achieved_goal, desired_goal)
        return (d < self.distance_threshold).astype(np.float32)

    def _env_setup(self, initial_qpos):
        for name, value in initial_qpos.items():
            self.sim.data.set_joint_qpos(name, value)
        utils.reset_mocap_welds(self.sim)
        self.sim.forward()

        # Move end effector into position.
        gripper_target = np.array([-0.498, 0.005, -0.431 + self.gripper_extra_height]) + self.sim.data.get_site_xpos('robot0:grip')
        gripper_rotation = np.array([1., 0., 1., 0.])
        self.sim.data.set_mocap_pos('robot0:mocap', gripper_target)
        self.sim.data.set_mocap_quat('robot0:mocap', gripper_rotation)
        for _ in range(10):
            self.sim.step()

        # Extract information for sampling goals.
        self.initial_gripper_xpos = self.sim.data.get_site_xpos('robot0:grip').copy()
        if self.has_object:
            self.height_offset = self.sim.data.get_site_xpos('object0')[2]

    def render(self, mode='human', width=500, height=500):
        return super(FetchEnv, self).render(mode, width, height)
