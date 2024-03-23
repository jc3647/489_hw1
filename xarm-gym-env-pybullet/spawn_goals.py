import argparse
import yaml
import pybullet_data


def load_model(p, model_path, position, orientation=(0, 0, 0, 1)):
    """
    Load a model in PyBullet.
    """
    # PyBullet adds a model to the simulation and returns its unique ID
    print(position, orientation)
    model_id = p.loadSDF(model_path)
    if len(model_id) ==0:
        return 0
    p.resetBasePositionAndOrientation(model_id[0], posObj=list(position), ornObj=list(orientation))
    return model_id


def main(p, env=1):
    parser = argparse.ArgumentParser()

    model_subdirs = ["/home/lab/489_hw1/config/box/model.sdf",
                     "/home/lab/489_hw1/config/box_2/model.sdf",
                    "/home/lab/489_hw1/config/box_3/model.sdf",
                    "/home/lab/489_hw1/config/box_4/model.sdf",
                    "/home/lab/489_hw1/config/decoys/model1.sdf",
                    "/home/lab/489_hw1/config/decoys/model2.sdf",
                    "/home/lab/489_hw1/config/decoys/model3.sdf",
                    "/home/lab/489_hw1/config/decoys/model4.sdf"
    ]

    # model_subdirs = ["/home/liam/dev_ws/src/TAMER/tamer/env/config/box/model.sdf",
    #                  "/home/liam/dev_ws/src/TAMER/tamer/env/config/box_2/model.sdf",
    #                  "/home/liam/dev_ws/src/TAMER/tamer/env/config/box_3/model.sdf",
    #                  "/home/liam/dev_ws/src/TAMER/tamer/env/config/box_4/model.sdf",
    #                  "/home/liam/dev_ws/src/TAMER/tamer/env/config/decoys/model1.sdf",
    #                  "/home/liam/dev_ws/src/TAMER/tamer/env/config/decoys/model2.sdf",
    #                  "/home/liam/dev_ws/src/TAMER/tamer/env/config/decoys/model3.sdf",
    #                  "/home/liam/dev_ws/src/TAMER/tamer/env/config/decoys/model4.sdf"]

    # Initialize PyBullet
    p.setAdditionalSearchPath(pybullet_data.getDataPath())  # Optionally set a search path for PyBullet to find URDFs
    p.loadURDF("plane.urdf")
    # Load the table (cube) beneath the robot and objects
    table_position = [0, 0, 0.5]  # Adjust the position as needed
    table_orientation = p.getQuaternionFromEuler([0, 0, 0])
    table_id = p.loadURDF("/home/lab/489_hw1/xarm-gym-env-pybullet/env/table.urdf", table_position, table_orientation, useFixedBase=True)
    # table_id = p.loadURDF("/home/liam/dev_ws/src/TAMER/tamer/env/table.urdf", table_position, table_orientation, useFixedBase=True)
    # Load configuration files
    # with open('/home/liam/dev_ws/src/TAMER/tamer/env/config/pickup_position.yaml', 'r') as file:
    with open('/home/lab/489_hw1/xarm-gym-env-pybullet/env/config/pickup_position.yaml', 'r') as file:
        pickup_position = yaml.safe_load(file)

    with open('/home/lab/489_hw1/xarm-gym-env-pybullet/env/config/goals.yaml', 'r') as file:
    # with open('/home/liam/dev_ws/src/TAMER/tamer/env/config/goals.yaml', 'r') as file:
        goals = yaml.safe_load(file)

    # Environment selection
    env_key = f'env{env}'
    if env_key not in goals:
        print(f"Environment {env} not found in goals configuration.")
        return
    decoys_seen = 0
    # Spawn entities based on the selected environment configuration
    for key, value in goals[env_key].items():
        model_name = key
        index = int(model_name.split("d")[-1])
        model_path = model_subdirs[index-1]
        position = (value[1] - 0.2, value[0] - 0.5, value[2] + 1.0)  # Adjust positions as necessary

        # Check if model file exists or adjust the path accordingly
        load_model(p, model_path, position)

    print("Finished spawning entities.")


if __name__ == "__main__":
    main()
