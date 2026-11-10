from pathlib import Path
import opensim as osim

PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "results" / "opensim"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_FILE = OUTPUT_DIR / "prosthetic_leg_2dof.osim"

# Initial geometric parameters, in meters.
# Replace with verified CAD values before validation.
SHANK_LENGTH = 0.4064
FOOT_LENGTH = 0.25

# Approximate masses from your previous CAD data.
LB_TO_KG = 0.45359237
SHANK_MASS = 1.21 * LB_TO_KG
FOOT_MASS = 5.09 * LB_TO_KG


def create_body(name, mass, com, inertia):
    return osim.Body(
        name,
        mass,
        osim.Vec3(*com),
        osim.Inertia(*inertia)
    )


def build_model():
    model = osim.Model()
    model.setName("ProstheticLeg2DOF")

    # Grounded thigh / knee attachment.
    ground = model.getGround()

    # Provisional principal inertias in kg*m^2.
    # These are placeholders, not finalized CAD tensors.
    shank = create_body(
        "shank",
        SHANK_MASS,
        (0, -SHANK_LENGTH / 2, 0),
        (0.01, 0.01, 0.01)
    )

    foot = create_body(
        "foot",
        FOOT_MASS,
        (FOOT_LENGTH / 2, 0, 0),
        (0.02, 0.02, 0.02)
    )

    model.addBody(shank)
    model.addBody(foot)

    # PinJoint rotates about its local Z axis.
    # Both joints lie in the sagittal XY plane.
    knee = osim.PinJoint(
        "knee_joint",
        ground,
        osim.Vec3(0, 0, 0),
        osim.Vec3(0, 0, 0),
        shank,
        osim.Vec3(0, 0, 0),
        osim.Vec3(0, 0, 0)
    )

    ankle = osim.PinJoint(
        "ankle_joint",
        shank,
        osim.Vec3(0, -SHANK_LENGTH, 0),
        osim.Vec3(0, 0, 0),
        foot,
        osim.Vec3(0, 0, 0),
        osim.Vec3(0, 0, 0)
    )

    model.addJoint(knee)
    model.addJoint(ankle)

    knee_coordinate = knee.updCoordinate()
    knee_coordinate.setName("knee_angle")
    knee_coordinate.setDefaultValue(0.0)

    ankle_coordinate = ankle.updCoordinate()
    ankle_coordinate.setName("ankle_angle")
    ankle_coordinate.setDefaultValue(0.0)

    # Actuators apply generalized joint torques.
    knee_actuator = osim.CoordinateActuator()
    knee_actuator.setName("knee_motor")
    knee_actuator.setCoordinate(knee_coordinate)
    knee_actuator.setOptimalForce(1.0)
    knee_actuator.setMinControl(-1000.0)
    knee_actuator.setMaxControl(1000.0)

    ankle_actuator = osim.CoordinateActuator()
    ankle_actuator.setName("ankle_motor")
    ankle_actuator.setCoordinate(ankle_coordinate)
    ankle_actuator.setOptimalForce(1.0)
    ankle_actuator.setMinControl(-1000.0)
    ankle_actuator.setMaxControl(1000.0)

    model.addForce(knee_actuator)
    model.addForce(ankle_actuator)

    model.setGravity(osim.Vec3(0, -9.81, 0))

    model.finalizeConnections()
    state = model.initSystem()

    print("\n--- OPENSIM MODEL ---")
    print("Model:", model.getName())
    print(
        "Coordinates:",
        model.getCoordinateSet().getSize()
    )
    print(
        "Bodies:",
        model.getBodySet().getSize()
    )
    print(
        "Actuators:",
        model.getActuators().getSize()
    )

    for i in range(model.getCoordinateSet().getSize()):
        coordinate = model.getCoordinateSet().get(i)
        print("Coordinate:", coordinate.getName())

    model.printToXML(str(MODEL_FILE))

    print("\nModel saved:", MODEL_FILE)
    print("Initialization successful.")

    return model, state


if __name__ == "__main__":
    build_model()
