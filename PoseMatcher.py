import maya.cmds as cmds
import maya.api.OpenMaya as om
import numpy as np
import math
from typing import Tuple, List
import json
import os
import re

################################### Pose alignment functions Start ################################### 
# region pose alignment
def get_base_name(obj_name):
    """Get the base name of an object (excluding namespace and path)"""
    return obj_name.split(":")[-1].split("|")[-1]

def apply_global_rotation_to_vector(rotation_quaternion, vector):
    vector_mvector = om.MVector(vector[0], vector[1], vector[2])
    rotated_vector = vector_mvector.rotateBy(rotation_quaternion)
    return [rotated_vector.x, rotated_vector.y, rotated_vector.z]

def getlocalRotation(joint_name):
    return cmds.xform(joint_name, query=True, worldSpace=False, rotation=True)

def calculate_angle_3d(a, b):
    a = np.array(a)
    b = np.array(b)    
    dot_product = np.dot(a, b)    
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    cos_theta = dot_product / (norm_a * norm_b)
    theta = np.arccos(cos_theta)    
    return np.degrees(theta)

def get_joint_lra(joint, length=10.0):
    # Get the world position of the joint
    pos = cmds.xform(joint, q=True, ws=True, t=True)
    origin = om.MVector(pos)

    # 获取 jointOrient 并转换为旋转矩阵
    orient = cmds.getAttr(joint + ".jointOrient")[0]
    euler = om.MEulerRotation(
        math.radians(orient[0]),
        math.radians(orient[1]),
        math.radians(orient[2])
    )
    rot_matrix = euler.asMatrix()

    # Extract the rotation axis direction vector
    x_axis = om.MVector(rot_matrix[0], rot_matrix[1], rot_matrix[2]).normal()
    y_axis = om.MVector(rot_matrix[4], rot_matrix[5], rot_matrix[6]).normal()
    z_axis = om.MVector(rot_matrix[8], rot_matrix[9], rot_matrix[10]).normal()
    # Calculate the endpoint
    x_end = origin + x_axis * length
    y_end = origin + y_axis * length
    z_end = origin + z_axis * length
    xyzVec = []
    xyzVec.append([x_end.x-origin.x, x_end.y-origin.y, x_end.z-origin.z])
    xyzVec.append([y_end.x-origin.x, y_end.y-origin.y, y_end.z-origin.z])
    xyzVec.append([z_end.x-origin.x, z_end.y-origin.y, z_end.z-origin.z])
    return xyzVec

def get_joint_global_lra(joint, length=100.0):
    pos = cmds.xform(joint, q=True, ws=True, t=True)
    origin = om.MVector(pos)

    matrix = cmds.xform(joint, q=True, m=True, ws=True)
    m_matrix = om.MMatrix(matrix)

    joint_axis = cmds.getAttr(joint + ".rotateAxis")[0]
    order = cmds.getAttr(joint + ".rotateOrder")
    euler = om.MEulerRotation(
        math.radians(joint_axis[0]),
        math.radians(joint_axis[1]),
        math.radians(joint_axis[2]),
        order
    )
    euler_matrix = euler.asMatrix()
    rot_matrix = m_matrix*(euler_matrix.inverse())
    # Extract the rotation axis direction vector
    x_axis = om.MVector(rot_matrix[0], rot_matrix[1], rot_matrix[2]).normal()
    y_axis = om.MVector(rot_matrix[4], rot_matrix[5], rot_matrix[6]).normal()
    z_axis = om.MVector(rot_matrix[8], rot_matrix[9], rot_matrix[10]).normal()
    # Calculate the endpoint
    x_end = origin + x_axis * length
    y_end = origin + y_axis * length
    z_end = origin + z_axis * length
    xyzVec = []
    xyzVec.append([x_end.x-origin.x, x_end.y-origin.y, x_end.z-origin.z])
    xyzVec.append([y_end.x-origin.x, y_end.y-origin.y, y_end.z-origin.z])
    xyzVec.append([z_end.x-origin.x, z_end.y-origin.y, z_end.z-origin.z])
    return xyzVec

def findClose0or180(angles):
    newAngles= []
    for angle in angles:
        newAngles.append(min(angle,180-abs(angle)))
    min_value = min(newAngles)  # find the minimize value
    min_index = newAngles.index(min_value)  # find the minimize value's index
    return min_index

def getAimUpVector(joint_name, boneDir):
    lraVec = get_joint_global_lra(joint_name)
    angle_x = calculate_angle_3d(np.array(lraVec[0]),np.array(boneDir))
    angle_y = calculate_angle_3d(np.array(lraVec[1]),np.array(boneDir))
    angle_z = calculate_angle_3d(np.array(lraVec[2]),np.array(boneDir))
    axis = [[1,0,0],[0,1,0],[0,0,1]]
    angles = [angle_x, angle_y, angle_z]
    min_index = findClose0or180(angles)
    sign = 1
    if(angles[min_index]>180-abs(angles[min_index])):
        sign = -1
    
    aimVector = [sign*axis[min_index][0],sign*axis[min_index][1],sign*axis[min_index][2]]
    return [aimVector, axis[(min_index+1)%3]]

def rotate_joint_to_direction(joint_name,aimVector,target_direction):
    # create an empty object as a target
    target_object = cmds.spaceLocator(name="target")[0]
    # Position the null object at joint1's location to ensure it coincides with joint1
    cmds.delete(cmds.parentConstraint(joint_name, target_object))
    # Move the null object's position to the target direction.
    global_position = cmds.xform(joint_name, query=True, worldSpace=True, translation=True)
    cmds.move(global_position[0]+target_direction[0], global_position[1]+target_direction[1], global_position[2]+target_direction[2], target_object, absolute=True)
    # Use an aimConstraint to make joint2 point at the null object
    cmds.aimConstraint(target_object, joint_name, aimVector=aimVector)
    # if("_l" in joint_name or "l_" in joint_name):
    #     cmds.aimConstraint(target_object, joint_name, aimVector=[1,0,0],upVector=[0,1,0])
    # else:
    #     cmds.aimConstraint(target_object, joint_name, aimVector=[-1,0,0],upVector=[0,1,0])
    # Delete the temporarily created target object.
    # cmds.delete(target_object)

def normalize_vector(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

def get_joint_position(joint):
    """get joint's global position"""
    pos = cmds.xform(joint, query=True, worldSpace=True, translation=True)
    return np.array(pos)

def getGlobalRot(joint):
    matrix = cmds.xform(joint, q=True, m=True, ws=True)
    m_matrix = om.MMatrix(matrix)
    t_matrix = om.MTransformationMatrix(m_matrix)
    return t_matrix.rotation(asQuaternion=True).asMatrix()

def getJointOrient(joint):
    orient = cmds.getAttr(joint + ".jointOrient")[0]
    jo_euler = om.MEulerRotation(
        math.radians(orient[0]),
        math.radians(orient[1]),
        math.radians(orient[2])
    )
    jo_matrix = jo_euler.asMatrix()
    return jo_matrix

def getRotateAxisRot(joint):
    joint_axis = cmds.getAttr(joint + ".rotateAxis")[0]
    order = cmds.getAttr(joint + ".rotateOrder")
    ra_euler = om.MEulerRotation(
        math.radians(joint_axis[0]),
        math.radians(joint_axis[1]),
        math.radians(joint_axis[2]),
        0
    )
    ra_matrix = ra_euler.asMatrix()
    return ra_matrix

def getRotOrder(joint):
    order = cmds.getAttr(joint + ".rotateOrder")
    return order

def getAnimRotate(joint):
    animLocalRot = cmds.getAttr(joint + ".rotate")[0]
    animEuler = om.MEulerRotation(
        math.radians(animLocalRot[0]),
        math.radians(animLocalRot[1]),
        math.radians(animLocalRot[2]),
        getRotOrder(joint)
    )
    anim_matrix = animEuler.asMatrix()
    return anim_matrix

def getInherentRotWithoutParent(joint):
    jo_matrix = getJointOrient(joint)
    ra_matrix = getRotateAxisRot(joint)
    anim_matrix = getAnimRotate(joint)

    tmp_matrix=om.MMatrix(((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)))
    result_matrix=om.MMatrix(((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)))
    tmp_matrix.setToProduct(ra_matrix,anim_matrix)
    result_matrix.setToProduct(tmp_matrix,jo_matrix)
    return result_matrix

def get_parent_delta_global_rot(joint_name):
    inherent_global_rot = getInherentRotWithoutParent(joint_name)
    global_rot = getGlobalRot(joint_name)
    delta_rot = om.MMatrix(((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)))
    delta_rot.setToProduct(inherent_global_rot.inverse(), global_rot)
    return delta_rot

def getNeedRot(v1:om.MVector,v2:om.MVector):
    rot_quat = v1.rotateTo(v2)
    return rot_quat.asMatrix()

def align_parent_to_vector(child_joint, target_vector):
    """Align the parent joint's direction toward its child joint with the target vector."""    
    parent_joint = cmds.listRelatives(child_joint, parent=True, fullPath=True)[0]
    if(("twist" in parent_joint.rsplit('|', 1)[-1]) or ("Twist" in parent_joint.rsplit('|', 1)[-1])):
        parent_joint = cmds.listRelatives(parent_joint, parent=True, fullPath=True)[0]
    parent_delta_global_rot = get_parent_delta_global_rot(parent_joint)
    
    # get the position of child and parent joint
    parent_pos = get_joint_position(parent_joint)
    child_pos = get_joint_position(child_joint)
    boneVec = om.MVector(child_pos[0] - parent_pos[0], child_pos[1] - parent_pos[1],child_pos[2] - parent_pos[2])
    targetVec = om.MVector(float(target_vector[0]), float(target_vector[1]), float(target_vector[2]))
    needRot = boneVec.rotateTo(targetVec)
    needRotMat = needRot.asMatrix()

    jo_mat = getJointOrient(parent_joint)
    ra_mat = getRotateAxisRot(parent_joint)
    current_global = getGlobalRot(parent_joint)

    tmp1 = om.MMatrix(((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)))
    tmp2 = om.MMatrix(((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)))
    tmp3 = om.MMatrix(((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)))
    new_rot_mat = om.MMatrix(((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)))
    tmp1.setToProduct(jo_mat,parent_delta_global_rot)
    tmp2.setToProduct(current_global,needRotMat)
    tmp3.setToProduct(ra_mat.inverse(),tmp2)
    new_rot_mat.setToProduct(tmp3,tmp1.inverse())

    new_rot_transMat =  om.MTransformationMatrix(new_rot_mat)
    new_rot = new_rot_transMat.rotation(asQuaternion=True)
    new_rot_euler = new_rot.asEulerRotation()

    order = getRotOrder(parent_joint)
    ordered_euler = new_rot_euler.reorderIt(order)
    # Apply the rotation to the parent joint's local rotation.
    print(f"Adjust {parent_joint}")
    cmds.setAttr(parent_joint + '.rotateX', math.degrees(ordered_euler[0]))
    cmds.setAttr(parent_joint + '.rotateY', math.degrees(ordered_euler[1]))
    cmds.setAttr(parent_joint + '.rotateZ', math.degrees(ordered_euler[2]))

def find_joint_by_base_name(base_name, root_joint):
    """Find joints with the same base name under the specified root joint and return full path."""
    all_joints = cmds.listRelatives(root_joint, allDescendents=True, type="joint", fullPath=True) or []
    all_joints.append(cmds.ls(root_joint, long=True)[0])
    
    for joint in all_joints:
        short_name = get_base_name(joint)
        if short_name == base_name:
            return joint
    return None

def get_joint_direction(joint_name):
    """Get the direction vector of the joint relative to its parent joint."""
    parent = cmds.listRelatives(joint_name, parent=True, fullPath=True)
    
    if not parent:
        return [0, 0, 0]
    
    pos1 = cmds.xform(joint_name, q=True, ws=True, t=True)    
    child_trans = om.MVector(pos1)
    pos2 = cmds.xform(parent[0], q=True, ws=True, t=True)
    parent_trans = om.MVector(pos2)
    child_pos = [child_trans[0],child_trans[1],child_trans[2]]
    parent_pos = [parent_trans[0],parent_trans[1],parent_trans[2]]
    
    return [
        child_pos[0] - parent_pos[0],
        child_pos[1] - parent_pos[1],
        child_pos[2] - parent_pos[2]
    ]

def resetTwist(jointname, original_rot, rotAxis):
    rotAixsName = ".rotateX"
    rotIndex = 0
    if(rotAxis==[1,0,0]):
        rotAixsName = ".rotateX"
        rotIndex = 0
    if(rotAxis==[0,1,0]):
        rotAixsName = ".rotateY"
        rotIndex = 1
    if(rotAxis==[0,0,1]):
        rotAixsName = ".rotateZ"
        rotIndex = 2    
    cmds.setAttr(jointname + rotAixsName, original_rot[rotIndex])

def align_skeleton(joint_map, mh_root, daz_root):
    """Main function for aligning the skeleton."""
    processed_parents = set()
    error_joints = []
    joint_aim_map = {}
    joint_local_rot = {}

    for mh_joint_name, daz_joint_name in joint_map.items():
        daz_joint = find_joint_by_base_name(daz_joint_name, daz_root)
        parent = cmds.listRelatives(daz_joint, parent=True, fullPath=True)
        joint_aim_map[parent[0]] = getAimUpVector(parent[0],get_joint_direction(daz_joint))[0]
        joint_local_rot[parent[0]] = getlocalRotation(parent[0])

    

    for mh_joint_name, daz_joint_name in joint_map.items():
        # Find MetaHuman joints.
        mh_joint = find_joint_by_base_name(mh_joint_name, mh_root)
        if not mh_joint:
            cmds.warning(f"⚠️ MetaHuman joint '{mh_joint_name}' not found under root '{get_base_name(mh_root)}'")
            error_joints.append(mh_joint_name)
            continue
            
        # Find DAZ joints.
        daz_joint = find_joint_by_base_name(daz_joint_name, daz_root)        
        if not daz_joint:
            cmds.warning(f"⚠️ DAZ joint '{daz_joint_name}' not found under root '{get_base_name(daz_root)}'")
            error_joints.append(daz_joint_name)
            continue
        
        # Get the DAZ parent joint.
        daz_parents = cmds.listRelatives(daz_joint, parent=True, fullPath=True)
        if not daz_parents:
            cmds.warning(f"⚠️ DAZ joint '{get_base_name(daz_joint)}' has no parent")
            continue
        daz_parent = daz_parents[0]
        
        # Avoid processing the same parent joint multiple times.
        if daz_parent in processed_parents:
            continue
            
        # Get the direction vector.
        target_direction = get_joint_direction(mh_joint)        
        
        # Apply the rotation
        try:
            # rotate_joint_to_direction(daz_parent,joint_aim_map[daz_parent],target_direction)
            # resetTwist(daz_parent,joint_local_rot[daz_parent],joint_aim_map[daz_parent])
            align_parent_to_vector(daz_joint,target_direction)
            processed_parents.add(daz_parent)
            cmds.warning(f"✅ Successfully aligned: {get_base_name(daz_parent)}")
        except Exception as e:
            cmds.warning(f"⚠️ Failed to align {get_base_name(daz_parent)}: {str(e)}")
            error_joints.append(mh_joint_name)
    
    # summary
    if error_joints:
        cmds.warning(f"⚠️ Alignment completed with {len(error_joints)} errors.")
    else:
        cmds.warning("🎉 Alignment completed successfully!")

def save_joint_map(file_path, joint_map):
    """Save the joint mapping to a JSON file."""
    try:
        with open(file_path, 'w') as f:
            json.dump(joint_map, f, indent=4)
        return True
    except Exception as e:
        cmds.warning(f"⚠️ Failed to save joint map: {str(e)}")
        return False

def load_joint_map(file_path):
    """Load the joint mapping from a JSON file."""
    try:
        if not os.path.exists(file_path):
            return {}
            
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Handle compatibility with legacy formats.
        if isinstance(data, list):
            # legacy format: [{"mh_joint": "name", "daz_joint": "name"}]
            new_data = {}
            for item in data:
                if isinstance(item, dict) and "mh_joint" in item:
                    mh_name = item["mh_joint"]
                    new_data[mh_name] = item.get("daz_joint", "")
            return new_data
        
        # new format: {"mh_joint_name": "daz_joint_name"}
        return data
    except Exception as e:
        cmds.warning(f"⚠️ Failed to load joint map: {str(e)}")
        return {}

def auto_detect_namespace(joint_list):
    """Automatically detect and return the most common namespace."""
    namespaces = {}
    for joint in joint_list:
        base_name = get_base_name(joint)
        # Detect the namespace format. (mynamespace:joint)
        if ":" in joint:
            ns = joint.split(":")[0]
            namespaces[ns] = namespaces.get(ns, 0) + 1
        # Detect the prefix format (prefix_joint)
        else:
            prefix_match = re.search(r"^([^_]+)_", base_name)
            if prefix_match:
                prefix = prefix_match.group(1)
                namespaces[prefix] = namespaces.get(prefix, 0) + 1
    
    # Return the most common namespace or prefix.
    if namespaces:
        return max(namespaces, key=namespaces.get)
    return ""

_syncing_selection = False
def sync_selections(*_):
    """Synchronize the selection state between the two lists."""
    global _syncing_selection
    
    # Skip if synchronization is in progress.
    if _syncing_selection:
        return
    # Set the synchronization flag.
    _syncing_selection = True
    
    # Get the currently active list.
    active_list = cmds.textScrollList("mhJointsList", q=True, selectItem=True)
    if active_list:
        # MetaHuman list is selected
        selected_indices = cmds.textScrollList("mhJointsList", q=True, selectIndexedItem=True) or []
        if selected_indices:
            # Sync to DAZ list
            cmds.textScrollList("dazJointsList", e=True, deselectAll=True)
            for index in selected_indices:
                cmds.textScrollList("dazJointsList", e=True, selectIndexedItem=index)
    else:
        # The DAZ list is selected.
        selected_indices = cmds.textScrollList("dazJointsList", q=True, selectIndexedItem=True) or []
        if selected_indices:
            # Sync to MetaHuman list
            cmds.textScrollList("mhJointsList", e=True, deselectAll=True)
            for index in selected_indices:
                cmds.textScrollList("mhJointsList", e=True, selectIndexedItem=index)
    
    # Clear the sync flag
    _syncing_selection = False

def auto_detect_ns_cmd():
    """Automatically detect and set the DAZ namespace"""
    daz_root = cmds.textField("dazRootField", q=True, text=True)
    
    if not daz_root or not cmds.objExists(daz_root):
        cmds.warning("⚠️ Set the DAZ root joint first")
        return
    
    all_joints = cmds.listRelatives(daz_root, allDescendents=True, type="joint", fullPath=True) or []
    all_joints.append(daz_root)
    
    ns = auto_detect_namespace(all_joints)
    
    if ns:
        cmds.textField("dazNsField", e=True, text=ns)
        cmds.warning(f"🔍 Detected DAZ namespace/prefix: {ns}")
    else:
        cmds.warning("⚠️ No common DAZ namespace/prefix found")

def load_map_cmd():
    """load joint map file"""
    json_path = cmds.textField("jsonPathField", q=True, text=True)
    if not json_path or not os.path.exists(json_path):
        cmds.warning("⚠️ Select a valid JSON file path")
        return
    
    joint_map = load_joint_map(json_path)
    cmds.textScrollList("mhJointsList", e=True, removeAll=True)
    cmds.textScrollList("dazJointsList", e=True, removeAll=True)
    
    for mh_joint, daz_joint in joint_map.items():
        cmds.textScrollList("mhJointsList", e=True, append=mh_joint)
        cmds.textScrollList("dazJointsList", e=True, append=daz_joint)
    
    cmds.warning(f"📖 Loaded {len(joint_map)} mappings from {json_path}")

def save_map_cmd():
    """save joint map file"""
    json_path = cmds.textField("jsonPathField", q=True, text=True)
    if not json_path:
        cmds.warning("⚠️ Set a JSON file path first")
        return
    
    # get the map data
    joint_map = {}
    mh_joints = cmds.textScrollList("mhJointsList", q=True, allItems=True) or []
    daz_joints = cmds.textScrollList("dazJointsList", q=True, allItems=True) or []
    
    if len(mh_joints) != len(daz_joints):
        cmds.warning("⚠️ MetaHuman and DAZ joint lists must be the same length")
        return
    
    for i in range(len(mh_joints)):
        joint_map[mh_joints[i]] = daz_joints[i]
    
    # save to file
    if save_joint_map(json_path, joint_map):
        cmds.warning(f"💾 Saved {len(joint_map)} mappings to {json_path}")

def reset_table():
    """重置映射表到默认状态"""
    cmds.textScrollList("mhJointsList", e=True, removeAll=True)
    cmds.textScrollList("dazJointsList", e=True, removeAll=True)
    
    # add the default value
    defaults = [
        ("upperarm_l", "l_upperarm"),
        ("lowerarm_l", "l_forearm"),
        ("hand_l", "l_hand"),
        ("upperarm_r", "r_upperarm"),
        ("lowerarm_r", "r_forearm"),
        ("hand_r", "r_hand"),
    ]
    
    for mh, daz in defaults:
        cmds.textScrollList("mhJointsList", e=True, append=mh)
        cmds.textScrollList("dazJointsList", e=True, append=daz)
    
    cmds.warning("🔄 Reset to default mappings")

def add_selected_cmd():
    """Add the selected joint pairs to the mapping table"""
    selected = cmds.ls(selection=True, type="joint")
    if not selected or len(selected) < 2:
        cmds.warning("⚠️ Select 2 joints (MetaHuman joint first, then DAZ joint)")
        return
    
    # 获取基名称
    mh_joint = selected[0]
    daz_joint = selected[1]
    
    mh_base = get_base_name(mh_joint)
    daz_base = get_base_name(daz_joint)
    
    # 添加到列表
    cmds.textScrollList("mhJointsList", e=True, append=mh_base)
    cmds.textScrollList("dazJointsList", e=True, append=daz_base)

def remove_selected_cmd(*_):
    """Remove the selected joint pairs from the mapping table"""
    # get the select item
    mh_selected = cmds.textScrollList("mhJointsList", q=True, selectItem=True) or []
    daz_selected = cmds.textScrollList("dazJointsList", q=True, selectItem=True) or []
    
    # return if nothing select
    if not mh_selected and not daz_selected:
        return
    
    # get all the items
    mh_items = cmds.textScrollList("mhJointsList", q=True, allItems=True) or []
    daz_items = cmds.textScrollList("dazJointsList", q=True, allItems=True) or []
    
    # create the items which need to be removed
    items_to_remove = set()
    
    # add MetaHuman select item
    for item in mh_selected:
        if item in mh_items:
            items_to_remove.add(item)
    
    # add DAZ select item
    for item in daz_selected:
        if item in daz_items:
            items_to_remove.add(item)
    
    # return if nothing remove
    if not items_to_remove:
        return
    
    # remove MetaHuman items
    for item in items_to_remove:
        if item in mh_items:
            cmds.textScrollList("mhJointsList", e=True, removeItem=item)
    
    # remove DAZ items
    for item in items_to_remove:
        if item in daz_items:
            cmds.textScrollList("dazJointsList", e=True, removeItem=item)

def add_finger_mapping():
    """add fingers"""
    finger_types = ["thumb", "index", "middle", "ring", "pinky"]
    
    for side in ["_l", "_r"]:
        for finger in finger_types:
            for i in range(1, 4):
                mh_joint = f"{finger}_0{i}{side}"
                daz_joint = f"{side[1:]}_{finger}{i}"
                cmds.textScrollList("mhJointsList", e=True, append=mh_joint)
                cmds.textScrollList("dazJointsList", e=True, append=daz_joint)
    
    cmds.warning("🖐 Added finger mappings")

def auto_detect_cmd():
    """Auto detect joints"""
    # get root bone
    mh_root = cmds.textField("mhRootField", q=True, text=True)
    daz_root = cmds.textField("dazRootField", q=True, text=True)
    daz_ns = cmds.textField("dazNsField", q=True, text=True) or ""
    
    if not mh_root or not cmds.objExists(mh_root):
        cmds.warning("⚠️ Set the MetaHuman root joint first")
        return
        
    if not daz_root or not cmds.objExists(daz_root):
        cmds.warning("⚠️ Set the DAZ root joint first")
        return
    
    # reset the table
    cmds.textScrollList("mhJointsList", e=True, removeAll=True)
    cmds.textScrollList("dazJointsList", e=True, removeAll=True)
    
    # get all the metahuman joints
    mh_joints = cmds.listRelatives(mh_root, allDescendents=True, type="joint", fullPath=True) or []
    mh_joints.append(mh_root)
    
    # get all the daz joints
    daz_joints = cmds.listRelatives(daz_root, allDescendents=True, type="joint", fullPath=True) or []
    daz_joints.append(daz_root)
    
    matched_count = 0
    
    # Create Mapping - Based on Base Name Matching
    for mh_joint in mh_joints:
        mh_base = get_base_name(mh_joint)
        
        # Search for matches in DAZ joints
        for daz_joint in daz_joints:
            daz_base = get_base_name(daz_joint)
            
            # 应用命名空间
            if daz_ns and daz_base.startswith(daz_ns):
                clean_name = daz_base[len(daz_ns):]
            else:
                clean_name = daz_base
            
            if mh_base == clean_name:
                cmds.textScrollList("mhJointsList", e=True, append=mh_base)
                cmds.textScrollList("dazJointsList", e=True, append=daz_base)
                matched_count += 1
                break
    
    cmds.warning(f"🔍 Auto-detected {matched_count} joint mappings")

def execute_alignment_cmd():
    """Execute Skeleton Alignment Operation"""
    # Get Root Joint
    mh_root = cmds.textField("mhRootField", q=True, text=True)
    daz_root = cmds.textField("dazRootField", q=True, text=True)
    
    if not mh_root or not cmds.objExists(mh_root):
        cmds.warning("⚠️ Set the MetaHuman root joint first")
        return
    
    if not daz_root or not cmds.objExists(daz_root):
        cmds.warning("⚠️ Set the DAZ root joint first")
        return
    
    # Build Joint Mapping
    joint_map = {}
    mh_joints = cmds.textScrollList("mhJointsList", q=True, allItems=True) or []
    daz_joints = cmds.textScrollList("dazJointsList", q=True, allItems=True) or []
    
    if not mh_joints or not daz_joints:
        cmds.warning("⚠️ No joint mappings found")
        return
    
    if len(mh_joints) != len(daz_joints):
        cmds.warning("⚠️ MetaHuman and DAZ joint lists must be the same length")
        return
    
    for i in range(len(mh_joints)):
        joint_map[mh_joints[i]] = daz_joints[i]
    
    # Automatically save the current mapping (optional)
    json_path = cmds.textField("jsonPathField", q=True, text=True)
    if json_path:
        if save_joint_map(json_path, joint_map):
            cmds.warning(f"💾 Map saved to {json_path}")
    
    # Perform alignment
    cmds.warning("🔧 Starting skeleton alignment...")
    cmds.refresh(suspend=True)
    align_skeleton(joint_map, mh_root, daz_root)
    cmds.refresh(suspend=False)
    cmds.refresh()
    cmds.warning("✅ Done!")

#endregion
################################### Pose alignment functions End ################################### 

################################### Merge/Split meshes functions Start ################################### 
#region merge and split meshes
# --------------------------------------------------
# 1. 抽取单个模型的顶点 / UV / vt
# --------------------------------------------------
def get_mesh_arrays_fast(transform_path: str):
    """
    返回:
        vertices : np.ndarray (N,3)  顶点世界坐标（顶点级）
        uvs      : np.ndarray (M,2)  UV 坐标（顶点级）
        vt       : np.ndarray (K,4)  face-vertex 映射
                     [face_id, vert_id, uv_id, vn_id]
    """
    sel = om.MSelectionList()
    sel.add(transform_path)
    dag = sel.getDagPath(0)
    dag.extendToShape()
    if dag.apiType() != om.MFn.kMesh:
        raise RuntimeError(f"{transform_path} 不是多边形网格。")

    mesh = om.MFnMesh(dag)

    # 1) 顶点
    points = mesh.getPoints(om.MSpace.kWorld)
    vertices = np.array([[p.x, p.y, p.z] for p in points], dtype=np.float64)

    # 2) UV
    u_array, v_array = mesh.getUVs()
    uvs = np.column_stack((u_array, v_array)).astype(np.float64)

    # 3) VN
    normals = np.array(mesh.getVertexNormals(angleWeighted=True, space=om.MSpace.kWorld))

    # 3) 获取 face-vertex 级索引
    face_counts, vert_ids = mesh.getVertices()
    _, uv_ids   = mesh.getAssignedUVs()
    _, norm_ids = mesh.getNormalIds()

    # face_id 列
    face_ids = np.repeat(np.arange(len(face_counts), dtype=np.int32), face_counts)

    # 组装 4 列：face_id, vert_id, uv_id, vn_id
    vt = np.column_stack((face_ids, vert_ids, uv_ids, norm_ids)).astype(np.int32)

    return vertices, uvs, normals, vt


# --------------------------------------------------
# 2. 获取所有选中模型
# --------------------------------------------------
def get_selected_meshes_numpy(mode="merge") -> List[Tuple[str, np.ndarray, np.ndarray, np.ndarray]]:
    sel_list = om.MGlobal.getActiveSelectionList()
    # if sel_list.isEmpty():
    #     raise RuntimeError("请先至少选中一个多边形模型。")
    if(mode=="merge"):
        if sel_list.length() != 2:
            cmds.confirmDialog(title='Error', message='Please select only two meshes!', icon='critical')
            raise RuntimeError("Please select only two meshes!")
    elif(mode=="split"):
        if sel_list.length() != 1:
            cmds.confirmDialog(title='Error', message='Please select only one mesh!', icon='critical')
            raise RuntimeError("Please select only onetwo mesh!")

    results = []
    for i in range(sel_list.length()):
        dag = sel_list.getDagPath(i)
        dag.extendToShape()
        if dag.apiType() != om.MFn.kMesh:
            continue
        name = sel_list.getDagPath(i).fullPathName()
        v, u, vn, vt = get_mesh_arrays_fast(name)
        results.append((name, v, u, vn, vt))

    if not results:
        raise RuntimeError("选中的节点里没有任何多边形网格。")

    return results


# --------------------------------------------------
# 3. 重合顶点检测
# --------------------------------------------------
def find_overlaps(a: np.ndarray,
                  b: np.ndarray,
                  decimals: int = 3) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    返回:
        coords : (L,3) 重合坐标
        idx_a  : (L,)  在 a 中的行索引
        idx_b  : (L,)  在 b 中的行索引
    """
    a = np.asarray(a, dtype=float).reshape(-1, 3)
    b = np.asarray(b, dtype=float).reshape(-1, 3)

    a_r = np.round(a, decimals)
    b_r = np.round(b, decimals)

    a_view = a_r.view([('', a.dtype)] * 3)
    b_view = b_r.view([('', b.dtype)] * 3)

    _, idx_a, idx_b = np.intersect1d(
        a_view, b_view, assume_unique=False, return_indices=True
    )
    return a[idx_a], idx_a, idx_b


# --------------------------------------------------
# 4. Maya 场景内选中指定顶点
# --------------------------------------------------
def select_vertices(mesh_name: str, indices: np.ndarray):
    vtx_strings = [f"{mesh_name}.vtx[{i}]" for i in indices]
    if vtx_strings:
        cmds.select(vtx_strings, replace=True)
    else:
        cmds.select(clear=True)


# --------------------------------------------------
# 5. 一键检测两个选中模型的重合顶点
# --------------------------------------------------
def detect_and_select_overlaps(name1,name2,v1,v2,decimals: int = 3):
    """
    选中两个模型 → 检测重合顶点 → 在第一个模型上选中它们
    """
    _, idx1, idx2 = find_overlaps(v1, v2, decimals=decimals)
    print(f"共找到 {len(idx1)} 个重合顶点，已在 {name1} 中选中。")
    return name1, name2, idx1, idx2

# --------------------------------------------------
# 5. 合并模型
# --------------------------------------------------
# 将顶点和面写入obj
def writeWithColor(f,v,uv,vn,IdxsWithColor,name):
    fp = open(name,'w')
    for i in range(v.shape[0]):
        if(IdxsWithColor!=[]):
            if i in IdxsWithColor:
                fp.write("v {0} {1} {2} 1 0 0\n".format(v[i,0],v[i,1],v[i,2]))
            else:
                fp.write("v {0} {1} {2} 1 1 1\n".format(v[i,0],v[i,1],v[i,2]))
        else:
            fp.write("v {0} {1} {2}\n".format(v[i,0],v[i,1],v[i,2]))
    for i in range(uv.shape[0]):        
        fp.write("vt {0} {1}\n".format(uv[i,0],uv[i,1]))
    for i in range(vn.shape[0]):        
        fp.write("vn {0} {1} {2}\n".format(vn[i,0],vn[i,1],vn[i,2]))
    curr_fid = -1
    for i in range(f.shape[0]):
        if(f[i,0]==curr_fid):
            fp.write(" {0}/{1}/{2}".format(f[i,1]+1,f[i,2]+1,f[i,3]+1))
        else:
            fp.write("\nf {0}/{1}/{2}".format(f[i,1]+1,f[i,2]+1,f[i,3]+1))
            curr_fid = f[i,0]
    # for i in range(len(f)):
    #     for j in range(len(f[i])):
    #         if(j==0):
    #             fp.write("f {0} ".format(f[i][j]+1))
    #         else:
    #             fp.write("{0} ".format(f[i][j]+1))
    #         if(j==len(f[i])-1):
    #             fp.write("\n")       
    fp.close()

def ProcessVertices(mesh1_v,mesh2_v,overlap1,overlap2):
    cmds.progressWindow(
        title='vertices...',
        progress=0,
        status='0 / %d' % mesh2_v.shape[0],
        isInterruptable=True
    )
    cmds.refresh()
    total_v = mesh1_v.copy()
    map_v = np.array([], dtype=np.int32)
    for i in range(mesh2_v.shape[0]):
        if cmds.progressWindow(query=True, isCancelled=True):
            cmds.warning('用户取消！')
            break

        if(i not in overlap2):
            total_v = np.vstack([total_v,mesh2_v[i,...]])
            map_v = np.hstack([map_v, np.array([total_v.shape[0]-1], dtype=np.int32)])
        else:
            indices = np.where(i == overlap2)[0]
            map_v = np.hstack([map_v,overlap1[indices[0]]])

        # ---- 更新进度 ----
        percent = float(i + 1) / mesh2_v.shape[0] * 100
        cmds.progressWindow(edit=True,
                          progress=percent,
                          status='%d / %d' % (i + 1, mesh2_v.shape[0]))
    cmds.progressWindow(endProgress=True)
    return total_v, map_v

def is_overlap(a, b):
    """
    a, b: (x_min, y_min, x_max, y_max)
    边贴合不算重叠
    返回 True 表示两矩形严格重叠
    """
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    # 任一方向不重叠 → 整体不重叠
    no_overlap_x = ax2 <= bx1 or bx2 <= ax1
    no_overlap_y = ay2 <= by1 or by2 <= ay1

    # 反推：两个方向都不“不重叠”才重叠
    return not (no_overlap_x or no_overlap_y)

def ProcessUV(mesh1_uv,mesh2_uv):
    merged_uv = mesh1_uv.copy()
    uv1_range = [math.floor(min(mesh1_uv[...,0])), math.floor(min(mesh1_uv[...,1])),math.ceil(max(mesh1_uv[...,0])), math.ceil(max(mesh1_uv[...,1]))]
    uv2_range = [math.floor(min(mesh2_uv[...,0])), math.floor(min(mesh2_uv[...,1])),math.ceil(max(mesh2_uv[...,0])), math.ceil(max(mesh2_uv[...,1]))]
    if(is_overlap(uv1_range,uv2_range)):
        if(uv1_range[0]<uv2_range[0]):
            mesh2_uv[...,0] += uv1_range[3]-uv2_range[0]
    merged_uv = np.vstack([mesh1_uv,mesh2_uv])
    map_uv = np.arange(mesh2_uv.shape[0]) + mesh1_uv.shape[0]
    return merged_uv, map_uv

def ProcessVN(mesh1_vn,mesh2_vn):
    return np.vstack([mesh1_vn,mesh2_vn]), np.arange(mesh2_vn.shape[0])+mesh1_vn.shape[0]

def ProcessFaces(mesh1_f,mesh2_f,map_v,map_uv,map_vn):
    cmds.progressWindow(
        title='faces...',
        progress=0,
        status='0 / %d' % mesh2_f.shape[0],
        isInterruptable=True
    )
    cmds.refresh()
    merged_f = mesh1_f.copy()
    for i in range(mesh2_f.shape[0]):
        # ---- 每帧检查是否用户点了取消 ----
        if cmds.progressWindow(query=True, isCancelled=True):
            cmds.warning('用户取消！')
            break

        cur_face_id = mesh2_f[i,0] + mesh1_f.shape[0]
        cur_vert_id = map_v[mesh2_f[i,1]]
        cur_uv_id = map_uv[mesh2_f[i,2]]
        cur_norm_id = map_vn[mesh2_f[i,3]]
        merged_f = np.vstack([merged_f,np.column_stack((cur_face_id, cur_vert_id, cur_uv_id, cur_norm_id)).astype(np.int32)])

        # ---- 更新进度 ----
        percent = float(i + 1) / mesh2_f.shape[0] * 100
        cmds.progressWindow(edit=True,
                          progress=percent,
                          status='%d / %d' % (i + 1, mesh2_f.shape[0]))
    cmds.progressWindow(endProgress=True)
    return merged_f

def MergeMeshes(folder_path,mesh1_f,mesh2_f,mesh1_v,mesh2_v,mesh1_uv,mesh2_uv,mesh1_vn,mesh2_vn,overlap1,overlap2):
    merged_v, map_v = ProcessVertices(mesh1_v,mesh2_v,overlap1,overlap2)
    merged_uv, map_uv = ProcessUV(mesh1_uv,mesh2_uv)
    merged_vn, map_vn = ProcessVN(mesh1_vn,mesh2_vn)
    merged_f = ProcessFaces(mesh1_f,mesh2_f,map_v,map_uv,map_vn)

    json_data = {}
    json_data["num_vertices_mesh1"] = mesh1_v.shape[0]
    json_data["num_vertices_mesh2"] = mesh2_v.shape[0]
    # json_data["overlap1"] = overlap1.tolist()
    # json_data["overlap2"] = overlap2.tolist()
    json_data["map_v"] = map_v.tolist()
    json_data["uv_mesh1"] = mesh1_uv.tolist()
    json_data["uv_mesh2"] = mesh2_uv.tolist()
    json_data["faces_mesh1"] = mesh1_f.tolist()
    json_data["faces_mesh2"] = mesh2_f.tolist()
    # 写入 JSON 文件
    with open(folder_path + "_MergeInfo.json", "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=4)

    return merged_f, merged_v,merged_uv,merged_vn

def SplitMeshes(mesh_name,jsonPath, merged_v, merged_vn):
    with open(jsonPath, "r", encoding="utf-8") as f:
        json_data = json.load(f)          # data 就是原来的 dict
    
    mesh1_v_num = np.array(json_data["num_vertices_mesh1"])
    mesh2_v_num = np.array(json_data["num_vertices_mesh2"])
    # overlap1 = json_data["overlap1"]
    # overlap2 = json_data["overlap2"]
    map_v = json_data["map_v"]
    mesh1_uv = np.array(json_data["uv_mesh1"])
    mesh2_uv = np.array(json_data["uv_mesh2"])
    mesh1_f = np.array(json_data["faces_mesh1"])
    mesh2_f = np.array(json_data["faces_mesh2"])

    # get mesh1
    split_v_mesh1 = np.array(merged_v[0:mesh1_v_num,...])
    split_vn_mesh1 = np.array(merged_vn[0:mesh1_v_num,...])
    writeWithColor(mesh1_f,split_v_mesh1,mesh1_uv,split_vn_mesh1,[],os.path.dirname(jsonPath)+"/"+mesh_name.replace(":","_")+"_Part1.obj")
    build_mesh_from_numpy(split_v_mesh1,mesh1_uv,mesh1_f,split_vn_mesh1,mesh_name+"_Part1")

    # get mesh2
    split_vn_mesh2 = merged_vn[map_v,...]
    split_v_mesh2 = merged_v[map_v,...]
    writeWithColor(mesh2_f,split_v_mesh2,mesh2_uv,split_vn_mesh2,[],os.path.dirname(jsonPath)+"/"+mesh_name.replace(":","_")+"_Part2.obj")
    build_mesh_from_numpy(split_v_mesh2,mesh2_uv,mesh2_f,split_vn_mesh2,mesh_name+"_Part2")

def assign_lambert_to_mesh(mesh_transform, color=(0.5, 0.5, 0.5)):
    """给 transform 指定一个 Lambert 材质"""
    shape = cmds.listRelatives(mesh_transform, s=True, type='mesh')[0]
    sg = cmds.sets(renderable=True, noSurfaceShader=True, empty=True, name="lambert1SG")
    lamb = cmds.shadingNode("lambert", asShader=True, name="lambert1")
    cmds.setAttr(lamb + ".color", *color, type="double3")
    cmds.connectAttr(lamb + ".outColor", sg + ".surfaceShader", force=True)
    cmds.sets(shape, edit=True, forceElement=sg)

def build_mesh_from_numpy(v, uv, vt, vn, name='newMesh'):
    """
    v   : (N,3)  float   顶点
    uv  : (M,2)  float   顶点级 UV
    vt  : (K,4)  int     每行 [face_id, vert_id, uv_id, vn_id]
                         同一 face_id 连续出现，长度=该面顶点数
    vn  : (P,3)  float   顶点级法线
    """
    # ---------- 1. 顶点 ----------
    points = om.MPointArray([om.MPoint(*p) for p in v])

    # ---------- 2. face_counts & 所有 per-face-vertex 索引 ----------
    face_ids = vt[:, 0]
    unique, counts = np.unique(face_ids, return_counts=True)
    face_counts   = om.MIntArray(counts.tolist())
    face_connects = om.MIntArray(vt[:, 1].tolist())
    uv_ids        = om.MIntArray(vt[:, 2].tolist())
    norm_coords   = [om.MVector(*vn[i]) for i in vt[:, 3]]
    norm_arr      = om.MVectorArray(norm_coords)  # 预先生成法线数组

    # ---------- 3. 创建 mesh ----------
    mfn = om.MFnMesh()
    mobj = mfn.create(points, face_counts, face_connects)

    # ---------- 4. UV ----------
    u_arr = om.MFloatArray(uv[:, 0].tolist())
    v_arr = om.MFloatArray(uv[:, 1].tolist())
    mfn.setUVs(u_arr, v_arr)
    mfn.assignUVs(face_counts, uv_ids)

    # ---------- 5. 法线 (关键新增部分) ----------
    # norm_coords = [om.MVector(*vn[i]) for i in vt[:, 3]]  # 每行第4列是法线索引
    # norm_arr    = om.MVectorArray(norm_coords)
    # face_ids   = om.MIntArray(vt[:, 0].tolist())  # 每行第1列：面 ID
    # vertex_ids = om.MIntArray(vt[:, 1].tolist())  # 每行第2列：顶点 ID
    # mfn.setFaceVertexNormals(norm_arr, face_ids, vertex_ids, space=om.MSpace.kObject)
    
    # ---------- 6. 改名 + 默认材质 ----------
    dag   = om.MDagPath.getAPathTo(mobj)
    final = cmds.rename(dag.fullPathName(), name)
    assign_lambert_to_mesh(final)
    cmds.select(final)
    return final

# -------------------------------------------------
# 写文件（导出）
# -------------------------------------------------
# 全局变量，存一下上次选的路径，可省略
g_lastPath = cmds.workspace(q=True, rootDirectory=True)
def MergeMeshes_cmd():
    global g_lastPath

    meshes = get_selected_meshes_numpy("merge")
    filePath = cmds.fileDialog2(fileMode=0,  # 0 = 保存路径
                            startingDirectory=g_lastPath,
                            fileFilter="obj Files (*.obj);;All Files (*)")
    if filePath:
        g_lastPath = os.path.dirname(filePath[0])

        name1, v1, uv1,vn1, vt1 = meshes[0]
        name2, v2, uv2,vn2, vt2 = meshes[1]

        name1, name2, overlap_idx1, overlap_idx2 = detect_and_select_overlaps(name1,name2,v1,v2,decimals=3)
        select_vertices(name1, overlap_idx1)
        # select_vertices(name2, idx2)
        merged_f, merged_v,merged_uv,merged_vn = MergeMeshes(os.path.splitext(filePath[0])[0],vt1,vt2,v1,v2,uv1,uv2,vn1,vn2,overlap_idx1,overlap_idx2)
        filename = os.path.splitext(os.path.basename(filePath[0]))[0]
        build_mesh_from_numpy(merged_v,merged_uv,merged_f, merged_vn, name=filename)
        writeWithColor(merged_f,merged_v,merged_uv,merged_vn, [], filePath[0])

# -------------------------------------------------
# 弹窗：让用户选 txt，然后决定读还是写
# -------------------------------------------------
def SplitMeshes_cmd():
    global g_lastPath
    path = cmds.fileDialog2(fileMode=1,  # 1 = 单选已存在文件
                            startingDirectory=g_lastPath,
                            fileFilter="json Files (*.json);;All Files (*)")
    if path:
        g_lastPath = os.path.dirname(path[0])
        meshes = get_selected_meshes_numpy("split")
        meshName, merged_v, merged_uv,merged_vn, merged_vt = meshes[0]
        SplitMeshes(meshName.rsplit('|', 1)[-1],path[0], merged_v, merged_vn)
        
#endregion
################################### Merge/Split meshes functions End ################################### 
def create_alignment_ui():
    """Create the UI window"""
    win_name = "skeletonAlignmentUI"
    if cmds.window(win_name, exists=True):
        cmds.deleteUI(win_name)
    
    # Create main window - Fixed size to ensure all content is visible
    cmds.window(win_name, title="Skeleton Alignment Tool", width=520, height=700, sizeable=False)
    
    # Main layout - Vertical column layout
    main_layout = cmds.columnLayout(
        adjustableColumn=True,
        columnAttach=('both', 5),
        rowSpacing=10,
        height=690
    )
    
    # ================ set skeleton ================
    skeleton_frame = cmds.frameLayout(
        label="Skeleton Settings",
        collapsable=True,
        borderStyle="etchedIn",
        marginWidth=5,
        marginHeight=5
    )
    
    # MetaHuman root joint
    cmds.gridLayout(numberOfColumns=3, cellWidth=170)
    cmds.text(label="MetaHuman Root:", align="right")
    mh_root_field = cmds.textField("mhRootField", width=120)
    cmds.button(
        label="Get Selected", 
        annotation="Select MetaHuman root joint and click",
        command=lambda *_: cmds.textField(
            mh_root_field, 
            e=True, 
            text=cmds.ls(selection=True)[0] if cmds.ls(selection=True) else ""
        )
    )
    cmds.setParent("..")
    
    # DAZ root joint
    cmds.gridLayout(numberOfColumns=3, cellWidth=170)
    cmds.text(label="DAZ Root Joint:", align="right")
    daz_root_field = cmds.textField("dazRootField", width=120)
    cmds.button(
        label="Get Selected", 
        annotation="Select DAZ root joint and click",
        command=lambda *_: cmds.textField(
            daz_root_field, 
            e=True, 
            text=cmds.ls(selection=True)[0] if cmds.ls(selection=True) else ""
        )
    )
    cmds.setParent("..")
    
    # DAZ namepsace
    cmds.gridLayout(numberOfColumns=3, cellWidth=170)
    cmds.text(label="DAZ Namespace:", align="right")
    daz_ns_field = cmds.textField("dazNsField", width=120)
    cmds.button(
        label="Auto Detect", 
        command=lambda *_: auto_detect_ns_cmd()
    )
    cmds.setParent("..")
    cmds.setParent("..")
    
    # ================ file ================
    file_frame = cmds.frameLayout(
        label="File Settings",
        collapsable=True,
        borderStyle="etchedIn",
        marginWidth=5,
        marginHeight=5
    )
    
    # JSON file
    cmds.gridLayout(numberOfColumns=3, cellWidth=170)
    cmds.text(label="JSON File Path:", align="right")
    json_path_field = cmds.textField("jsonPathField", width=120)
    cmds.button(
        label="Browse...", 
        command=lambda *_: cmds.textField(
            json_path_field, 
            e=True, 
            text=(cmds.fileDialog2(
                fileMode=1,
                fileFilter="JSON Files (*.json)"
            ) or [""])[0]
        )
    )
    cmds.setParent("..")
    
    # file button
    cmds.gridLayout(numberOfColumns=3, cellWidth=170)
    cmds.button(label="Load Map", command=lambda *_: load_map_cmd(), height=30)
    cmds.button(label="Save Map", command=lambda *_: save_map_cmd(), height=30)
    cmds.button(label="Reset Map", command=reset_table, height=30)
    cmds.setParent("..")
    cmds.setParent("..")
    
    # ================ joint map ================
    map_frame = cmds.frameLayout(
        label="Joint Mapping",
        collapsable=False,
        borderStyle="etchedIn",
        marginWidth=5,
        marginHeight=5
    )
    
    # row title
    cmds.gridLayout(numberOfColumns=2, cellWidth=250)
    cmds.text(label="MetaHuman Joints", align="center", height=20)
    cmds.text(label="Your Model Joints", align="center", height=20)
    cmds.setParent("..")
    
    # List area - Ensures sufficient height
    cmds.rowLayout(numberOfColumns=2, height=200)
    mh_list = cmds.textScrollList("mhJointsList", allowMultiSelection=True, height=200, width=250, selectCommand=sync_selections)
    daz_list = cmds.textScrollList("dazJointsList", allowMultiSelection=True, height=200, width=250, selectCommand=sync_selections)
    cmds.setParent("..")
    
    # Controll button
    cmds.gridLayout(numberOfColumns=4, cellWidth=130, cellHeight=30)
    cmds.button(label="Add Selected", command=lambda *_:add_selected_cmd())
    cmds.button(label="Remove Selected", command=lambda *_:remove_selected_cmd())
    cmds.button(label="Auto-Detect All", command=lambda *_:auto_detect_cmd())
    cmds.button(label="Add Fingers", command=lambda *_:add_finger_mapping())
    cmds.setParent("..")
    cmds.setParent("..")
    
    # ================ Execute ================
    cmds.separator(height=10)
    cmds.button(
        label="ALIGN SKELETONS", 
        height=50, 
        backgroundColor=[0.3, 0.6, 0.8],
        command=lambda *_: execute_alignment_cmd()
    )

    # ================ Mesh tool ================    
    mesh_frame = cmds.frameLayout(
        label="Mesh utils",
        collapsable=True,
        borderStyle="etchedIn",
        marginWidth=5,
        marginHeight=5
    )
    cmds.gridLayout(numberOfColumns=2, cellWidth=170)
    cmds.button(label="Merge to one mesh", command=lambda *_: MergeMeshes_cmd(), height=30)
    cmds.button(label="Restore to two meshes", command=lambda *_: SplitMeshes_cmd(), height=30)
    cmds.setParent("..")
    
    # initialize
    reset_table()
    
    cmds.showWindow(win_name)

# Start the UI
if __name__ == "__main__":
    create_alignment_ui()