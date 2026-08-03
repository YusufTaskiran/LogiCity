import cv2
import numpy as np
import pickle as pkl
from PIL import Image, ImageDraw, ImageFont
import torch
import os
from tqdm import trange
from scipy.ndimage import label
from logicity.core.config import *
import argparse


def _to_numpy_world(world):
    if hasattr(world, "detach"):
        return world.detach().cpu().numpy()
    if hasattr(world, "cpu"):
        return world.cpu().numpy()
    return np.asarray(world)


def _normalize_agent_concepts(concepts):
    if isinstance(concepts, dict):
        return concepts
    if isinstance(concepts, list):
        return {str(concept): 1.0 for concept in concepts}
    return {}


def _normalize_agents_from_trace(agent_list):
    agents = {}
    if not isinstance(agent_list, list):
        return agents

    for item in agent_list:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        if not name:
            continue
        agents[name] = {
            "layer_id": item.get("layer_id"),
            "concepts": _normalize_agent_concepts(item.get("concepts", {})),
            "is_rl_agent": name.startswith("Car_"),
            "goal": item.get("goal"),
            "start": item.get("start"),
            "reach_goal": item.get("reach_goal", False),
            "priority": item.get("priority"),
        }
    return agents


def _load_rollout_like_data(pkl_path):
    with open(pkl_path, "rb") as f:
        data = pkl.load(f)

    # Legacy cached-world rollout format.
    if isinstance(data, dict) and "Time_Obs" in data and "Static Info" in data:
        obs = data["Time_Obs"]
        agents = data["Static Info"]["Agents"]
        return obs, agents

    # Debug trace format saved from main.py.
    if isinstance(data, dict) and "steps" in data:
        steps = data.get("steps", [])
        if not steps:
            raise RuntimeError("Debug trace contains no steps.")

        obs = {}
        first_step = steps[0]
        obs[int(first_step["step"])] = {"World": _to_numpy_world(first_step["pre_world"])}
        for step_data in steps:
            obs[int(step_data["step"]) + 1] = {"World": _to_numpy_world(step_data["post_world"])}

        agents = _normalize_agents_from_trace(first_step.get("pre_agents", []))
        return obs, agents

    raise RuntimeError("Unsupported PKL format for visualization: {}".format(type(data)))

IMAGE_BASE_PATH = "./imgs"
SCALE = 8
CAR_RENDER_LENGTH_SCALE = 0.85

PATH_DICT = {
    "Car": [os.path.join(IMAGE_BASE_PATH, "car{}.png").format(i) for i in range(1, 2)],
    "Ambulance": os.path.join(IMAGE_BASE_PATH, "car_ambulance.png"),
    "Bus": os.path.join(IMAGE_BASE_PATH, "car_bus.png"),
    "Tiro": os.path.join(IMAGE_BASE_PATH, "car_tiro.png"),
    "Police": os.path.join(IMAGE_BASE_PATH, "car_police.png"),
    "Reckless": os.path.join(IMAGE_BASE_PATH, "car_reckless.png"),
    "Pedestrian": [os.path.join(IMAGE_BASE_PATH, "pedestrian{}.png").format(i) for i in range(1, 3)],
    "Pedestrian_old": os.path.join(IMAGE_BASE_PATH, "pedestrian_old.png"),
    "Pedestrian_young": os.path.join(IMAGE_BASE_PATH, "pedestrian_young.png"),
    "Walking Street": os.path.join(IMAGE_BASE_PATH, "walking.png"),
    "Traffic Street": os.path.join(IMAGE_BASE_PATH, "traffic.png"),
    "Overlap": os.path.join(IMAGE_BASE_PATH, "crossing.png"),
    "Gas Station": os.path.join(IMAGE_BASE_PATH, "gas.png"),
    "Garage": os.path.join(IMAGE_BASE_PATH, "garage.png"),
    "House": [os.path.join(IMAGE_BASE_PATH, "house{}.png").format(i) for i in range(1, 4)],
    "Office": [os.path.join(IMAGE_BASE_PATH, "office{}.png").format(i) for i in range(1, 4)],
    "Store": [os.path.join(IMAGE_BASE_PATH, "store{}.png").format(i) for i in range(1, 4)],
}

ICON_SIZE_DICT = {
    "Car": SCALE*6,
    "Ambulance": SCALE*6,
    "Bus": SCALE*6,
    "Tiro": SCALE*6,
    "Police": SCALE*6,
    "Reckless": SCALE*6,
    "Pedestrian": SCALE*4,
    "Pedestrian_old": SCALE*4,
    "Pedestrian_young": SCALE*4,
    "Walking Street": SCALE*10,
    "Traffic Street": SCALE*10,
    "Overlap": SCALE*10,
    "Gas Station": SCALE*BUILDING_SIZE,
    "Garage": SCALE*BUILDING_SIZE,
    "House": SCALE*BUILDING_SIZE,
    "Office": SCALE*BUILDING_SIZE,
    "Store": SCALE*BUILDING_SIZE,
}

def resize_with_aspect_ratio(image, base_size):
    # Determine the shorter side of the image
    short_side = min(image.shape[:2])
    
    # Calculate the scaling factor
    scale_factor = base_size / short_side
    
    # Calculate the new dimensions of the image
    new_dims = (int(image.shape[1] * scale_factor), int(image.shape[0] * scale_factor))
    
    # Resize the image with the new dimensions
    resized_img = cv2.resize(image, new_dims, interpolation=cv2.INTER_LINEAR)
    
    return resized_img

def gridmap2img_static(gridmap, icon_dict, ego_id, show_start_goal_markers=True):
    # step 1: get the size of the gridmap, create a blank image with size*SCALE
    height, width = gridmap.shape[1], gridmap.shape[2]
    img = np.ones((height*SCALE, width*SCALE, 3), np.uint8) * 255  # assuming white background
    resized_grid = np.repeat(np.repeat(gridmap, SCALE, axis=1), SCALE, axis=2)
    
    # step 2: fill the image with walking street icons
    walking_icon = icon_dict["Walking Street"]
    for i in range(0, img.shape[0], walking_icon.shape[0]):
        for j in range(0, img.shape[1], walking_icon.shape[1]):
            # Calculate the dimensions of the region left in img
            h_space_left = min(walking_icon.shape[0], img.shape[0] - i)
            w_space_left = min(walking_icon.shape[1], img.shape[1] - j)

            # Paste the walking_icon (or its sliced version) to img
            img[i:i+h_space_left, j:j+w_space_left] = walking_icon[:h_space_left, :w_space_left]

    # step 3: read the STREET layer of gridmap, paste the traffic street icons on the traffic street region
    # For the traffic icon
    traffic_icon = icon_dict["Traffic Street"]
    traffic_img = np.zeros_like(img)
    traffic_mask = resized_grid[STREET_ID] == TYPE_MAP["Traffic Street"]
    for i in range(0, img.shape[0], traffic_icon.shape[0]):
        for j in range(0, img.shape[1], traffic_icon.shape[1]):
            # Calculate the dimensions of the region left in img
            h_space_left = min(traffic_icon.shape[0], img.shape[0] - i)
            w_space_left = min(traffic_icon.shape[1], img.shape[1] - j)

            # Paste the walking_icon (or its sliced version) to img
            traffic_img[i:i+h_space_left, j:j+w_space_left] = traffic_icon[:h_space_left, :w_space_left]
    img[traffic_mask] = traffic_img[traffic_mask]

    # For the Overlap and mid lane icon
    traffic_icon = icon_dict["Overlap"]
    traffic_img = np.zeros_like(img)
    traffic_mask = resized_grid[STREET_ID] == TYPE_MAP["Overlap"]
    for i in range(0, img.shape[0], traffic_icon.shape[0]):
        for j in range(0, img.shape[1], traffic_icon.shape[1]):
            # Calculate the dimensions of the region left in img
            h_space_left = min(traffic_icon.shape[0], img.shape[0] - i)
            w_space_left = min(traffic_icon.shape[1], img.shape[1] - j)

            # Paste the walking_icon (or its sliced version) to img
            traffic_img[i:i+h_space_left, j:j+w_space_left] = traffic_icon[:h_space_left, :w_space_left]
    img[traffic_mask] = traffic_img[traffic_mask]

    traffic_img = np.zeros_like(img)
    traffic_mask = resized_grid[STREET_ID] == TYPE_MAP["Mid Lane"]
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            # Paste the walking_icon (or its sliced version) to img
            traffic_img[i:i+h_space_left, j:j+w_space_left] = [255, 215, 0]
    img[traffic_mask] = traffic_img[traffic_mask]

    # For the building icons
    for building in BUILDING_TYPES:
        building_map = resized_grid[BUILDING_ID] == TYPE_MAP[building]
        building_icon = icon_dict[building]
        labeled_matrix, num = label(building_map)

        for i in range(1, num+1):
            local = torch.tensor(labeled_matrix == i)
            pixels = torch.nonzero(local.float())
            rows = pixels[:, 0]
            cols = pixels[:, 1]
            left = torch.min(cols).item()
            right = torch.max(cols).item()
            top = torch.min(rows).item()
            bottom = torch.max(rows).item()
            if building in ["House", "Office", "Store"]:
                icon_id = np.random.choice(3)
                icon = building_icon[icon_id]
            else:
                icon = building_icon
            icon_mask = np.sum(icon > 1, axis=2) > 0
            img[bottom-icon.shape[0]:bottom, left:left+icon.shape[1]][icon_mask] = icon[icon_mask]

    # add ego agent start and goal from the explicit layer encodings
    if show_start_goal_markers and ego_id > 0:
        ego_map = gridmap[ego_id]
        ego_type_value = int(np.floor(np.max(ego_map)))
        goal_mask = np.isclose(ego_map, ego_type_value + AGENT_GOAL_PLUS)
        start_mask = np.isclose(ego_map, ego_type_value + AGENT_START_PLUS)

        if goal_mask.any():
            goal_pos = np.argwhere(goal_mask)[0]
            goal_x, goal_y = goal_pos[0] * SCALE, goal_pos[1] * SCALE
            cv2.drawMarker(
                img,
                (goal_y, goal_x),
                (255, 0, 0),
                markerType=cv2.MARKER_STAR,
                markerSize=30,
                thickness=5,
            )

        if start_mask.any():
            start_pos = np.argwhere(start_mask)[0]
            start_x, start_y = start_pos[0] * SCALE, start_pos[1] * SCALE
            cv2.drawMarker(
                img,
                (start_y, start_x),
                (0, 0, 255),
                markerType=cv2.MARKER_STAR,
                markerSize=30,
                thickness=5,
            )

    return img

def get_pos(local_layer):
    local_layer[local_layer==0] += 0.1
    pos_layer = local_layer == local_layer.astype(np.int64)
    pixels = torch.nonzero(torch.tensor(pos_layer.astype(np.float32)))
    rows = pixels[:, 0]
    cols = pixels[:, 1]
    left = torch.min(cols).item()
    right = torch.max(cols).item()
    top = torch.min(rows).item()
    bottom = torch.max(rows).item()
    return (left, top, right, bottom)

def get_direction(left, left_, top, top_):
    if left_ > left:
        return "right"
    elif left_ < left:
        return "left"
    elif top_ > top:
        return "down"
    elif top_ < top:
        return "up"
    else:
        return "none"

def rotate_image(image, angle):
    """ Rotate the given image by the specified angle """
    if angle >= 0:
        return image.rotate(angle, expand=True)
    else:
        return image.transpose(Image.FLIP_LEFT_RIGHT)


def shorten_car_along_heading(image, direction, length_scale=CAR_RENDER_LENGTH_SCALE):
    """Shorten only the car's long axis while preserving its width."""
    if length_scale >= 0.999:
        return image

    width, height = image.size
    if direction in ("up", "down"):
        new_height = max(1, int(round(height * length_scale)))
        if new_height == height:
            return image
        return image.resize((width, new_height), Image.Resampling.LANCZOS)
    if direction in ("left", "right"):
        new_width = max(1, int(round(width * length_scale)))
        if new_width == width:
            return image
        return image.resize((new_width, height), Image.Resampling.LANCZOS)
    return image

def create_custom_mask(image, threshold=0.1):
    if image.mode == 'RGBA':
        # Use the existing alpha channel
        r, g, b, alpha = image.split()
        alpha = alpha.point(lambda p: 255 if p > threshold else 0)
        return alpha
    else:
        # Create a new mask
        mask = Image.new('L', image.size, 0)  # Start with a fully transparent mask
        pixels = image.load()
        mask_pixels = mask.load()
        
        for i in range(image.size[0]):  # Iterate over width
            for j in range(image.size[1]):  # Iterate over height
                r, g, b = pixels[i, j][:3]
                luminance = int(0.299*r + 0.587*g + 0.114*b)
                if luminance > threshold:
                    mask_pixels[i, j] = 255
        return mask
    
def get_steet_type(gridmap, position):
    l, t, r, b = position
    partial_grid_horizontal = gridmap[STREET_ID, t, l-10:l+10]
    if np.sum(partial_grid_horizontal == TYPE_MAP["Mid Lane"]) > 0:
        return "v"
    partial_grid_vertical = gridmap[STREET_ID, t-10:t+10, l]
    if np.sum(partial_grid_vertical == TYPE_MAP["Mid Lane"]) > 0:
        return "h"
    return None


def _get_agent_metadata(agents, agent_type, layer_id):
    if not agents:
        return None

    agent_name = "{}_{}".format(agent_type, layer_id)
    if agent_name in agents:
        return agents[agent_name]

    for metadata in agents.values():
        if metadata.get("layer_id") == layer_id:
            return metadata

    return None

def paste_car_on_map(map_image, car_image, position, direction, type, position_last=None, street_type=None):
    """ Paste car on the map with the correct orientation and position """
    l, t, r, b = position
    if type == "Car":
        # Define rotation angles for directions
        rotation_angles = {
            'up': 0,
            'right': 270,
            'down': 180,
            'left': 90,
            'none': 0
        }
    elif type == "Pedestrian":
        rotation_angles = {
            'up': 0,
            'right': 0,
            'down': -1,
            'left': -1,
            'none': 0
        }


    # Rotate the car image based on the direction
    rotated_car = rotate_image(car_image, rotation_angles[direction])
    if type == "Car":
        rotated_car = shorten_car_along_heading(rotated_car, direction)

    mask = create_custom_mask(rotated_car)

    # Anchor sprites by the simulator state cell center. This keeps the visual
    # icon aligned with the actual occupancy/debug cell instead of making the
    # state look shifted toward the front bumper.
    if type == "Car":
        if direction == 'none':
            if position_last is not None:
                new_position = tuple(position_last)
            else:
                center_position = ((l+r)//2, (t+b)//2)
                new_position = (center_position[0] - rotated_car.width//2, center_position[1] - rotated_car.height//2)
        else:
            center_position = ((l+r)//2, (t+b)//2)
            new_position = (center_position[0] - rotated_car.width//2, center_position[1] - rotated_car.height//2)
    elif type == "Pedestrian":
        if direction == "none":
            if position_last is not None:
                new_position = tuple(position_last)
            else:
                new_position = (l, t)
        else:
            center_position = ((l+r)//2, (t+b)//2)
            new_position = (center_position[0] - rotated_car.width//2, center_position[1] - rotated_car.height//2)
    

    # Paste the car image onto the map
    map_image.paste(rotated_car, new_position, mask)

    return rotated_car, map_image, list(new_position)


def draw_debug_cell(map_image, position, outline=(0, 255, 255), width=2):
    """Draw the exact simulator grid cell/bbox used for the agent state."""
    draw = ImageDraw.Draw(map_image)
    l, t, r, b = position
    draw.rectangle([l, t, max(l, r - 1), max(t, b - 1)], outline=outline, width=width)
    cx = (l + r) // 2
    cy = (t + b) // 2
    draw.line([(cx - 3, cy), (cx + 3, cy)], fill=outline, width=width)
    draw.line([(cx, cy - 3), (cx, cy + 3)], fill=outline, width=width)

def gridmap2img_agents(
    gridmap,
    gridmap_,
    icon_dict,
    static_map,
    ego_id,
    last_icons=None,
    agents=None,
    show_agent_debug_overlay=True,
):
    current_map = static_map.copy()
    current_map = Image.fromarray(current_map)
    resized_world = np.repeat(np.repeat(gridmap, SCALE, axis=1), SCALE, axis=2)
    agent_layer = gridmap[BASIC_LAYER:]
    resized_grid = np.repeat(np.repeat(agent_layer, SCALE, axis=1), SCALE, axis=2)
    agent_layer_ = gridmap_[BASIC_LAYER:]
    resized_grid_ = np.repeat(np.repeat(agent_layer_, SCALE, axis=1), SCALE, axis=2)
    icon_dict_local = {
        "icon": {},
        "pos": {}
    }

    for i in range(resized_grid.shape[0]):
        local_layer = resized_grid[i]
        left, top, right, bottom = get_pos(local_layer)
        local_layer_ = resized_grid_[i]
        left_, top_, right_, bottom_ = get_pos(local_layer_)
        direction = get_direction(left, left_, top, top_)
        pos = (left, top, right, bottom)
        
        agent_type = LABEL_MAP[local_layer[top, left].item()]     
        layer_id = BASIC_LAYER + i
        agent_metadata = _get_agent_metadata(agents, agent_type, layer_id)
        if agent_metadata is not None:
            concepts = agent_metadata.get("concepts", {})
            is_young = False
            is_old = False
            if "old" in concepts.keys():
                if concepts["old"] == 1.0:
                    is_old = True
            if "young" in concepts.keys():
                if concepts["young"] == 1.0:
                    is_young = True
            if agent_type == "Car":
                icon_list = icon_dict[agent_type].copy()
                icon_id = i % len(icon_list)
                icon = icon_list[icon_id]
            elif is_old:
                icon = icon_dict["Pedestrian_old"]
            elif is_young:
                icon = icon_dict["Pedestrian_young"]
            else:
                if agent_type == "Pedestrian":
                    icon_list = icon_dict[agent_type].copy()
                    icon_id = i%len(icon_list)
                    icon = icon_list[icon_id]
        else:
            if agent_type == "Car":
                icon_list = icon_dict[agent_type].copy()
                icon_id = i % len(icon_list)
                icon = icon_list[icon_id]
            else:
                icon_list = icon_dict[agent_type]
                icon_id = i%len(icon_list)
                icon = icon_list[icon_id]

        if agent_type == "Car":
            street_type = get_steet_type(resized_world, pos)
        else:
            street_type = None    

        if last_icons is not None:
            if direction == "none":
                icon = last_icons["icon"]["{}_{}".format(agent_type, i)][1]
                position = last_icons["pos"]["{}_{}".format(agent_type, i)]
                icon, current_map, last_position = paste_car_on_map(
                    current_map,
                    icon,
                    pos,
                    direction,
                    agent_type,
                    position_last=position,
                    street_type=street_type,
                )
            else:
                icon = last_icons["icon"]["{}_{}".format(agent_type, i)][0]
                icon, current_map, last_position = paste_car_on_map(
                    current_map,
                    icon,
                    pos,
                    direction,
                    agent_type,
                    street_type=street_type,
                )
            last_icons["icon"]["{}_{}".format(agent_type, i)][1] = icon
            last_icons["pos"]["{}_{}".format(agent_type, i)] = last_position
        else:
            icon_img = Image.fromarray(icon) 
            icon_dict_local["icon"]["{}_{}".format(agent_type, i)] = [icon_img]
            current_icon, current_map, last_position = paste_car_on_map(
                current_map,
                icon_img,
                pos,
                direction,
                agent_type,
                street_type=street_type,
            )
            icon_dict_local["icon"]["{}_{}".format(agent_type, i)].append(current_icon)
            icon_dict_local["pos"]["{}_{}".format(agent_type, i)] = last_position

        # The sprite is intentionally larger than a single grid cell, which can
        # make the true simulator location look misleading near intersections.
        # Overlay the exact ego state cell so debugging matches the trace.
        if show_agent_debug_overlay and (
            layer_id == ego_id or (agent_metadata is not None and bool(agent_metadata.get("is_rl_agent", False)))
        ):
            draw_debug_cell(current_map, pos)

    if last_icons is not None:
        return current_map, last_icons
    else:
        return current_map, icon_dict_local

def main(
    pkl_path,
    ego_id,
    output_folder,
    scale_factor=1,
    crop_size=None,
    max_step=None,
    target_step=None,
    show_step_label=True,
    show_start_goal_markers=True,
    show_agent_debug_overlay=True,
):
    icon_dict = {}
    os.path.exists(output_folder) or os.makedirs(output_folder)
    for key in PATH_DICT.keys():
        if isinstance(PATH_DICT[key], list):
            raw_img = [cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB) for path in PATH_DICT[key]]
            resized_img = [resize_with_aspect_ratio(img, ICON_SIZE_DICT[key]) for img in raw_img]
            icon_dict[key] = resized_img
        else:
            raw_img = cv2.cvtColor(cv2.imread(PATH_DICT[key]), cv2.COLOR_BGR2RGB)
            resized_img = resize_with_aspect_ratio(raw_img, ICON_SIZE_DICT[key])
            icon_dict[key] = resized_img

    obs, agents = _load_rollout_like_data(pkl_path)

    time_steps = list(obs.keys())
    time_steps.sort()
    if max_step is not None:
        time_steps = [step for step in time_steps if step <= max_step]
        if len(time_steps) < 2:
            raise RuntimeError("Need at least two timesteps to render up to max_step={}.".format(max_step))
    if target_step is not None:
        if target_step not in time_steps:
            raise RuntimeError("target_step={} not found in rollout.".format(target_step))
        next_step = target_step + 1
        if next_step not in obs:
            raise RuntimeError("Need target_step+1={} to render target_step={}.".format(next_step, target_step))
        time_steps = [target_step, next_step]
    static_map = gridmap2img_static(
        _to_numpy_world(obs[time_steps[0]]["World"]),
        icon_dict,
        ego_id,
        show_start_goal_markers=show_start_goal_markers,
    )
    static_map_img = Image.fromarray(static_map)
    # static_map_img.save("{}/static_layout.png".format(output_folder))
    last_icons = None
    if target_step is not None:
        render_iter = [(time_steps[0], time_steps[1])]
        use_pairs = True
    else:
        render_iter = trange(time_steps[0], time_steps[-2] + 1)
        use_pairs = False

    for item in render_iter:
        if use_pairs:
            key, next_key = item
        else:
            key = item
            next_key = key + 1
        grid = _to_numpy_world(obs[key]["World"])
        grid_ = _to_numpy_world(obs[next_key]["World"])
        img, last_icons = gridmap2img_agents(
            grid,
            grid_,
            icon_dict,
            static_map,
            ego_id,
            last_icons,
            agents,
            show_agent_debug_overlay=show_agent_debug_overlay,
        )
        if show_step_label:
            text = "#{}".format(key)
            position = (10, 10)
            draw = ImageDraw.Draw(img)
            try:
                font = ImageFont.truetype("arial.ttf", size=100)
            except IOError:
                font = ImageFont.load_default()
            color = (255, 255, 255)
            draw.text(position, text, fill=color, font=font)

        if crop_size is not None:
            img = img.crop((0, 0, min(crop_size, img.width), min(crop_size, img.height)))
        if scale_factor != 1:
            img = img.resize(
                (int(img.width * scale_factor), int(img.height * scale_factor)),
                Image.Resampling.NEAREST,
            )

        # Save the image
        output_path = "{}/step_{}.png".format(output_folder, key)
        img.save(output_path)
    cv2.destroyAllWindows()

    return

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Create an animated GIF from a sequence of images.")
    parser.add_argument("--pkl", default='log_rl/oracle_test_train_hard_1.pkl', help="Path to the folder containing image files.")
    parser.add_argument("--ego_id", type=int, default=3, help="which agent is ego agent. Visualize the ego agent's start and goal. This is layer_id")
    parser.add_argument("--output_folder", default="vis", help="Output folder.")
    parser.add_argument("--scale_factor", type=float, default=1.0, help="Upscale rendered frames by this factor.")
    parser.add_argument("--crop_size", type=int, default=None, help="Optional top-left square crop size. Defaults to full frame.")
    parser.add_argument("--max_step", type=int, default=None, help="Optional maximum timestep to render.")
    parser.add_argument("--target_step", type=int, default=None, help="Optional single timestep to render.")
    parser.add_argument("--hide_step_label", action="store_true", help="Do not draw the timestep label on rendered frames.")
    parser.add_argument("--hide_start_goal_markers", action="store_true", help="Do not draw ego start/goal markers on rendered frames.")
    parser.add_argument("--hide_agent_debug_overlay", action="store_true", help="Do not draw the RL/ego debug cell overlay.")
    
    args = parser.parse_args()

    # Call the function with provided arguments
    main(
        args.pkl,
        args.ego_id,
        args.output_folder,
        scale_factor=args.scale_factor,
        crop_size=args.crop_size,
        max_step=args.max_step,
        target_step=args.target_step,
        show_step_label=not args.hide_step_label,
        show_start_goal_markers=not args.hide_start_goal_markers,
        show_agent_debug_overlay=not args.hide_agent_debug_overlay,
    )
