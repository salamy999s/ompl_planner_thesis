import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import gaussian_filter
import utm
import math
import ompl.base as ob
import ompl.geometric as og
import time

# Load CSV data and normalize elevations (as in salma.py)
csv_filepath = '/home/salmaaldhanhani/ompl/maps/csv/remah_map_0.csv'
data = pd.read_csv(csv_filepath)
longitudes = data['Longitude'].values  
latitudes = data['Latitude'].values    
elevations = data['Elevation'].values  
elevations = elevations - np.min(elevations)

# Convert lat/lon to UTM coordinates
utm_coords = np.array([utm.from_latlon(lat, lon)[:2] for lat, lon in zip(latitudes, longitudes)])
utm_x = utm_coords[:, 0]
utm_y = utm_coords[:, 1]

# Function to create grid
def create_grid(utm_x, utm_y, elevations, window_width=1500, window_height=1500, debug=False):
    numRows = int(window_height/4)
    numColumns = int(window_width/4)

    x_min = np.min(utm_x)
    x_max = np.max(utm_x)
    y_min = np.min(utm_y)
    y_max = np.max(utm_y)

    x_step = (x_max - x_min) / (numColumns - 1)
    y_step = (y_max - y_min) / (numRows - 1)

    elevation_grid = np.zeros((numRows, numColumns))
    cost_grid = np.zeros((numRows, numColumns))

    for x, y, ele in zip(utm_x, utm_y, elevations):
        x_idx = int((x - x_min) / x_step)
        y_idx = int((y - y_min) / y_step)

        x_idx = max(0, min(x_idx, numColumns - 1))
        y_idx = max(0, min(y_idx, numRows - 1))

        elevation_grid[x_idx, y_idx] = ele
        cost_grid[x_idx, y_idx] = ele / np.max(elevations)

        if debug:
            print(f"Point ({x}, {y}, {ele}) mapped to grid index ({x_idx}, {y_idx}) with normalized cost {cost_grid[y_idx, x_idx]:.4f}")

    return elevation_grid, cost_grid, x_min, x_max, y_min, y_max


elevation_grid, cost_grid, x_min, x_max, y_min, y_max = create_grid(utm_x, utm_y, elevations)

def euclidean_heuristic_2_5d(node1, node2, elevation_grid, elevation_weight=3.0, slope_weight=1.0, debug=True):
    x1, y1, z1 = node1
    x2, y2, z2 = node2
    dist_2d = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    elevation_diff = abs(z1 - z2)
    slope = elevation_diff / dist_2d if dist_2d > 0 else 0
    cost = elevation_grid[int(y1 * elevation_grid.shape[0]), int(x1 * elevation_grid.shape[1])]

    total_cost = dist_2d + elevation_weight * elevation_diff + slope_weight * slope + cost
    
    if debug:
        print(f"Heuristic between ({x1}, {y1}, {z1}) and ({x2}, {y2}, {z2}):")
        print(f"2D Distance={dist_2d}, Elevation Diff={elevation_diff}, Slope={slope}, Cost={cost}, Total Cost={total_cost}")
    
    return total_cost

def manhattan_heuristic_2_5d(node1, node2, elevation_grid):
    x1, y1, z1 = node1
    x2, y2, z2 = node2
    elevation_diff = abs(z1 - z2)
    slope = elevation_diff / (abs(x1 - x2) + abs(y1 - y2)) if (x1 != x2 or y1 != y2) else 0
    cost = elevation_grid[int(y1 * elevation_grid.shape[0]), int(x1 * elevation_grid.shape[1])]
    return abs(x1 - x2) + abs(y1 - y2) + elevation_diff + slope + cost

def geodesic_heuristic_25d(node1, node2, elevation_grid, num_samples=200, elevation_scale=10.0):
    x_samples = np.linspace(node1[0], node2[0], num_samples)
    y_samples = np.linspace(node1[1], node2[1], num_samples)

    heightmap_height, heightmap_width = elevation_grid.shape
    
    z_samples = []
    for x, y in zip(x_samples, y_samples):
        x_index = int(x * (heightmap_width - 1))
        y_index = int(y * (heightmap_height - 1))

        if 0 <= x_index < heightmap_width and 0 <= y_index < heightmap_height:
            z_samples.append(elevation_grid[y_index, x_index])
        else:
            x_clamped = np.clip(x_index, 0, heightmap_width - 1)
            y_clamped = np.clip(y_index, 0, heightmap_height - 1)
            z_samples.append(elevation_grid[y_clamped, x_clamped])

    total_distance = sum(math.sqrt((x_samples[i] - x_samples[i-1])**2 +
                                   (y_samples[i] - y_samples[i-1])**2 +
                                   (elevation_scale * (z_samples[i] - z_samples[i-1]))**2)
                         for i in range(1, num_samples))
    return total_distance

def calculate_path_cost(path, elevation_grid, elevation_weight=3.0, slope_weight=1.0):
    total_cost = 0
    for i in range(1, len(path)):
        node1 = path[i - 1]
        node2 = path[i]
        cost = euclidean_heuristic_2_5d(node1, node2, elevation_grid, elevation_weight, slope_weight, debug=False)
        total_cost += cost
    return total_cost

def isStateValid25D(state, elevation_grid, max_slope=1.0, elevation_threshold=0.8, debug=False):
    x_idx = int(state.getX() * (elevation_grid.shape[1] - 1))
    y_idx = int(state.getY() * (elevation_grid.shape[0] - 1))

    if 0 <= x_idx < elevation_grid.shape[1] and 0 <= y_idx < elevation_grid.shape[0]:
        z_value = float(elevation_grid[y_idx, x_idx])

        if debug:
            print(f"Checking state at index ({x_idx}, {y_idx}) with elevation {z_value}")

        # Enforce elevation constraint
        if z_value > elevation_threshold * np.max(elevation_grid):
            if debug:
                print(f"Invalid state due to high elevation: x_idx={x_idx}, y_idx={y_idx}, elevation={z_value}")
            return False

        # Calculate slopes
        if x_idx > 0 and y_idx > 0:
            z_left = float(elevation_grid[y_idx, x_idx - 1])
            z_down = float(elevation_grid[y_idx - 1, x_idx])

            slope_x = abs(z_value - z_left)
            slope_y = abs(z_value - z_down)
            slope = max(slope_x, slope_y)

            if debug:
                print(f"Slope at ({x_idx}, {y_idx}): slope_x={slope_x}, slope_y={slope_y}, max_slope={slope}")

            if slope > max_slope:
                if debug:
                    print(f"Invalid state due to steep slope: x_idx={x_idx}, y_idx={y_idx}, slope={slope}")
                return False

        if debug:
            print(f"Valid state: x_idx={x_idx}, y_idx={y_idx}, z_value={z_value}")
        return True

    if debug:
        print(f"Out of bounds: x_idx={x_idx}, y_idx={y_idx}")
    return False
class ElevationSlopeObjective(ob.OptimizationObjective):
    def __init__(self, si, elevation_map, slope_weight=1.0, elevation_weight=0.5):
        super().__init__(si)
        self.elevation_map = elevation_map
        self.slope_weight = slope_weight
        self.elevation_weight = elevation_weight

    def stateCost(self, state):
        # Get the elevation at the current state's coordinates
        x, y = int(state[0]), int(state[1])
        if 0 <= x < self.elevation_map.shape[1] and 0 <= y < self.elevation_map.shape[0]:
            elevation = self.elevation_map[y, x]
            # Penalize based on elevation to avoid high areas
            return ob.Cost(self.elevation_weight * elevation)
        return ob.Cost(float('inf'))  # Invalid state or out of bounds

    def motionCost(self, s1, s2):
        # Euclidean distance between states s1 and s2
        distance = math.sqrt((s1[0] - s2[0])**2 + (s1[1] - s2[1])**2)

        # Calculate elevation change
        x1, y1 = int(s1[0]), int(s1[1])
        x2, y2 = int(s2[0]), int(s2[1])
        
        if (0 <= x1 < self.elevation_map.shape[1] and 0 <= y1 < self.elevation_map.shape[0] and
            0 <= x2 < self.elevation_map.shape[1] and 0 <= y2 < self.elevation_map.shape[0]):
            elevation1 = self.elevation_map[y1, x1]
            elevation2 = self.elevation_map[y2, x2]
            elevation_diff = abs(elevation2 - elevation1)

            # Calculate slope as elevation change over distance
            slope = elevation_diff / (distance + 1e-5)  # Add small epsilon to avoid division by zero

            # Combined cost with weights for slope and distance
            cost = distance + self.slope_weight * slope + self.elevation_weight * elevation_diff
            return ob.Cost(cost)
        
        return ob.Cost(float('inf'))  # Return infinite cost if out of bounds


class EuclideanObjective(ob.OptimizationObjective):
    def __init__(self, si, elevation_grid):
        super().__init__(si)
        self.elevation_grid = elevation_grid

    def motionCost(self, s1, s2):
        node1 = (s1.getX(), s1.getY(), float(self.elevation_grid[int(s1.getY() * self.elevation_grid.shape[0]), int(s1.getX() * self.elevation_grid.shape[1])]))
        node2 = (s2.getX(), s2.getY(), float(self.elevation_grid[int(s2.getY() * self.elevation_grid.shape[0]), int(s2.getX() * self.elevation_grid.shape[1])]))
        cost = euclidean_heuristic_2_5d(node1, node2, self.elevation_grid)
        return ob.Cost(cost)

class ManhattanObjective(ob.OptimizationObjective):
    def __init__(self, si, elevation_grid):
        super().__init__(si)
        self.elevation_grid = elevation_grid

    def motionCost(self, s1, s2):
        node1 = (s1.getX(), s1.getY(), float(self.elevation_grid[int(s1.getY() * self.elevation_grid.shape[0]), int(s1.getX() * self.elevation_grid.shape[1])]))
        node2 = (s2.getX(), s2.getY(), float(self.elevation_grid[int(s2.getY() * self.elevation_grid.shape[0]), int(s2.getX() * self.elevation_grid.shape[1])]))
        cost = manhattan_heuristic_2_5d(node1, node2, self.elevation_grid)
        return ob.Cost(cost)

class GeodesicObjective(ob.OptimizationObjective):
    def __init__(self, si, elevation_grid, num_samples=200, elevation_scale=10.0):
        super().__init__(si)
        self.elevation_grid = elevation_grid
        self.num_samples = num_samples
        self.elevation_scale = elevation_scale

    def motionCost(self, s1, s2):
        node1 = (s1.getX(), s1.getY(), float(self.elevation_grid[int(s1.getY() * self.elevation_grid.shape[0]), int(s1.getX() * self.elevation_grid.shape[1])]))
        node2 = (s2.getX(), s2.getY(), float(self.elevation_grid[int(s2.getY() * self.elevation_grid.shape[0]), int(s2.getX() * self.elevation_grid.shape[1])]))
        cost = geodesic_heuristic_25d(node1, node2, self.elevation_grid, num_samples=self.num_samples, elevation_scale=self.elevation_scale)
        return ob.Cost(cost)

# 2.5D Path Planning Function with debug information
def plan_2_5d_with_slope_constraints(elevation_grid, turning_radius=0.05, solve_time=100.0, debug=False, objective_type="Euclidean"):
    start_time = time.time()
    space = ob.DubinsStateSpace(turning_radius)
    
    bounds = ob.RealVectorBounds(2)
    bounds.setLow(0, 0.05)
    bounds.setHigh(0, 0.95)
    bounds.setLow(1, 0.05)
    bounds.setHigh(1, 0.95)
    space.setBounds(bounds)

    space_information = ob.SpaceInformation(space)
    space_information.setStateValidityChecker(ob.StateValidityCheckerFn(
        lambda s: isStateValid25D(s, elevation_grid, debug=debug)))

    start, goal = ob.State(space), ob.State(space)
    start[0], start[1], start[2] = 0.2, 0.2, 0.0
    goal[0], goal[1], goal[2] = 0.8, 0.7, 0.0

    if debug:
        print(f"Start: ({start[0]}, {start[1]}) | Goal: ({goal[0]}, {goal[1]})")

    # Choose the objective based on the objective_type argument
    if objective_type == "Euclidean":
        objective = EuclideanObjective(space_information, elevation_grid)
    elif objective_type == "Manhattan":
        objective = ManhattanObjective(space_information, elevation_grid)
    elif objective_type == "Geodesic":
        objective = GeodesicObjective(space_information, elevation_grid, num_samples=200, elevation_scale=10.0)
    else:
        raise ValueError("Invalid objective type. Choose from 'Euclidean', 'Manhattan', or 'Geodesic'.")

    # Set up problem definition with the chosen objective
    pdef = ob.ProblemDefinition(space_information)
    pdef.setStartAndGoalStates(start, goal)
    pdef.setOptimizationObjective(objective)

    planner = og.BITstar(space_information)
    planner.setProblemDefinition(pdef)
    planner.setup()

    solved = planner.solve(solve_time)
    if solved:
        path_geometric = pdef.getSolutionPath()
        path_geometric.interpolate(200)
        path_states = [(state.getX(), state.getY(), elevation_grid[int(state.getY() * elevation_grid.shape[0]), 
                                                                   int(state.getX() * elevation_grid.shape[1])])
                       for state in path_geometric.getStates()]
        elapsed_time = time.time() - start_time
        if debug:
            print(f"Path found in {elapsed_time:.4f} seconds with {len(path_states)} states.")
        return path_states, elapsed_time
    else:
        if debug:
            print("No solution found.")
    return None, None

def check_path_constraints(path, elevation_grid, max_slope=0.8, elevation_threshold=0.6):
    violations = {
        "elevation_violations": 0,
        "slope_violations": 0,
        "total_segments": len(path) - 1
    }
    elevation_limit = elevation_threshold * np.max(elevation_grid)
    
    for i in range(1, len(path)):
        node1 = path[i - 1]
        node2 = path[i]
        
        x1, y1, z1 = node1
        x2, y2, z2 = node2
        
        # Check elevation constraint
        if z1 > elevation_limit or z2 > elevation_limit:
            violations["elevation_violations"] += 1
            
        # Calculate slope and check slope constraint
        elevation_diff = abs(z1 - z2)
        dist_2d = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        slope = elevation_diff / dist_2d if dist_2d > 0 else 0
        if slope > max_slope:
            violations["slope_violations"] += 1
            
    return violations

def analyze_path(path, elevation_grid, elevation_weight=3.0, slope_weight=1.0, max_slope=0.8, elevation_threshold=0.6):
    # Calculate total path cost
    total_cost = calculate_path_cost(path, elevation_grid, elevation_weight, slope_weight)
    
    # Check for constraint violations
    violations = check_path_constraints(path, elevation_grid, max_slope, elevation_threshold)
    
    # Print analysis report
    print("Path Analysis Report:")
    print(f"Total Path Cost: {total_cost:.4f}")
    print(f"Total Segments: {violations['total_segments']}")
    print(f"Elevation Violations: {violations['elevation_violations']}")
    print(f"Slope Violations: {violations['slope_violations']}")
    print("Path respects all constraints" if violations["elevation_violations"] == 0 and violations["slope_violations"] == 0 else "Path does not fully respect constraints")

def plot_path_analysis(path, elevation_grid):
    elevations = [elevation_grid[int(p[1] * elevation_grid.shape[0]), int(p[0] * elevation_grid.shape[1])] for p in path]
    slopes = [abs(elevations[i+1] - elevations[i]) for i in range(len(elevations) - 1)]
    
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))
    
    axs[0].plot(range(len(elevations)), elevations, label="Elevation", color="blue")
    axs[0].set_title("Elevation along Path")
    axs[0].set_xlabel("Path Segment")
    axs[0].set_ylabel("Elevation")
    axs[0].legend()
    
    axs[1].plot(range(len(slopes)), slopes, label="Slope", color="red")
    axs[1].set_title("Slope along Path")
    axs[1].set_xlabel("Path Segment")
    axs[1].set_ylabel("Slope")
    axs[1].legend()
    
    plt.tight_layout()
    plt.show()

def calculate_total_distance(path, heuristic_func, elevation_grid=None):
    total_distance = 0
    for i in range(len(path) - 1):
        if elevation_grid is not None:
            total_distance += heuristic_func(path[i], path[i + 1], elevation_grid)
        else:
            total_distance += heuristic_func(path[i], path[i + 1])
    return total_distance


# Plotting function for visualization
def plot_paths(elevation_grid, path_2_5d):
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    smoothed_grid_2d = gaussian_filter(elevation_grid, sigma=2)
    axs[0].imshow(smoothed_grid_2d, cmap='terrain', origin='lower')
    axs[0].set_title("2D Elevation Map")
    
    if path_2_5d:
        x_vals = [p[0] * elevation_grid.shape[1] for p in path_2_5d]
        y_vals = [p[1] * elevation_grid.shape[0] for p in path_2_5d]
        axs[0].plot(x_vals, y_vals, 'r-', label='Optimal Path')
    
    ax3d = fig.add_subplot(122, projection='3d')
    x_grid = np.linspace(0, elevation_grid.shape[1], elevation_grid.shape[1])
    y_grid = np.linspace(0, elevation_grid.shape[0], elevation_grid.shape[0])
    x_grid, y_grid = np.meshgrid(x_grid, y_grid)
    smoothed_grid_3d = gaussian_filter(elevation_grid, sigma=2)
    ax3d.plot_surface(x_grid, y_grid, smoothed_grid_3d, cmap='terrain', edgecolor='none')

    if path_2_5d:
        x_vals = [p[0] * elevation_grid.shape[1] for p in path_2_5d]
        y_vals = [p[1] * elevation_grid.shape[0] for p in path_2_5d]
        z_vals = [p[2] for p in path_2_5d]
        ax3d.plot(x_vals, y_vals, z_vals, 'r-', label='Optimal Path', lw=2)

    plt.tight_layout()
    plt.show()

# Adjust isStateValid25D parameters in plan_2_5d_with_slope_constraints call
# Run path planning
solve_time = 200.0  # Adjusted solve time for path planning

# Test with Euclidean Objective
path_euclidean, time_euclidean = plan_2_5d_with_slope_constraints(elevation_grid, turning_radius=0.05, solve_time=solve_time, objective_type="Euclidean")
if path_euclidean:
    print(f"Euclidean Objective - Path found in {time_euclidean:.4f} seconds")
    analyze_path(path_euclidean, elevation_grid, elevation_weight=3.0, slope_weight=1.0, max_slope=0.8, elevation_threshold=0.6)
    plot_path_analysis(path_euclidean, elevation_grid)
    total_euclidean = calculate_total_distance(path_euclidean, lambda n1, n2: euclidean_heuristic_2_5d(n1, n2, elevation_grid))
    print(f"Total Euclidean Distance: {total_euclidean:.4f}")
    plot_paths(elevation_grid, path_euclidean)

# Test with Manhattan Objective
path_manhattan, time_manhattan = plan_2_5d_with_slope_constraints(elevation_grid, turning_radius=0.05, solve_time=solve_time, objective_type="Manhattan")
if path_manhattan:
    print(f"Manhattan Objective - Path found in {time_manhattan:.4f} seconds")
    analyze_path(path_manhattan, elevation_grid, elevation_weight=3.0, slope_weight=1.0, max_slope=0.8, elevation_threshold=0.6)
    plot_path_analysis(path_manhattan, elevation_grid)
    total_manhattan = calculate_total_distance(path_manhattan, lambda n1, n2: manhattan_heuristic_2_5d(n1, n2, elevation_grid))
    print(f"Total Manhattan Distance: {total_manhattan:.4f}")
    plot_paths(elevation_grid, path_manhattan)

# Test with Geodesic Objective
path_geodesic, time_geodesic = plan_2_5d_with_slope_constraints(elevation_grid, turning_radius=0.05, solve_time=solve_time, objective_type="Geodesic")
if path_geodesic:
    print(f"Geodesic Objective - Path found in {time_geodesic:.4f} seconds")
    analyze_path(path_geodesic, elevation_grid, elevation_weight=3.0, slope_weight=1.0, max_slope=0.8, elevation_threshold=0.6)
    plot_path_analysis(path_geodesic, elevation_grid)
    total_geodesic = calculate_total_distance(path_geodesic, lambda n1, n2: geodesic_heuristic_25d(n1, n2, elevation_grid))
    print(f"Total Geodesic Distance: {total_geodesic:.4f}")
    plot_paths(elevation_grid, path_geodesic)

else:
    print("No solution found.")
