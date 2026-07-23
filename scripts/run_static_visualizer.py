#!/usr/bin/env python3

"""Static DSG Visualizer - Visualize a Dynamic Scene Graph from JSON using Rerun."""

import argparse
import numpy as np
import rerun as rr
from pathlib import Path
import textwrap
import traceback
from scipy.spatial.transform import Rotation
import re

import spark_dsg
from spark_dsg import (
	DynamicSceneGraph,
	DsgLayers,
	NodeSymbol,
	SceneGraphNode,
	LayerView,
	BoundingBoxType
)

from daaam.utils.static_visualizer import StaticDSGVisualizer

# Layer ID to name mapping - use string layer names directly
LAYER_NAMES = {
	DsgLayers.OBJECTS: "objects_agents",
	DsgLayers.PLACES: "places",
	DsgLayers.ROOMS: "rooms",
	DsgLayers.BUILDINGS: "buildings",
	DsgLayers.AGENTS: "objects_agents",
}

# Get numeric IDs for color mapping
OBJECTS_ID = DsgLayers.name_to_layer_id(DsgLayers.OBJECTS)
PLACES_ID = DsgLayers.name_to_layer_id(DsgLayers.PLACES)
ROOMS_ID = DsgLayers.name_to_layer_id(DsgLayers.ROOMS)
BUILDINGS_ID = DsgLayers.name_to_layer_id(DsgLayers.BUILDINGS)
AGENTS_ID = DsgLayers.name_to_layer_id(DsgLayers.AGENTS)

LAYER_COLORS = {
	OBJECTS_ID: [255, 0, 0],
	PLACES_ID: [0, 255, 0],
	ROOMS_ID: [0, 0, 255],
	BUILDINGS_ID: [255, 255, 0],
	AGENTS_ID: [255, 0, 255],
}

def parse_arguments():
	parser = argparse.ArgumentParser(
		description="Visualize a Dynamic Scene Graph from JSON using Rerun",
		epilog="Example: python static_visualizer.py --dsg output/dsg.json --log-object-meshes"
	)
	parser.add_argument(
		"--dsg",
		type=str,
		default="/path/to/clustered_dsg_with_summaries.json",
		help="Path to the DSG JSON file"
	)
	parser.add_argument(
		"--color-map",
		type=str,
		default=None,
		help="Path to the color map CSV file (optional)"
	)
	parser.add_argument(
		"--gt-dsgs",
		type=str,
		default="",
		help="Path to ground truth DSG directory or single JSON file (optional)"
	)
	parser.add_argument(
		"--log-object-meshes",
		action="store_true",
		default=True,
		help="Log individual object meshes (enabled by default)"
	)
	parser.add_argument(
		"--no-log-object-meshes",
		dest="log_object_meshes",
		action="store_false",
		help="Disable individual object meshes"
	)
	parser.add_argument(
		"--main-mesh-opacity",
		type=float,
		default=0.90,
		help="Main scene mesh opacity in [0,1] (default: 0.90)"
	)
	parser.add_argument(
		"--object-mesh-colors",
		choices=["semantic", "rgb"],
		default="rgb",
		help="Color object meshes by semantic ID or source RGB (default: rgb)"
	)
	parser.add_argument(
		"--main-mesh-min-component-area-m2",
		type=float,
		default=0.005,
		help="Hide disconnected mesh fragments below this area (default: 0.005)"
	)
	parser.add_argument(
		"--no-focused-blueprint",
		dest="focused_blueprint",
		action="store_false",
		default=True,
		help="Disable the clean Semantic Map / DSG Debug tab layout"
	)
	parser.add_argument(
		"--show-node-labels",
		action="store_true",
		default=False,
		help="Show all DSG node labels (off by default to avoid occlusion)"
	)
	parser.add_argument(
		"--no-semantic-sidecar",
		dest="show_semantic_sidecar",
		action="store_false",
		default=True,
		help="Hide spatial-only DAM entities from dsg_updated.semantic.json"
	)
	parser.add_argument(
		"--object-image-card-scope",
		choices=["none", "mesh-bound", "all"],
		default="none",
		help=(
			"Show floating FastSAM RGB cards for no objects, mesh-bound objects, "
			"or all queryable objects (default: none; clouds are spatially accurate)"
		)
	)
	parser.add_argument(
		"--object-mask-cloud-scope",
		choices=["none", "mesh-bound", "all"],
		default="all",
		help=(
			"Show world-aligned FastSAM RGB-D mask clouds for no objects, "
			"mesh-bound objects, or all queryable objects (default: all)"
		)
	)
	parser.add_argument(
		"--mask-cloud-point-radius-m",
		type=float,
		default=None,
		help="Optional world-space RGB-D mask point radius; overrides screen sizing"
	)
	parser.add_argument(
		"--mask-cloud-point-size-ui",
		type=float,
		default=1.25,
		help="Fixed RGB-D mask point radius in UI points (default: 1.25)"
	)
	parser.add_argument(
		"--dense-map",
		type=Path,
		help=(
			"Accepted direct RGB-D fusion PLY; by default it is discovered from "
			"the checksum-bound evidence provenance"
		),
	)
	parser.add_argument(
		"--no-dense-map",
		dest="enable_dense_map",
		action="store_false",
		default=True,
		help="Disable the dense RGB-D overview tab",
	)
	parser.add_argument(
		"--dense-map-point-radius-m",
		type=float,
		default=None,
		help="Optional world-space dense point radius; overrides screen sizing",
	)
	parser.add_argument(
		"--dense-map-point-size-ui",
		type=float,
		default=1.0,
		help="Fixed dense-map point radius in UI points (default: 1.0)",
	)
	parser.add_argument(
		"--image-card-scale",
		type=float,
		default=0.75,
		help="Image-card height relative to object size (default: 0.75)"
	)
	parser.add_argument(
		"--image-card-max-height-m",
		type=float,
		default=1.0,
		help="Maximum image-card height in meters (default: 1.0)"
	)
	parser.add_argument(
		"--image-card-grid-step-px",
		type=int,
		default=8,
		help="Pixel grid step for mask-shaped texture meshes (default: 8)"
	)
	parser.add_argument(
		"--spawn",
		action="store_true",
		default=True,
		help="Spawn Rerun viewer (default: True)"
	)
	parser.add_argument(
		"--no-spawn",
		dest="spawn",
		action="store_false",
		help="Don't spawn Rerun viewer"
	)
	parser.add_argument(
		"--connect",
		type=str,
		default="",
		help=(
			"Connect to an existing Rerun gRPC server instead of spawning the "
			"native viewer (for example, rerun+http://127.0.0.1:9876/proxy)"
		)
	)
	parser.add_argument(
		"--save-rrd",
		type=str,
		default="",
		help="Save the recording and blueprint to an .rrd file instead of spawning",
	)
	parser.add_argument(
		"--z-offset-objects",
		type=float,
		default=0.0,
		help="Z-offset for objects layer in meters (default: 0.0)"
	)
	parser.add_argument(
		"--z-offset-places",
		type=float,
		default=10.0,
		help="Z-offset for places/traversability layer in meters (default: 10.0)"
	)
	parser.add_argument(
		"--z-offset-rooms",
		type=float,
		default=20.0,
		help="Z-offset for rooms layer in meters (default: 20.0)"
	)
	parser.add_argument(
		"--z-offset-buildings",
		type=float,
		default=40.0,
		help="Z-offset for buildings layer in meters (default: 40.0)"
	)
	parser.add_argument(
		"--z-offset-gt",
		type=float,
		default=0.0,
		help="Z-offset for ground truth objects in meters (default: 0.0)"
	)
	parser.add_argument(
		"--interlayer-edge-subsample",
		type=int,
		default=1,
		help="Show every Nth interlayer edge (default: 1 = all edges)"
	)
	parser.add_argument(
		"--object-subsample-grid-size",
		type=float,
		default=None,
		help="Grid size in meters for spatial object downsampling (default: None = disabled)"
	)
	parser.add_argument(
		"--log-regions-separately",
		action="store_true",
		default=False,
		help="Log each region (room + traversability) to separate entity paths for independent coloring"
	)

	args = parser.parse_args()
	return args

def main():
	"""Main entry point."""
	args = parse_arguments()
	
	# Check if file exists
	if not Path(args.dsg).exists():
		print(f"Error: DSG file not found: {args.dsg}")
		exit(1)

	# Build layer z-offsets dict from command-line arguments
	layer_z_offsets = {
		OBJECTS_ID: args.z_offset_objects,
		(3, 2): args.z_offset_places,  # TRAVERSABILITY layer (layer 3, partition 2)
		ROOMS_ID: args.z_offset_rooms,
		BUILDINGS_ID: args.z_offset_buildings,
		AGENTS_ID: args.z_offset_objects,  # Share offset with objects
		10: args.z_offset_gt,  # GT_OBJECTS layer
	}

	# Create and run visualizer
	visualizer = StaticDSGVisualizer(
		args.dsg,
		gt_dsg_path=args.gt_dsgs,
		color_map_path=args.color_map,
		log_object_meshes=args.log_object_meshes,
		spawn=args.spawn,
		connect_url=args.connect or None,
		save_path=args.save_rrd or None,
		layer_z_offsets=layer_z_offsets,
		interlayer_edge_subsample=args.interlayer_edge_subsample,
		object_subsample_grid_size=args.object_subsample_grid_size,
		log_regions_separately=args.log_regions_separately,
		main_mesh_opacity=args.main_mesh_opacity,
		object_mesh_color_mode=args.object_mesh_colors,
		show_semantic_sidecar=args.show_semantic_sidecar,
		object_image_card_scope=args.object_image_card_scope,
		object_mask_cloud_scope=args.object_mask_cloud_scope,
		image_card_scale=args.image_card_scale,
		image_card_max_height_m=args.image_card_max_height_m,
		image_card_grid_step_px=args.image_card_grid_step_px,
		mask_cloud_point_radius_m=args.mask_cloud_point_radius_m,
		mask_cloud_point_size_ui=args.mask_cloud_point_size_ui,
		main_mesh_min_component_area_m2=args.main_mesh_min_component_area_m2,
		focused_blueprint=args.focused_blueprint,
		show_node_labels=args.show_node_labels,
		dense_map_path=args.dense_map,
		enable_dense_map=args.enable_dense_map,
		dense_map_point_radius_m=args.dense_map_point_radius_m,
		dense_map_point_size_ui=args.dense_map_point_size_ui,
	)
	visualizer.visualize()
	if args.connect:
		rr.disconnect()
	if args.save_rrd:
		rr.disconnect()
	
	# Keep the script running if spawned
	if args.spawn and not args.connect and not args.save_rrd:
		print("\nVisualization ready. Press Ctrl+C to exit.")
		try:
			import time
			while True:
				time.sleep(1)
		except KeyboardInterrupt:
			print("\nExiting...")


if __name__ == "__main__":
	main()
