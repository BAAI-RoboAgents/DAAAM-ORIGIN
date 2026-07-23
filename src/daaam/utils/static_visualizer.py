#!/usr/bin/env python3

"""Static DSG Visualizer - Visualize a Dynamic Scene Graph from JSON using Rerun."""

import argparse
import colorsys
import cv2
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from pathlib import Path
import textwrap
import traceback
import json
import yaml
import re

from scipy.spatial.transform import Rotation
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

import spark_dsg
from spark_dsg import (
	DynamicSceneGraph,
	DsgLayers,
	NodeSymbol,
	SceneGraphNode,
	LayerView,
	BoundingBoxType
)
from daaam.query_index import QueryIndexError, load_query_index
from daaam.query_evidence import (
	QueryEvidenceError,
	load_query_evidence,
	sha256_file,
)


# Layer ID to name mapping - use string layer names directly
LAYER_NAMES = {
	DsgLayers.OBJECTS: "objects_agents",
	DsgLayers.PLACES: "places",
	DsgLayers.ROOMS: "rooms",
	DsgLayers.BUILDINGS: "buildings",
	DsgLayers.AGENTS: "objects_agents",
	"BACKGROUND_OBJECTS": "background_objects",  # Custom layer for background objects
	"GT_OBJECTS": "gt_objects",  # GT_OBJECTS layer (numeric ID from create_coda_gt_object_scene_graph.py)
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
	10: [0, 255, 255],  # Cyan for GT objects
}


def masked_image_card_geometry(alpha, width_m, height_m, center, face_direction, grid_step_px=8):
	"""Build a camera-facing textured grid only where a FastSAM alpha mask exists."""
	alpha = np.asarray(alpha, dtype=np.uint8)
	center = np.asarray(center, dtype=np.float32).reshape(-1)
	face_direction = np.asarray(face_direction, dtype=np.float32).reshape(-1)
	if alpha.ndim != 2 or min(alpha.shape) < 2:
		raise ValueError("image-card alpha mask must be at least 2x2")
	if center.size != 3 or not np.isfinite(center).all():
		raise ValueError("image-card center must be a finite 3D point")
	if face_direction.size != 2 or not np.isfinite(face_direction).all():
		raise ValueError("image-card face direction must be a finite XY vector")
	if not np.isfinite(width_m) or not np.isfinite(height_m) or width_m <= 0 or height_m <= 0:
		raise ValueError("image-card dimensions must be finite and positive")
	if int(grid_step_px) < 1:
		raise ValueError("image-card grid step must be positive")

	direction_norm = float(np.linalg.norm(face_direction))
	if direction_norm < 1.0e-6:
		face_direction = np.asarray([0.0, -1.0], dtype=np.float32)
	else:
		face_direction = face_direction / direction_norm
	right = np.asarray([face_direction[1], -face_direction[0], 0.0], dtype=np.float32)
	up = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
	height_px, width_px = alpha.shape
	x_samples = list(range(0, width_px - 1, int(grid_step_px))) + [width_px - 1]
	y_samples = list(range(0, height_px - 1, int(grid_step_px))) + [height_px - 1]
	x_samples = sorted(set(x_samples))
	y_samples = sorted(set(y_samples))
	texcoords = []
	vertices = []
	for y in y_samples:
		v = float(y) / float(height_px - 1)
		for x in x_samples:
			u = float(x) / float(width_px - 1)
			vertices.append(
				center
				+ right * ((u - 0.5) * float(width_m))
				+ up * ((0.5 - v) * float(height_m))
			)
			texcoords.append([u, v])

	column_count = len(x_samples)
	faces = []
	for row in range(len(y_samples) - 1):
		for column in range(len(x_samples) - 1):
			x1, x2 = x_samples[column], x_samples[column + 1]
			y1, y2 = y_samples[row], y_samples[row + 1]
			coverage = float(np.mean(alpha[y1 : y2 + 1, x1 : x2 + 1] >= 32))
			if coverage < 0.12:
				continue
			a = row * column_count + column
			b = a + 1
			c = a + column_count + 1
			d = a + column_count
			faces.extend(([a, b, c], [a, c, d], [c, b, a], [d, c, a]))
	if not faces:
		raise ValueError("image-card mask produced no textured cells")
	return (
		np.asarray(vertices, dtype=np.float32),
		np.asarray(faces, dtype=np.uint32),
		np.asarray(texcoords, dtype=np.float32),
	)


def filter_mesh_faces_by_component_area(
	vertices, faces, *, minimum_area_m2=0.005, weld_tolerance_m=1.0e-4
):
	"""Remove small disconnected mesh fragments after tolerance-based welding."""
	vertex_array = np.asarray(vertices, dtype=np.float64)
	face_array = np.asarray(faces, dtype=np.int64)
	if vertex_array.ndim != 2 or vertex_array.shape[1] != 3:
		raise ValueError("mesh vertices must be Nx3")
	if face_array.ndim != 2 or face_array.shape[1] != 3:
		raise ValueError("mesh faces must be Nx3")
	if minimum_area_m2 < 0.0 or weld_tolerance_m <= 0.0:
		raise ValueError("mesh component filtering parameters are invalid")
	if not len(face_array) or minimum_area_m2 == 0.0:
		return face_array.astype(np.uint32, copy=False), {
			"components": 0,
			"removed_faces": 0,
			"removed_area_m2": 0.0,
		}
	quantized = np.rint(vertex_array / float(weld_tolerance_m)).astype(np.int64)
	_, inverse = np.unique(quantized, axis=0, return_inverse=True)
	welded_faces = inverse[face_array]
	valid = (
		(welded_faces[:, 0] != welded_faces[:, 1])
		& (welded_faces[:, 1] != welded_faces[:, 2])
		& (welded_faces[:, 0] != welded_faces[:, 2])
	)
	edges = welded_faces[valid]
	rows = np.concatenate(
		[edges[:, 0], edges[:, 1], edges[:, 2], edges[:, 1], edges[:, 2], edges[:, 0]]
	)
	columns = np.concatenate(
		[edges[:, 1], edges[:, 2], edges[:, 0], edges[:, 0], edges[:, 1], edges[:, 2]]
	)
	vertex_count = int(inverse.max()) + 1
	graph = coo_matrix(
		(np.ones(len(rows), dtype=np.uint8), (rows, columns)),
		shape=(vertex_count, vertex_count),
	).tocsr()
	component_count, labels = connected_components(graph, directed=False)
	triangles = vertex_array[face_array]
	areas = 0.5 * np.linalg.norm(
		np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
		axis=1,
	)
	component_areas = np.bincount(
		labels[welded_faces[:, 0]], weights=areas, minlength=component_count
	)
	keep = valid & (
		component_areas[labels[welded_faces[:, 0]]] >= float(minimum_area_m2)
	)
	return face_array[keep].astype(np.uint32, copy=False), {
		"components": int(component_count),
		"removed_faces": int(np.count_nonzero(~keep)),
		"removed_area_m2": float(np.sum(areas[~keep])),
	}


def discover_dense_rgbd_map(dsg_path):
	"""Find the accepted direct RGB-D fusion that produced this DSG's evidence."""
	dsg_path = Path(dsg_path).expanduser().resolve()
	manifest_path = dsg_path.with_suffix(".evidence.json")
	if not manifest_path.is_file():
		return None
	try:
		manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
		source = manifest["source"]
		pose_path = Path(source["camera_pose_file"]).expanduser().resolve()
		expected_pose_sha256 = str(source["camera_pose_file_sha256"])
	except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
		return None
	try:
		pose_valid = (
			pose_path.is_file() and sha256_file(pose_path) == expected_pose_sha256
		)
	except OSError:
		return None
	if not pose_valid:
		return None
	try:
		run_root = pose_path.parents[2]
	except IndexError:
		return None
	visualization_dir = run_root / "11_dense_rgbd_visualization_candidate"
	acceptance_path = visualization_dir / "visualization_acceptance.json"
	try:
		acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
		artifacts = acceptance["artifacts"]
		visualization_cloud = visualization_dir / "direct_rgbd_fusion.ply"
		visualization_report = visualization_dir / "direct_rgbd_fusion_report.json"
		visualization_valid = (
			acceptance.get("schema")
			== "daaam.dense_rgbd_visualization_acceptance.v1"
			and acceptance.get("accepted") is True
			and acceptance.get("source_pose_sha256") == expected_pose_sha256
			and artifacts.get("point_cloud") == visualization_cloud.name
			and artifacts.get("report") == visualization_report.name
			and visualization_cloud.is_file()
			and visualization_report.is_file()
			and sha256_file(visualization_cloud)
			== artifacts.get("point_cloud_sha256")
			and sha256_file(visualization_report) == artifacts.get("report_sha256")
		)
	except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
		visualization_valid = False
	if visualization_valid:
		return visualization_cloud
	fusion_dir = run_root / "10_direct_rgbd_fusion"
	report_path = fusion_dir / "direct_rgbd_fusion_report.json"
	point_cloud_path = fusion_dir / "direct_rgbd_fusion.ply"
	try:
		report = json.loads(report_path.read_text(encoding="utf-8"))
		mapping_run = json.loads(
			(run_root / "mapping_run.json").read_text(encoding="utf-8")
		)
	except (OSError, TypeError, json.JSONDecodeError):
		return None
	accepted = mapping_run.get("direct_rgbd_fusion", {}).get("manually_accepted")
	if accepted is not True or not point_cloud_path.is_file():
		return None
	if Path(str(report.get("ply", ""))).expanduser().resolve() != point_cloud_path:
		return None
	return point_cloud_path

class StaticDSGVisualizer:
	"""Visualizes a static DSG using Rerun."""
	
	def __init__(self, dsg_path, gt_dsg_path=None, color_map_path=None, log_object_meshes=True, spawn=True, layer_z_offsets=None, interlayer_edge_subsample=1, object_subsample_grid_size=None, log_regions_separately=False, connect_url=None, save_path=None, main_mesh_opacity=0.90, object_mesh_color_mode="rgb", show_semantic_sidecar=True, object_image_card_scope="none", object_mask_cloud_scope="all", image_card_scale=0.75, image_card_max_height_m=1.0, image_card_grid_step_px=8, mask_cloud_point_radius_m=None, mask_cloud_point_size_ui=1.25, main_mesh_min_component_area_m2=0.005, focused_blueprint=True, show_node_labels=False, dense_map_path=None, enable_dense_map=True, dense_map_point_radius_m=None, dense_map_point_size_ui=1.0):
		"""Initialize the visualizer."""
		self.dsg_path = Path(dsg_path)
		self.log_object_meshes = log_object_meshes
		if not 0.0 <= float(main_mesh_opacity) <= 1.0:
			raise ValueError("main_mesh_opacity must be in [0, 1]")
		if object_mesh_color_mode not in {"semantic", "rgb"}:
			raise ValueError("object_mesh_color_mode must be semantic or rgb")
		if object_image_card_scope not in {"none", "mesh-bound", "all"}:
			raise ValueError("object_image_card_scope must be none, mesh-bound, or all")
		if object_mask_cloud_scope not in {"none", "mesh-bound", "all"}:
			raise ValueError("object_mask_cloud_scope must be none, mesh-bound, or all")
		if not np.isfinite(image_card_scale) or float(image_card_scale) <= 0.0:
			raise ValueError("image_card_scale must be finite and positive")
		if not np.isfinite(image_card_max_height_m) or float(image_card_max_height_m) <= 0.0:
			raise ValueError("image_card_max_height_m must be finite and positive")
		if int(image_card_grid_step_px) < 1:
			raise ValueError("image_card_grid_step_px must be positive")
		if mask_cloud_point_radius_m is not None and (
			not np.isfinite(mask_cloud_point_radius_m)
			or float(mask_cloud_point_radius_m) <= 0.0
		):
			raise ValueError("mask_cloud_point_radius_m must be finite and positive")
		if not np.isfinite(mask_cloud_point_size_ui) or float(mask_cloud_point_size_ui) <= 0.0:
			raise ValueError("mask_cloud_point_size_ui must be finite and positive")
		if not np.isfinite(main_mesh_min_component_area_m2) or float(main_mesh_min_component_area_m2) < 0.0:
			raise ValueError("main_mesh_min_component_area_m2 must be finite and non-negative")
		if dense_map_point_radius_m is not None and (
			not np.isfinite(dense_map_point_radius_m)
			or float(dense_map_point_radius_m) <= 0.0
		):
			raise ValueError("dense_map_point_radius_m must be finite and positive")
		if not np.isfinite(dense_map_point_size_ui) or float(dense_map_point_size_ui) <= 0.0:
			raise ValueError("dense_map_point_size_ui must be finite and positive")
		self.main_mesh_opacity = float(main_mesh_opacity)
		self.object_mesh_color_mode = object_mesh_color_mode
		self.object_image_card_scope = object_image_card_scope
		self.object_mask_cloud_scope = object_mask_cloud_scope
		self.image_card_scale = float(image_card_scale)
		self.image_card_max_height_m = float(image_card_max_height_m)
		self.image_card_grid_step_px = int(image_card_grid_step_px)
		self.mask_cloud_point_radius_m = (
			None
			if mask_cloud_point_radius_m is None
			else float(mask_cloud_point_radius_m)
		)
		self.mask_cloud_point_size_ui = float(mask_cloud_point_size_ui)
		self.main_mesh_min_component_area_m2 = float(main_mesh_min_component_area_m2)
		self.focused_blueprint = bool(focused_blueprint)
		self.show_node_labels = bool(show_node_labels)
		self.enable_dense_map = bool(enable_dense_map)
		self.dense_map_point_radius_m = (
			None
			if dense_map_point_radius_m is None
			else float(dense_map_point_radius_m)
		)
		self.dense_map_point_size_ui = float(dense_map_point_size_ui)
		self.dense_map_path = None
		if self.enable_dense_map:
			if dense_map_path is not None:
				candidate = Path(dense_map_path).expanduser().resolve()
				if not candidate.is_file():
					raise FileNotFoundError(f"dense RGB-D map not found: {candidate}")
				self.dense_map_path = candidate
			else:
				self.dense_map_path = discover_dense_rgbd_map(self.dsg_path)
		self.dense_map_point_count = 0
		self.color_map = load_color_map(color_map_path)
		self.object_subsample_grid_size = object_subsample_grid_size  # None = disabled, float = grid size in meters
		self.log_regions_separately = log_regions_separately  # Log each region to separate entity path

		# Set default z-offsets for layers
		default_offsets = {
			OBJECTS_ID: 0.0,
			(3, 2): 10.0,  # TRAVERSABILITY layer (layer 3, partition 2)
			ROOMS_ID: 20.0,
			BUILDINGS_ID: 40.0,
			AGENTS_ID: 0.0,
			10: 0.0,  # GT_OBJECTS
		}
		self.layer_z_offsets = layer_z_offsets or default_offsets
		self.interlayer_edge_subsample = max(1, interlayer_edge_subsample)  # Prevent div by zero

		# Background objects are now handled as a layer in the scene graph
		# No need to load background_objects.yaml separately

		# Initialize Rerun. A remote sink is useful when the native viewer cannot
		# start (for example, when displaying through Rerun's web viewer).
		rr.init("static_dsg_visualizer", spawn=spawn and not connect_url and not save_path)
		if save_path:
			print(f"Saving Rerun recording to {save_path}")
			rr.save(str(Path(save_path).expanduser().resolve()))
		if connect_url:
			print(f"Connecting to Rerun server at {connect_url}")
			rr.connect_grpc(connect_url)
		rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

		# Load the DSG
		print(f"Loading DSG from {self.dsg_path}")
		self.dsg = DynamicSceneGraph.load(str(self.dsg_path))
		print(f"DSG loaded successfully")
		self.semantic_sidecar_records = []
		self.query_evidence_by_node = {}
		if show_semantic_sidecar:
			try:
				self.semantic_sidecar_records = load_query_index(
					self.dsg_path, allow_conventional_path=True
				)
			except QueryIndexError as error:
				print(f"Semantic sidecar ignored: {error}")
		self.object_image_cards = self._load_object_image_cards()

		# Get statistics
		self._print_dsg_stats()

		# Log visualization settings
		self._log_visualization_settings()

		# Load ground truth DSG(s) if provided
		self.gt_dsgs = []  # List of (sequence_id, DynamicSceneGraph) tuples
		if gt_dsg_path:
			gt_path = Path(gt_dsg_path)
			if gt_path.exists():
				if gt_path.is_dir():
					# Load all GT DSGs from directory
					print(f"\nLoading ground truth DSGs from directory: {gt_path}")
					self._load_gt_dsgs_from_directory(gt_path)
				elif gt_path.is_file() and gt_path.suffix == '.json':
					# Load single GT DSG file
					print(f"\nLoading ground truth DSG from {gt_path}")
					dsg = DynamicSceneGraph.load(str(gt_path))
					# Extract sequence ID from filename (assume format: <seq_id>/gt_scene_graph.json)
					seq_id = gt_path.parent.name if gt_path.name == 'gt_scene_graph.json' else 'unknown'
					self.gt_dsgs.append((seq_id, dsg))
					print(f"GT DSG loaded successfully")

				if self.gt_dsgs:
					self._print_gt_dsg_stats()

	def _load_object_image_cards(self):
		"""Join checksum-validated FastSAM cutouts to DSG/sidecar geometry."""
		if (
			self.object_image_card_scope == "none"
			and self.object_mask_cloud_scope == "none"
			and not self.semantic_sidecar_records
		):
			return []
		try:
			evidence_by_node, _ = load_query_evidence(self.dsg_path)
		except QueryEvidenceError as error:
			print(f"Object image cards ignored: {error}")
			return []
		self.query_evidence_by_node = evidence_by_node
		if not evidence_by_node:
			return []

		geometry = {}
		if self.dsg.has_layer(DsgLayers.OBJECTS):
			for node in self.dsg.get_layer(DsgLayers.OBJECTS).nodes:
				attributes = node.attributes
				try:
					mesh = attributes.mesh()
					has_mesh = bool(mesh is not None and mesh.num_vertices() > 4)
				except (AttributeError, RuntimeError, TypeError):
					has_mesh = False
				position = np.asarray(attributes.position, dtype=np.float32).reshape(-1).copy()
				dimensions = np.asarray(
					attributes.bounding_box.dimensions, dtype=np.float32
				).reshape(-1).copy()
				metadata = {}
				if hasattr(attributes, "metadata"):
					try:
						metadata = dict(attributes.metadata.get() or {})
					except (AttributeError, RuntimeError, TypeError, ValueError):
						metadata = {}
				geometry[str(node.id)] = {
					"position": position,
					"dimensions": dimensions,
					"geometry_status": "mesh_bound" if has_mesh else "spatial_only",
					"description": str(metadata.get("description", "")).strip(),
				}
		for record in self.semantic_sidecar_records:
			geometry[str(record["record_id"])] = {
				"position": np.asarray(record.get("position_m", []), dtype=np.float32),
				"dimensions": np.asarray(record.get("dimensions_m", []), dtype=np.float32),
				"geometry_status": str(record["geometry_status"]),
				"description": str(record["description"]),
			}

		cards = []
		for node_id, evidence in sorted(evidence_by_node.items()):
			item = geometry.get(node_id)
			if item is None or (
				evidence.cutout_path is None and evidence.point_cloud_path is None
			):
				continue
			card_in_scope = self.object_image_card_scope == "all" or (
				self.object_image_card_scope == "mesh-bound"
				and item["geometry_status"] == "mesh_bound"
			)
			cloud_in_scope = self.object_mask_cloud_scope == "all" or (
				self.object_mask_cloud_scope == "mesh-bound"
				and item["geometry_status"] == "mesh_bound"
			)
			if not card_in_scope and not cloud_in_scope:
				continue
			position = np.asarray(item["position"], dtype=np.float32).reshape(-1)
			dimensions = np.asarray(item["dimensions"], dtype=np.float32).reshape(-1)
			if position.size != 3 or not np.isfinite(position).all():
				continue
			if (
				dimensions.size != 3
				or not np.isfinite(dimensions).all()
				or np.any(dimensions <= 0.0)
			):
				dimensions = np.asarray([0.4, 0.4, 0.4], dtype=np.float32)
			cards.append(
				{
					"evidence": evidence,
					"position": position,
					"dimensions": dimensions,
					"geometry_status": item["geometry_status"],
					"description": item["description"],
				}
			)
		return cards

	def _object_overlay_count(self, scope, *, artifact):
		if scope == "none":
			return 0
		return sum(
			(scope == "all" or card["geometry_status"] == "mesh_bound")
			and getattr(card["evidence"], artifact) is not None
			for card in self.object_image_cards
		)
	
	def _print_dsg_stats(self):
		"""Print statistics about the loaded DSG."""
		print("\n=== DSG Statistics ===")
		print(f"Has mesh: {self.dsg.has_mesh()}")
		
		if self.dsg.has_mesh():
			mesh = self.dsg.mesh
			vertices = mesh.get_vertices()
			faces = mesh.get_faces()
			print(f"Main mesh: {vertices.shape[1]} vertices, {faces.shape[1]} faces")
		
		# Count nodes by layer
		for layer_str, layer_name in LAYER_NAMES.items():
			if self.dsg.has_layer(layer_str):
				layer = self.dsg.get_layer(layer_str)
				print(f"{layer_name}: {layer.num_nodes()} nodes")
		
		# Count edges
		all_edges = list(self.dsg.edges)
		print(f"Total edges: {len(all_edges)}")
		if self.semantic_sidecar_records:
			counts = {
				status: sum(
					record["geometry_status"] == status
					for record in self.semantic_sidecar_records
				)
				for status in ("spatial_only", "image_only")
			}
			print(
				"Semantic sidecar: "
				f"{len(self.semantic_sidecar_records)} records "
				f"(spatial_only={counts['spatial_only']}, "
				f"image_only={counts['image_only']})"
			)
		print(
			f"Object image cards: "
			f"{self._object_overlay_count(self.object_image_card_scope, artifact='cutout_path')} "
			f"(scope={self.object_image_card_scope})"
		)
		print(
			f"Object RGB-D mask clouds: "
			f"{self._object_overlay_count(self.object_mask_cloud_scope, artifact='point_cloud_path')} "
			f"(scope={self.object_mask_cloud_scope})"
		)
		print(f"Dense RGB-D overview: {self.dense_map_path or 'not available'}")
		print("=" * 22)

	def _log_visualization_settings(self):
		"""Log active visualization settings."""
		print("\n=== Visualization Settings ===")
		print("Layer Z-Offsets:")
		# Create a mapping from layer ID to layer name for display
		layer_id_to_name = {
			OBJECTS_ID: "OBJECTS",
			(3, 2): "TRAVERSABILITY",
			ROOMS_ID: "ROOMS",
			BUILDINGS_ID: "BUILDINGS",
			AGENTS_ID: "AGENTS",
			10: "GT_OBJECTS"
		}
		for layer_id, offset in sorted(self.layer_z_offsets.items(), key=lambda x: (str(type(x[0])), x[0])):
			layer_name = layer_id_to_name.get(layer_id, f"Layer_{layer_id}")
			print(f"  {layer_name}: {offset:.1f}m")
		print(f"Inter-layer edge subsampling: 1/{self.interlayer_edge_subsample}")
		print(f"Log regions separately: {self.log_regions_separately}")
		print(f"Log object meshes: {self.log_object_meshes}")
		print(f"Object mesh colors: {self.object_mesh_color_mode}")
		print(f"Main mesh opacity: {self.main_mesh_opacity:.2f}")
		print(
			f"Main mesh minimum component area: "
			f"{self.main_mesh_min_component_area_m2:.4f}m^2"
		)
		print(f"Focused map blueprint: {self.focused_blueprint}")
		print(f"Node labels: {self.show_node_labels}")
		print(
			f"Dense RGB-D overview: {bool(self.dense_map_path)} "
			f"({self._point_size_description(self.dense_map_point_radius_m, self.dense_map_point_size_ui)})"
		)
		print(f"Semantic sidecar overlay: {bool(self.semantic_sidecar_records)}")
		print(
			f"FastSAM image cards: "
			f"{self._object_overlay_count(self.object_image_card_scope, artifact='cutout_path')} "
			f"(scope={self.object_image_card_scope}, scale={self.image_card_scale:.2f}, "
			f"max_height={self.image_card_max_height_m:.2f}m)"
		)
		print(
			f"FastSAM RGB-D mask clouds: "
			f"{self._object_overlay_count(self.object_mask_cloud_scope, artifact='point_cloud_path')} "
			f"(scope={self.object_mask_cloud_scope}, "
			f"{self._point_size_description(self.mask_cloud_point_radius_m, self.mask_cloud_point_size_ui)})"
		)
		print("=" * 31)

	@staticmethod
	def _point_size_description(world_radius_m, ui_size):
		if world_radius_m is None:
			return f"screen_radius={ui_size:.2f}pt"
		return f"world_radius={world_radius_m:.3f}m"

	@staticmethod
	def _rerun_point_radius(world_radius_m, ui_size):
		if world_radius_m is None:
			return rr.Radius.ui_points(ui_size)
		return world_radius_m

	def _get_layer_z_offset(self, layer_id) -> float:
		"""Get z-offset for a given layer ID (LayerKey, int, or tuple)."""
		# Import LayerKey for type checking
		try:
			from spark_dsg import LayerKey
			is_layer_key = isinstance(layer_id, LayerKey)
		except ImportError:
			is_layer_key = False

		# Try direct lookup first (for int keys, LayerKey keys if dict was set up that way)
		if layer_id in self.layer_z_offsets:
			return self.layer_z_offsets[layer_id]

		# If it's a LayerKey object, convert to tuple and try lookup
		if is_layer_key:
			layer_tuple = (layer_id.layer, layer_id.partition)
			if layer_tuple in self.layer_z_offsets:
				return self.layer_z_offsets[layer_tuple]

			# Also try just the layer number (ignoring partition)
			if layer_id.layer in self.layer_z_offsets:
				return self.layer_z_offsets[layer_id.layer]

		# If it's already a tuple, try just the first element (layer number)
		if isinstance(layer_id, tuple) and len(layer_id) >= 1:
			if layer_id[0] in self.layer_z_offsets:
				return self.layer_z_offsets[layer_id[0]]

		# Default fallback
		return 0.0

	def _spatial_downsample_nodes(self, nodes, grid_size=1.0):
		"""Spatially downsample nodes using grid-based approach.

		Args:
			nodes: List of nodes to downsample
			grid_size: Size of grid cells in meters

		Returns:
			List of downsampled nodes (1 per grid cell)
		"""
		if not nodes or grid_size is None:
			return nodes

		grid = {}
		for node in nodes:
			if node.attributes is None:
				continue
			pos = node.attributes.position
			# Compute grid cell
			grid_x = int(pos[0] / grid_size)
			grid_y = int(pos[1] / grid_size)
			grid_key = (grid_x, grid_y)

			# Keep only first node in each cell
			if grid_key not in grid:
				grid[grid_key] = node

		return list(grid.values())

	def _load_gt_dsgs_from_directory(self, gt_dir: Path):
		"""Load all GT DSGs from directory structure."""
		for seq_dir in sorted(gt_dir.iterdir()):
			if seq_dir.is_dir():
				gt_file = seq_dir / "gt_scene_graph.json"
				if gt_file.exists():
					try:
						dsg = DynamicSceneGraph.load(str(gt_file))
						seq_id = seq_dir.name
						self.gt_dsgs.append((seq_id, dsg))
						print(f"  Loaded GT DSG for sequence {seq_id}")
					except Exception as e:
						print(f"  Failed to load {gt_file}: {e}")

	def _print_gt_dsg_stats(self):
		"""Print statistics about all loaded GT DSGs."""
		print("\n=== Ground Truth DSG Statistics ===")
		total_nodes = 0
		for seq_id, gt_dsg in self.gt_dsgs:
			if gt_dsg.has_layer(10):
				layer = gt_dsg.get_layer(10)
				node_count = layer.num_nodes()
				total_nodes += node_count
				print(f"SEQ{seq_id} GT_OBJECTS: {node_count} nodes")
			else:
				print(f"SEQ{seq_id}: No GT_OBJECTS layer found")
		print(f"Total GT objects across all sequences: {total_nodes}")
		print("=" * 36)

	def _collect_region_data(self):
		"""Collect traversability nodes and edges organized by room.

		Returns:
			dict mapping room_id -> {
				'room_node': SceneGraphNode,
				'traversability_nodes': list of nodes,
				'intralayer_edges': list of edges between traversability nodes,
				'interlayer_edges': list of edges from traversability to room
			}
		"""
		import sys

		# Import LayerKey
		try:
			from spark_dsg import LayerKey
			print(f"[REGION_DEBUG] LayerKey imported: {LayerKey}", flush=True)
		except ImportError as e:
			print(f"[REGION_DEBUG] LayerKey import FAILED: {e}", flush=True)
			LayerKey = None
			return {}

		region_data = {}

		# Get all room nodes
		print(f"[REGION_DEBUG] Checking for ROOMS layer...", flush=True)
		if not self.dsg.has_layer(DsgLayers.ROOMS):
			print(f"[REGION_DEBUG] No ROOMS layer found!", flush=True)
			return region_data

		room_layer = self.dsg.get_layer(DsgLayers.ROOMS)
		room_nodes = {node.id: node for node in room_layer.nodes}
		print(f"[REGION_DEBUG] Found {len(room_nodes)} room nodes: {list(room_nodes.keys())[:5]}", flush=True)

		# Get all traversability nodes (layer 3, partition 2)
		traversability_nodes = {}
		total_nodes = 0
		layer_3_nodes = 0
		layer_3_part_2_nodes = 0

		for node in self.dsg.nodes:
			total_nodes += 1
			if LayerKey and isinstance(node.layer, LayerKey):
				if node.layer.layer == 3:
					layer_3_nodes += 1
					if node.layer.partition == 2:
						layer_3_part_2_nodes += 1
						traversability_nodes[node.id] = node
						if layer_3_part_2_nodes <= 3:
							print(f"[REGION_DEBUG] Found trav node {node.id}: layer={node.layer.layer}, partition={node.layer.partition}", flush=True)

		print(f"[REGION_DEBUG] Total nodes: {total_nodes}", flush=True)
		print(f"[REGION_DEBUG] Layer 3 nodes: {layer_3_nodes}", flush=True)
		print(f"[REGION_DEBUG] Layer 3, partition 2 nodes: {layer_3_part_2_nodes}", flush=True)

		if not traversability_nodes:
			print(f"[REGION_DEBUG] No traversability nodes found! Cannot create regions.", flush=True)
			return {}

		# Initialize region data for each room
		for room_id, room_node in room_nodes.items():
			region_data[room_id] = {
				'room_node': room_node,
				'traversability_nodes': [],
				'traversability_node_ids': set(),
				'intralayer_edges': [],
				'interlayer_edges': []
			}

		# Categorize edges and nodes by room
		total_edges = 0
		trav_to_room_edges = 0
		room_to_trav_edges = 0

		for edge in self.dsg.edges:
			total_edges += 1
			source_node = self.dsg.find_node(edge.source)
			target_node = self.dsg.find_node(edge.target)

			if not source_node or not target_node:
				continue

			# Check if edge connects traversability to room
			# Use node.id for membership tests (NodeSymbol objects from edge.source/target don't compare correctly)
			source_is_trav = source_node.id in traversability_nodes
			target_is_trav = target_node.id in traversability_nodes
			source_is_room = source_node.id in room_nodes
			target_is_room = target_node.id in room_nodes

			# Interlayer edge: traversability <-> room
			if source_is_trav and target_is_room:
				trav_to_room_edges += 1
				region_data[target_node.id]['interlayer_edges'].append(edge)
				region_data[target_node.id]['traversability_node_ids'].add(source_node.id)
			elif target_is_trav and source_is_room:
				room_to_trav_edges += 1
				region_data[source_node.id]['interlayer_edges'].append(edge)
				region_data[source_node.id]['traversability_node_ids'].add(target_node.id)

		print(f"[REGION_DEBUG] Total edges: {total_edges}", flush=True)
		print(f"[REGION_DEBUG] Trav->Room edges: {trav_to_room_edges}", flush=True)
		print(f"[REGION_DEBUG] Room->Trav edges: {room_to_trav_edges}", flush=True)

		# Collect intralayer edges
		all_intralayer_edges = []
		for edge in self.dsg.edges:
			source_node = self.dsg.find_node(edge.source)
			target_node = self.dsg.find_node(edge.target)

			if not source_node or not target_node:
				continue

			# Use node.id for membership tests
			source_is_trav = source_node.id in traversability_nodes
			target_is_trav = target_node.id in traversability_nodes

			if source_is_trav and target_is_trav and source_node.layer == target_node.layer:
				all_intralayer_edges.append(edge)

		print(f"[REGION_DEBUG] Total intralayer trav edges: {len(all_intralayer_edges)}", flush=True)

		# Assign intralayer edges to rooms
		for edge in all_intralayer_edges:
			source_node = self.dsg.find_node(edge.source)
			target_node = self.dsg.find_node(edge.target)
			if not source_node or not target_node:
				continue

			# Use node.id for membership tests
			for room_id, data in region_data.items():
				if source_node.id in data['traversability_node_ids'] and target_node.id in data['traversability_node_ids']:
					data['intralayer_edges'].append(edge)
					break  # Edge can only belong to one room

		# Collect actual node objects for each region
		for room_id, data in region_data.items():
			data['traversability_nodes'] = [
				traversability_nodes[node_id]
				for node_id in data['traversability_node_ids']
			]

		# Print summary
		print(f"[REGION_DEBUG] Region summary:", flush=True)
		for room_id, data in region_data.items():
			if data['traversability_nodes']:
				print(f"[REGION_DEBUG]   Room {room_id}: {len(data['traversability_nodes'])} nodes, "
					  f"{len(data['intralayer_edges'])} intralayer edges, "
					  f"{len(data['interlayer_edges'])} interlayer edges", flush=True)

		return region_data

	def _log_regions(self, region_data):
		"""Log each region to a separate entity path for independent coloring.

		Args:
			region_data: dict from _collect_region_data()
		"""
		import sys

		try:
			from spark_dsg import LayerKey
		except ImportError:
			LayerKey = None

		print(f"\n[REGION_DEBUG] _log_regions called with {len(region_data)} regions", flush=True)

		if not region_data:
			print(f"[REGION_DEBUG] region_data is empty! Nothing to log.", flush=True)
			return

		print(f"\nLogging {len(region_data)} regions separately...")

		regions_logged = 0
		for room_id, data in region_data.items():
			room_node = data['room_node']
			trav_nodes = data['traversability_nodes']
			intralayer_edges = data['intralayer_edges']
			interlayer_edges = data['interlayer_edges']

			# Generate unique color for this region
			import hashlib
			hash_val = int(hashlib.md5(str(room_id).encode()).hexdigest()[:6], 16)
			region_color = [
				(hash_val >> 16) & 0xFF,
				(hash_val >> 8) & 0xFF,
				hash_val & 0xFF
			]

			if not trav_nodes:
				print(f"[REGION_DEBUG] Skipping room {room_id} - no trav nodes", flush=True)
				continue

			# Get room name for logging
			room_name = f"room_{room_id}"
			if hasattr(room_node.attributes, 'name') and room_node.attributes.name:
				room_name = f"room_{room_id}_{room_node.attributes.name}"

			base_path = f"world/dsg/regions/{room_name}"
			print(f"[REGION_DEBUG] Logging room {room_id} to {base_path}", flush=True)
			regions_logged += 1

			# Log traversability nodes
			if trav_nodes:
				positions = []
				colors = []
				labels = []

				for node in trav_nodes:
					if node.attributes is None:
						continue

					# Apply z-offset (should be 10m for traversability)
					z_offset = self._get_layer_z_offset(node.layer)
					positions.append(node.attributes.position + np.array([0, 0, z_offset]))

					# Color (can be customized per region if desired)
					colors.append(np.array([0, 255, 0], dtype=np.uint8))  # Green for traversability

					# Label
					labels.append(str(node.id))

				if positions:
					rr.log(
						f"{base_path}/traversability_nodes",
						rr.Points3D(
							np.array(positions),
							colors=np.array(colors),
							labels=labels,
							show_labels=self.show_node_labels,
							radii=0.05
						)
					)

			# Log intralayer edges (traversability graph within room)
			if intralayer_edges:
				line_strips = []
				for edge in intralayer_edges:
					source_node = self.dsg.find_node(edge.source)
					target_node = self.dsg.find_node(edge.target)

					if not source_node or not target_node:
						continue

					source_z_offset = self._get_layer_z_offset(source_node.layer)
					target_z_offset = self._get_layer_z_offset(target_node.layer)

					source_pos = source_node.attributes.position + np.array([0, 0, source_z_offset])
					target_pos = target_node.attributes.position + np.array([0, 0, target_z_offset])

					line_strips.append([source_pos, target_pos])

				if line_strips:
					edge_colors = [region_color] * len(line_strips)
					rr.log(
						f"{base_path}/traversability_edges",
						rr.LineStrips3D(line_strips, colors=region_color)
					)

			# Log interlayer edges (connections to room node) with minimum retention and subsampling
			if interlayer_edges:
				line_strips = []
				edges_to_log = []

				# Ensure at least 10 edges per region (overrides global subsample)
				if len(interlayer_edges) <= 10:
					edges_to_log = interlayer_edges
				else:
					# Keep first 10 edges
					edges_to_log = interlayer_edges[:10]
					# Apply subsampling to remaining edges
					for i in range(10, len(interlayer_edges)):
						if (i - 10) % self.interlayer_edge_subsample == 0:
							edges_to_log.append(interlayer_edges[i])

				# Convert edges to line strips
				for edge in edges_to_log:
					source_node = self.dsg.find_node(edge.source)
					target_node = self.dsg.find_node(edge.target)

					if not source_node or not target_node:
						continue

					source_z_offset = self._get_layer_z_offset(source_node.layer)
					target_z_offset = self._get_layer_z_offset(target_node.layer)

					source_pos = source_node.attributes.position + np.array([0, 0, source_z_offset])
					target_pos = target_node.attributes.position + np.array([0, 0, target_z_offset])

					line_strips.append([source_pos, target_pos])

				if line_strips:
					edge_colors = [region_color] * len(line_strips)
					rr.log(
						f"{base_path}/room_connections",
						rr.LineStrips3D(line_strips, colors=region_color)
					)

			# Log region summary with retention info
			connection_str = f"{len(interlayer_edges)} room connections"
			if interlayer_edges:
				if len(interlayer_edges) <= 10:
					connection_str = f"all {len(interlayer_edges)} room connections"
				else:
					kept = min(10 + ((len(interlayer_edges) - 10 + self.interlayer_edge_subsample - 1) // self.interlayer_edge_subsample), len(interlayer_edges))
					if self.interlayer_edge_subsample > 1:
						connection_str = f"{len(edges_to_log)}/{len(interlayer_edges)} room connections (min 10 + 1/{self.interlayer_edge_subsample} subsample)"
					else:
						connection_str = f"all {len(interlayer_edges)} room connections"
			print(f"  {room_name}: {len(trav_nodes)} nodes, {len(intralayer_edges)} intralayer edges, {connection_str}")

		print(f"[REGION_DEBUG] Successfully logged {regions_logged} regions", flush=True)

	def visualize(self):
		"""Visualize the entire DSG."""
		print("\nVisualizing DSG...")

		# Set a static timestamp
		rr.set_time("frame", sequence=0)

		# Visualize the main mesh if present
		if self.dsg.has_mesh():
			self._log_main_mesh()
		if self.dense_map_path is not None:
			self._log_dense_rgbd_map()

		# Visualize all nodes and edges
		self._log_full_dsg()

		# DAM entities without a real Hydra object mesh stay queryable and are
		# rendered as explicit lower-confidence boxes instead of invisible nodes.
		if self.semantic_sidecar_records:
			self._log_semantic_sidecar()

		# FastSAM foreground textures are rendered on sparse silhouette meshes so
		# Rerun 0.23 does not turn ignored alpha into a rectangular background.
		if self._object_overlay_count(
			self.object_image_card_scope, artifact="cutout_path"
		):
			self._log_object_image_cards()
		if self._object_overlay_count(
			self.object_mask_cloud_scope, artifact="point_cloud_path"
		):
			self._log_object_mask_clouds()

		# Visualize ground truth objects if available
		if self.gt_dsgs:
			self._log_gt_objects()

		# Visualize regions separately if enabled
		if self.log_regions_separately:
			region_data = self._collect_region_data()
			self._log_regions(region_data)

		if self.focused_blueprint:
			self._send_focused_blueprint()

		print("Visualization complete!")

	def _send_focused_blueprint(self):
		"""Open focused map tabs while retaining the full DSG for debugging."""
		map_view = rrb.Spatial3DView(
			origin="/world",
			name="Clean Map",
			contents=[
				"/world/dsg/mesh",
				"/world/dsg/nodes/objects_agents_meshes/**",
				"/world/dsg/object_mask_clouds/mesh_bound/**",
			],
			background=[26, 29, 33, 255],
			line_grid=rrb.archetypes.LineGrid3D(
				visible=True,
				spacing=1.0,
				stroke_width=1.0,
				color=[130, 140, 150, 55],
			),
		)
		all_objects_view = rrb.Spatial3DView(
			origin="/world",
			name="All Recognized Objects",
			contents=[
				"/world/dsg/mesh",
				"/world/dsg/object_mask_clouds/**",
			],
			background=[26, 29, 33, 255],
			line_grid=rrb.archetypes.LineGrid3D(
				visible=True,
				spacing=1.0,
				stroke_width=1.0,
				color=[130, 140, 150, 55],
			),
		)
		debug_view = rrb.Spatial3DView(
			origin="/world",
			name="DSG Debug",
			contents="/world/dsg/**",
			background=[26, 29, 33, 255],
		)
		views = []
		if self.dense_map_point_count:
			views.append(
				rrb.Spatial3DView(
					origin="/world",
					name="Dense RGB-D Overview",
					contents=[
						"/world/dense_rgbd_map",
						"/world/dsg/nodes/objects_agents_meshes/**",
						"/world/dsg/object_mask_clouds/mesh_bound/**",
					],
					background=[26, 29, 33, 255],
					line_grid=rrb.archetypes.LineGrid3D(
						visible=True,
						spacing=1.0,
						stroke_width=1.0,
						color=[130, 140, 150, 55],
					),
				)
			)
		views.extend([map_view, all_objects_view, debug_view])
		blueprint = rrb.Blueprint(
			rrb.Tabs(
				*views,
				active_tab=0,
				name="Map Views",
			),
			rrb.BlueprintPanel(state="collapsed"),
			rrb.SelectionPanel(state="collapsed"),
			rrb.TimePanel(state="collapsed"),
			auto_views=False,
			auto_layout=False,
		)
		rr.send_blueprint(blueprint)

	def _log_dense_rgbd_map(self):
		"""Log the accepted offline RGB-D fusion as dense visual context."""
		try:
			import open3d as o3d

			cloud = o3d.io.read_point_cloud(str(self.dense_map_path))
			points = np.asarray(cloud.points, dtype=np.float32)
			colors = np.asarray(cloud.colors, dtype=np.float32)
		except (ImportError, RuntimeError, OSError, ValueError) as error:
			print(f"Dense RGB-D overview ignored: {error}")
			return
		valid = points.ndim == 2 and points.shape[1:] == (3,)
		if not valid or not len(points):
			print(f"Dense RGB-D overview is empty: {self.dense_map_path}")
			return
		finite = np.isfinite(points).all(axis=1)
		points = points[finite]
		if colors.shape == (len(finite), 3):
			colors = colors[finite]
			colors = np.clip(np.rint(colors * 255.0), 0, 255).astype(np.uint8)
		else:
			colors = np.full((len(points), 3), 180, dtype=np.uint8)
		self.dense_map_point_count = int(len(points))
		rr.log(
			"world/dense_rgbd_map",
			rr.Points3D(
				points,
				colors=colors,
				radii=self._rerun_point_radius(
					self.dense_map_point_radius_m,
					self.dense_map_point_size_ui,
				),
			),
			static=True,
		)
		print(
			f"  Logged dense RGB-D overview with {self.dense_map_point_count} "
			f"points from {self.dense_map_path}"
		)
	
	def _log_main_mesh(self):
		"""Log the main mesh to Rerun."""
		try:
			mesh = self.dsg.mesh
			vertices = mesh.get_vertices()
			faces = mesh.get_faces()
			
			if vertices.size == 0 or faces.size == 0:
				print("Main mesh is empty, skipping")
				return
			
			# Extract vertex positions (first 3 rows) and colors (last 3 rows)
			vertex_positions = vertices[:3, :].T.astype(np.float32)
			
			# Check if we have color information
			vertex_colors = None
			if vertices.shape[0] >= 6:
				# Colors are in rows 3-5, scale from [0,1] to [0,255]
				vertex_colors = (vertices[3:6, :].T * 255).astype(np.uint8)
			else:
				vertex_colors = np.full(
					(len(vertex_positions), 3), 180, dtype=np.uint8
				)
			alpha = np.full(
				(len(vertex_colors), 1),
				int(round(255.0 * self.main_mesh_opacity)),
				dtype=np.uint8,
			)
			vertex_colors = np.concatenate([vertex_colors, alpha], axis=1)
			
			# Transpose faces to get Nx3 array
			triangle_indices = faces.T.astype(np.uint32)
			
			# Validate triangle indices
			max_vertex_idx = len(vertex_positions) - 1
			valid_face_mask = np.all(triangle_indices <= max_vertex_idx, axis=1)
			valid_face_mask = valid_face_mask & np.any(triangle_indices != 0, axis=1)
			# Remove degenerate triangles
			valid_face_mask = valid_face_mask & ~((triangle_indices[:, 0] == triangle_indices[:, 1]) & 
												   (triangle_indices[:, 1] == triangle_indices[:, 2]))
			
			num_original_triangles = len(triangle_indices)
			triangle_indices = triangle_indices[valid_face_mask]
			if len(triangle_indices) != num_original_triangles:
				print(f"Filtered {num_original_triangles - len(triangle_indices)} invalid triangles")
			triangle_indices, component_stats = filter_mesh_faces_by_component_area(
				vertex_positions,
				triangle_indices,
				minimum_area_m2=self.main_mesh_min_component_area_m2,
			)
			print(
				"Main mesh component cleanup: "
				f"removed_faces={component_stats['removed_faces']}, "
				f"removed_area={component_stats['removed_area_m2']:.3f}m^2, "
				f"welded_components={component_stats['components']}"
			)
			
			print(f"Logging main mesh: {len(vertex_positions)} vertices, {len(triangle_indices)} faces")
			
			rr.log(
				"world/dsg/mesh",
					rr.Mesh3D(
						vertex_positions=vertex_positions,
						vertex_colors=vertex_colors,
						triangle_indices=triangle_indices,
					)
			)
			
		except Exception as e:
			print(f"Failed to log main mesh: {e}")
			traceback.print_exc()
	
	def _log_full_dsg(self):
		"""Log the entire DSG to Rerun."""
		print("Logging full DSG...")
		all_nodes = list(self.dsg.nodes)
		self._log_dsg_nodes(all_nodes)
		all_edges = list(self.dsg.edges)
		# Exclude region edges from standard paths if logging regions separately
		self._log_dsg_edges(all_edges, exclude_region_edges=self.log_regions_separately)
	
	def _log_dsg_nodes(self, nodes_to_log):
		"""Log DSG nodes and their attributes."""
		print(f"Logging {len(nodes_to_log)} nodes...")
		nodes_by_layer = {}
		
		for node in nodes_to_log:
			layer_id = node.layer
			if layer_id not in nodes_by_layer:
				nodes_by_layer[layer_id] = []
			nodes_by_layer[layer_id].append(node)
		
		for layer_id, nodes in nodes_by_layer.items():
			# Convert numeric layer ID back to string name for lookup
			layer_name = None

			# Import LayerKey for type checking
			try:
				from spark_dsg import LayerKey
			except ImportError:
				LayerKey = None

			for layer_str, name in LAYER_NAMES.items():
				try:
					converted_id = DsgLayers.name_to_layer_id(layer_str)
					# Handle comparison with LayerKey
					if converted_id == layer_id:
						layer_name = name
						break
				except:
					pass

			if layer_name is None:
				# Generate descriptive name from LayerKey or tuple
				if LayerKey and isinstance(layer_id, LayerKey):
					if layer_id.partition != 0:
						layer_name = f"layer_{layer_id.layer}[{layer_id.partition}]"
					else:
						layer_name = f"layer_{layer_id.layer}"
				elif isinstance(layer_id, tuple):
					if layer_id[1] != 0:
						layer_name = f"layer_{layer_id[0]}[{layer_id[1]}]"
					else:
						layer_name = f"layer_{layer_id[0]}"
				else:
					layer_name = f"layer_{layer_id}"

			# Get z-offset for this layer
			z_offset = self._get_layer_z_offset(layer_id)

			# Apply spatial downsampling for OBJECTS layer
			# Handle LayerKey comparison
			is_objects_layer = False
			if LayerKey and isinstance(layer_id, LayerKey):
				is_objects_layer = (layer_id.layer == OBJECTS_ID)
			else:
				is_objects_layer = (layer_id == OBJECTS_ID)

			if is_objects_layer and self.object_subsample_grid_size is not None:
				original_count = len(nodes)
				nodes = self._spatial_downsample_nodes(nodes, self.object_subsample_grid_size)
				print(f"  Spatially downsampled {layer_name}: {original_count} -> {len(nodes)} nodes")

			positions = []
			colors = []
			labels = []
			node_ids = []
			bboxes = []
			bbox_colors = []
			bbox_labels = []
			
			for node in nodes:
				node_attrs = node.attributes
				if node_attrs is None:
					continue

				positions.append(node_attrs.position + np.array([0, 0, z_offset]))
				node_ids.append(node.id)
				
				# color
				if hasattr(node_attrs, 'semantic_label') and node_attrs.semantic_label in self.color_map:
					colors.append(np.array(self.color_map[node_attrs.semantic_label], dtype=np.uint8))
				elif is_objects_layer and hasattr(node_attrs, 'semantic_label'):
					colors.append(self._semantic_color(int(node_attrs.semantic_label)))
				elif layer_id in LAYER_COLORS:
					colors.append(np.array(LAYER_COLORS[layer_id], dtype=np.uint8))
				elif hasattr(node_attrs, 'color'):
					colors.append(np.array(node_attrs.color[:3], dtype=np.uint8))
				else:
					colors.append(np.array([200, 200, 200], dtype=np.uint8))
				
				# Create label
				if hasattr(node_attrs, 'name') and node_attrs.name:
					label = f"{node.id}: {node_attrs.name}"
				elif hasattr(node_attrs, 'metadata'):
					try:
						metadata = node_attrs.metadata.get()
						if 'description' in metadata:
							wrapped_name = "\n".join(
								textwrap.wrap(str(metadata['description']), width=30)
								)
							label = f"{node.id}: {wrapped_name}"
						else:
							label = str(node.id)
					except:
						label = str(node.id)
				else:
					label = str(node.id)
				labels.append(label)
				
				# Handle bounding boxes
				if (hasattr(node_attrs, 'bounding_box') and 
					node_attrs.bounding_box is not None and 
					node_attrs.bounding_box.type != BoundingBoxType.INVALID):
					bboxes.append(node_attrs.bounding_box)
					bbox_colors.append(colors[-1])
					bbox_labels.append(label)
				
				# Handle object meshes
				if (self.log_object_meshes and
					hasattr(node_attrs, 'mesh') and
					callable(node_attrs.mesh)):
					try:
						mesh_obj = node_attrs.mesh()
						if mesh_obj and mesh_obj.num_vertices() > 4:
							self._log_object_mesh(
								f"world/dsg/nodes/{layer_name}_meshes/{node.id}",
								node_attrs,
								z_offset=z_offset,
								semantic_color=colors[-1],
							)
					except Exception as e:
						print(f"Failed to log mesh for node {node.id}: {e}")
			
			# Log bounding boxes
			if bboxes:
				self._log_bounding_boxes(
					f"world/dsg/nodes/{layer_name}_bboxes",
					bboxes,
					bbox_colors,
					labels=bbox_labels,
					z_offset=z_offset
				)
			
			# Log node positions with labels
			if positions:
				entity_path = f"world/dsg/nodes/{layer_name}"
				print(f"  Logging {len(positions)} nodes to {entity_path}")
				rr.log(
					entity_path,
					rr.Points3D(
						np.array(positions),
						colors=np.array(colors) if colors else None,
						labels=labels if labels else None,
						show_labels=self.show_node_labels,
						radii=0.05,  # Set a visible radius for points
					)
				)

	def _categorize_interlayer_edges_by_room(self, edges_to_log):
		"""Categorize interlayer edges by which room they connect to.

		Returns:
			room_edges: dict mapping room_node_id -> list of edges
			unassigned_edges: list of edges not connected to any room
		"""
		room_edges = {}
		unassigned_edges = []

		# Get all room nodes
		room_nodes = set()
		if self.dsg.has_layer(DsgLayers.ROOMS):
			room_layer = self.dsg.get_layer(DsgLayers.ROOMS)
			room_nodes = set(node.id for node in room_layer.nodes)

		for edge in edges_to_log:
			source_node = self.dsg.find_node(edge.source)
			target_node = self.dsg.find_node(edge.target)

			if not source_node or not target_node:
				continue

			# Only process interlayer edges
			if source_node.layer == target_node.layer:
				continue

			# Check if either endpoint is a room
			source_is_room = edge.source in room_nodes
			target_is_room = edge.target in room_nodes

			if source_is_room:
				if edge.source not in room_edges:
					room_edges[edge.source] = []
				room_edges[edge.source].append(edge)
			elif target_is_room:
				if edge.target not in room_edges:
					room_edges[edge.target] = []
				room_edges[edge.target].append(edge)
			else:
				unassigned_edges.append(edge)

		return room_edges, unassigned_edges

	def _log_intra_layer_edges(self, intra_edges):
		"""Log intra-layer edges with z-offsets."""
		# Import LayerKey for conversion
		try:
			from spark_dsg import LayerKey
		except ImportError:
			LayerKey = None

		layer_edges = {}

		for edge in intra_edges:
			source_node = self.dsg.find_node(edge.source)
			target_node = self.dsg.find_node(edge.target)
			if not source_node or not target_node:
				continue

			source_pos = source_node.attributes.position
			target_pos = target_node.attributes.position

			# Intra-layer edge: apply same z-offset to both endpoints
			layer_id = source_node.layer

			# Convert LayerKey to tuple for consistent dict keys
			if LayerKey and isinstance(layer_id, LayerKey):
				layer_key = (layer_id.layer, layer_id.partition)
			else:
				layer_key = layer_id

			if layer_key not in layer_edges:
				layer_edges[layer_key] = {"points": [], "indices": [], "original_layer_id": layer_id}

			z_offset = self._get_layer_z_offset(layer_id)
			offset = np.array([0, 0, z_offset])

			point_idx_start = len(layer_edges[layer_key]["points"])
			layer_edges[layer_key]["points"].extend([source_pos + offset, target_pos + offset])
			layer_edges[layer_key]["indices"].append([point_idx_start, point_idx_start + 1])

		# Log intra-layer edges
		for layer_key, edge_data in layer_edges.items():
			if not edge_data["points"]:
				continue

			# Use original layer_id for name lookup
			layer_id = edge_data["original_layer_id"]

			# Convert numeric layer ID back to string name for lookup
			layer_name = None
			for layer_str, name in LAYER_NAMES.items():
				try:
					if DsgLayers.name_to_layer_id(layer_str) == layer_id:
						layer_name = name
						break
				except:
					pass

			if layer_name is None:
				# Generate name from layer_key tuple or LayerKey
				if isinstance(layer_key, tuple):
					layer_name = f"layer_{layer_key[0]}[{layer_key[1]}]" if layer_key[1] != 0 else f"layer_{layer_key[0]}"
				else:
					layer_name = f"layer_{layer_key}"

			entity_path = f"world/dsg/edges/{layer_name}"
			print(f"  Logging {len(edge_data['indices'])} edges for layer '{layer_name}'")

			# Create line strips from pairs of points
			line_strips = []
			for i, j in edge_data["indices"]:
				line_strips.append([edge_data["points"][i], edge_data["points"][j]])

			rr.log(
				entity_path,
				rr.LineStrips3D(line_strips)
			)

	def _log_interlayer_edges_list(self, edges_to_display):
		"""Log a list of interlayer edges with proper z-offsets."""
		interlayer_points = []
		interlayer_indices_pairs = []

		for edge in edges_to_display:
			source_node = self.dsg.find_node(edge.source)
			target_node = self.dsg.find_node(edge.target)
			if not source_node or not target_node:
				continue

			source_z_offset = self._get_layer_z_offset(source_node.layer)
			target_z_offset = self._get_layer_z_offset(target_node.layer)

			point_idx_start = len(interlayer_points)
			interlayer_points.extend([
				source_node.attributes.position + np.array([0, 0, source_z_offset]),
				target_node.attributes.position + np.array([0, 0, target_z_offset])
			])
			interlayer_indices_pairs.append([point_idx_start, point_idx_start + 1])

		if interlayer_points:
			print(f"  Logging {len(interlayer_indices_pairs)} inter-layer edges")
			line_strips = [
				[interlayer_points[i], interlayer_points[j]]
				for i, j in interlayer_indices_pairs
			]
			rr.log("world/dsg/edges/interlayer", rr.LineStrips3D(line_strips))

	def _log_dsg_edges(self, edges_to_log, exclude_region_edges=False):
		"""Log DSG edges with per-room interlayer edge retention.

		Args:
			edges_to_log: List of edges to log
			exclude_region_edges: If True, exclude traversability intralayer and
								  traversability-to-room interlayer edges (for separate region logging)
		"""
		print(f"Logging {len(edges_to_log)} edges...")
		if exclude_region_edges:
			print("  Region separation enabled - excluding traversability edges from standard paths")

		# Import LayerKey for type checking
		try:
			from spark_dsg import LayerKey
		except ImportError:
			LayerKey = None

		# Helper function to check if node is traversability (layer 3, partition 2)
		def is_traversability_node(node):
			if LayerKey and isinstance(node.layer, LayerKey):
				return node.layer.layer == 3 and node.layer.partition == 2
			elif isinstance(node.layer, tuple):
				return node.layer[0] == 3 and node.layer[1] == 2
			return False

		# Helper function to check if node is room (layer 4)
		def is_room_node(node):
			if LayerKey and isinstance(node.layer, LayerKey):
				return node.layer.layer == 4
			elif isinstance(node.layer, tuple):
				return node.layer[0] == 4
			else:
				return node.layer == 4

		# Separate intra-layer and interlayer edges
		intra_layer_edges = []
		interlayer_edges = []
		excluded_trav_intra = 0
		excluded_trav_room = 0

		for edge in edges_to_log:
			source_node = self.dsg.find_node(edge.source)
			target_node = self.dsg.find_node(edge.target)
			if not source_node or not target_node:
				continue

			# Skip traversability intralayer edges if excluding region edges
			if exclude_region_edges and source_node.layer == target_node.layer:
				if is_traversability_node(source_node):
					excluded_trav_intra += 1
					continue

			# Skip traversability-to-room interlayer edges if excluding region edges
			if exclude_region_edges and source_node.layer != target_node.layer:
				if (is_traversability_node(source_node) and is_room_node(target_node)) or \
				   (is_room_node(source_node) and is_traversability_node(target_node)):
					excluded_trav_room += 1
					continue

			if source_node.layer == target_node.layer:
				intra_layer_edges.append(edge)
			else:
				interlayer_edges.append(edge)

		if exclude_region_edges:
			print(f"  Excluded {excluded_trav_intra} traversability intralayer edges")
			print(f"  Excluded {excluded_trav_room} traversability-room interlayer edges")

		# Process intra-layer edges
		self._log_intra_layer_edges(intra_layer_edges)

		# Process interlayer edges with per-room retention
		room_edges, unassigned_edges = self._categorize_interlayer_edges_by_room(interlayer_edges)

		# Collect edges to display
		edges_to_display = []

		# For each room, ensure at least 10 edges (overrides global subsample)
		for room_id, edges in room_edges.items():
			if len(edges) <= 10:
				edges_to_display.extend(edges)
				print(f"  Room {room_id}: retaining all {len(edges)} interlayer edges")
			else:
				# Keep first 10
				edges_to_display.extend(edges[:10])
				print(f"  Room {room_id}: retaining 10 of {len(edges)} interlayer edges")

		# Apply global subsample to unassigned edges
		subsampled_count = 0
		for i, edge in enumerate(unassigned_edges):
			if i % self.interlayer_edge_subsample == 0:
				edges_to_display.append(edge)
				subsampled_count += 1
		if unassigned_edges:
			print(f"  Unassigned edges: retaining {subsampled_count} of {len(unassigned_edges)} (subsample 1/{self.interlayer_edge_subsample})")

		# Log the selected interlayer edges
		self._log_interlayer_edges_list(edges_to_display)
	
	@staticmethod
	def _semantic_color(semantic_label):
		"""Return a vivid deterministic RGB color when no label CSV is available."""
		hue = (int(semantic_label) * 0.618033988749895) % 1.0
		rgb = colorsys.hsv_to_rgb(hue, 0.78, 0.96)
		return np.asarray([round(channel * 255.0) for channel in rgb], dtype=np.uint8)

	def _log_bounding_boxes(self, entity_path, bboxes, colors, labels=None, z_offset=0.0):
		"""Log bounding boxes to Rerun."""
		centers_list = []
		half_sizes_list = []
		rotations_list = []
		colors_list = []

		for bbox, color in zip(bboxes, colors):
			centers_list.append(bbox.world_P_center + np.array([0, 0, z_offset]))
			half_sizes_list.append(bbox.dimensions * 0.5)
			
			# Convert rotation matrix to quaternion
			quat = Rotation.from_matrix(bbox.world_R_center).as_quat()  # Returns [x, y, z, w]
			rotations_list.append(quat)
			colors_list.append(color)
		
		if centers_list:
			rr.log(
				entity_path,
				rr.Boxes3D(
					centers=np.array(centers_list),
					half_sizes=np.array(half_sizes_list),
					colors=np.array(colors_list),
					quaternions=np.array(rotations_list),
					labels=labels,
					show_labels=self.show_node_labels if labels else None,
					fill_mode=getattr(
						rr.components.FillMode,
						"TransparentFillMajorWireframe",
						rr.components.FillMode.MajorWireframe,
					),
				)
			)
	
	def _log_object_mesh(self, entity_path, node_attr, z_offset=0.0, semantic_color=None):
		"""Log individual object meshes from the DSG."""
		try:
			pos = node_attr.position
			mesh_obj = node_attr.mesh()

			if not mesh_obj:
				return

			vertices = mesh_obj.get_vertices()  # 6xN (xyz + rgb)
			faces = mesh_obj.get_faces()  # 3xM

			if vertices.size == 0 or faces.size == 0:
				return

			# Extract positions and colors
			vertex_positions = pos + vertices[:3, :].T + np.array([0, 0, z_offset])
			vertex_colors = None
			if self.object_mesh_color_mode == "semantic" and semantic_color is not None:
				vertex_colors = np.tile(
					np.r_[np.asarray(semantic_color, dtype=np.uint8)[:3], 255],
					(len(vertex_positions), 1),
				)
			elif vertices.shape[0] >= 6:
				vertex_colors = (vertices[3:6, :].T * 255).astype(np.uint8)
			
			triangle_indices = faces.T
			
			rr.log(
				entity_path,
				rr.Mesh3D(
					vertex_positions=vertex_positions,
					vertex_colors=vertex_colors,
					triangle_indices=triangle_indices,
				)
			)
		except Exception as e:
			print(f"Failed to log object mesh: {e}")

	def _log_object_image_cards(self):
		"""Log FastSAM-masked RGB textures above their associated map objects."""
		logged = 0
		connectors = []
		connector_colors = []
		for card in self.object_image_cards:
			evidence = card["evidence"]
			if evidence.cutout_path is None or self.object_image_card_scope == "none":
				continue
			if (
				self.object_image_card_scope == "mesh-bound"
				and card["geometry_status"] != "mesh_bound"
			):
				continue
			try:
				cutout_bgra = cv2.imread(
					str(evidence.cutout_path), cv2.IMREAD_UNCHANGED
				)
				if (
					cutout_bgra is None
					or cutout_bgra.ndim != 3
					or cutout_bgra.shape[2] != 4
				):
					raise ValueError("masked cutout is not BGRA")
				texture = cv2.cvtColor(cutout_bgra, cv2.COLOR_BGRA2RGBA)
				alpha = texture[:, :, 3]
				position = np.asarray(card["position"], dtype=np.float32)
				dimensions = np.asarray(card["dimensions"], dtype=np.float32)
				physical_scale = max(
					float(dimensions[2]),
					min(float(max(dimensions[0], dimensions[1])), 0.8),
					0.24,
				)
				card_height = float(
					np.clip(
						physical_scale * self.image_card_scale,
						0.18,
						self.image_card_max_height_m,
					)
				)
				aspect = float(texture.shape[1]) / float(texture.shape[0])
				card_width = card_height * aspect
				maximum_width = 1.8 * self.image_card_max_height_m
				if card_width > maximum_width:
					card_width = maximum_width
					card_height = card_width / aspect
				object_top = position.copy()
				object_top[2] += 0.5 * float(dimensions[2])
				card_center = object_top.copy()
				card_center[2] += 0.08 + 0.5 * card_height
				if evidence.camera_position_m is None:
					face_direction = -position[:2]
				else:
					face_direction = (
						np.asarray(evidence.camera_position_m, dtype=np.float32)[:2]
						- position[:2]
					)
				vertices, faces, texcoords = masked_image_card_geometry(
					alpha,
					card_width,
					card_height,
					card_center,
					face_direction,
					grid_step_px=self.image_card_grid_step_px,
				)
				rr.log(
					f"world/dsg/object_image_cards/{card['geometry_status']}/{evidence.evidence_id}",
					rr.Mesh3D(
						vertex_positions=vertices,
						triangle_indices=faces,
						vertex_texcoords=texcoords,
						albedo_texture=texture,
					),
				)
				connectors.append(np.asarray([object_top, card_center], dtype=np.float32))
				connector_colors.append(
					self.color_map.get(
						int(evidence.semantic_label),
						self._semantic_color(int(evidence.semantic_label)),
					)
				)
				logged += 1
			except (OSError, RuntimeError, TypeError, ValueError) as error:
				print(f"Failed to log image card for {evidence.node_id}: {error}")
		if connectors:
			rr.log(
				"world/dsg/object_image_cards/connectors",
				rr.LineStrips3D(
					connectors,
					colors=np.asarray(connector_colors, dtype=np.uint8)[:, :3],
					radii=0.008,
				),
			)
		print(f"  Logged {logged} FastSAM masked object image cards")

	def _log_object_mask_clouds(self):
		"""Log mask pixels at their joint RGB-D world positions without offsets."""
		logged = 0
		point_count = 0
		for card in self.object_image_cards:
			evidence = card["evidence"]
			if evidence.point_cloud_path is None or self.object_mask_cloud_scope == "none":
				continue
			if (
				self.object_mask_cloud_scope == "mesh-bound"
				and card["geometry_status"] != "mesh_bound"
			):
				continue
			try:
				with np.load(evidence.point_cloud_path, allow_pickle=False) as payload:
					if set(payload.files) != {"points_world_m", "colors_rgb"}:
						raise ValueError("point-cloud archive has unexpected arrays")
					points = np.asarray(payload["points_world_m"], dtype=np.float32)
					colors = np.asarray(payload["colors_rgb"], dtype=np.uint8)
				if (
					points.ndim != 2
					or points.shape[1] != 3
					or colors.shape != points.shape
					or not len(points)
					or not np.isfinite(points).all()
					or len(points) != int(evidence.point_count or 0)
				):
					raise ValueError("point-cloud archive geometry is invalid")
				rr.log(
					f"world/dsg/object_mask_clouds/{card['geometry_status']}/{evidence.evidence_id}",
					rr.Points3D(
						points,
						colors=colors,
						radii=self._rerun_point_radius(
							self.mask_cloud_point_radius_m,
							self.mask_cloud_point_size_ui,
						),
					),
				)
				logged += 1
				point_count += len(points)
			except (OSError, RuntimeError, TypeError, ValueError) as error:
				print(f"Failed to log RGB-D mask cloud for {evidence.node_id}: {error}")
		print(
			f"  Logged {logged} FastSAM RGB-D mask clouds "
			f"with {point_count} world-aligned points"
		)

	def _log_semantic_sidecar(self):
		"""Log DAM-described MapMemory entities that have no real object mesh."""

		positions = []
		colors = []
		labels = []
		box_centers = []
		box_sizes = []
		box_colors = []
		box_labels = []
		for record in self.semantic_sidecar_records:
			evidence = self.query_evidence_by_node.get(str(record["record_id"]))
			position = (
				list(evidence.geometry_position_m)
				if evidence is not None and evidence.geometry_position_m is not None
				else record.get("position_m")
			)
			if not isinstance(position, (list, tuple)) or len(position) != 3:
				continue
			position_array = np.asarray(position, dtype=np.float32)
			if not np.isfinite(position_array).all():
				continue
			color = self.color_map.get(
				int(record["semantic_label"]),
				self._semantic_color(int(record["semantic_label"])),
			)
			color = np.asarray(color, dtype=np.uint8)[:3]
			label = (
				f"{record['record_id']}: {record['description']} "
				f"[{record['geometry_status']}]"
			)
			positions.append(position_array)
			colors.append(color)
			labels.append(label)
			dimensions = (
				list(evidence.geometry_dimensions_m)
				if evidence is not None and evidence.geometry_dimensions_m is not None
				else record.get("dimensions_m")
			)
			if isinstance(dimensions, (list, tuple)) and len(dimensions) == 3:
				dimensions_array = np.asarray(dimensions, dtype=np.float32)
				if np.isfinite(dimensions_array).all() and np.all(dimensions_array > 0.0):
					box_centers.append(position_array)
					box_sizes.append(dimensions_array)
					box_colors.append(color)
					box_labels.append(label)
		if positions:
			rr.log(
				"world/dsg/semantic_memory/spatial_only_centers",
				rr.Points3D(
					np.asarray(positions),
					colors=np.asarray(colors),
					labels=labels,
					show_labels=self.show_node_labels,
					radii=0.08,
				),
			)
		if box_centers:
			rr.log(
				"world/dsg/semantic_memory/spatial_only_boxes",
				rr.Boxes3D(
					centers=np.asarray(box_centers),
					sizes=np.asarray(box_sizes),
					colors=np.asarray(box_colors),
					labels=box_labels,
					show_labels=self.show_node_labels,
					fill_mode=getattr(
						rr.components.FillMode,
						"TransparentFillMajorWireframe",
						rr.components.FillMode.MajorWireframe,
					),
				),
			)
		print(
			f"  Logged {len(positions)} semantic sidecar centers and "
			f"{len(box_centers)} boxes"
		)

	def _log_gt_objects(self):
		"""Log ground truth objects from all sequences to Rerun."""
		print("\nLogging ground truth objects from all sequences...")
		total_logged = 0

		# Get z-offset for GT objects layer
		z_offset = self._get_layer_z_offset(10)

		for seq_id, gt_dsg in self.gt_dsgs:
			# Check for GT_OBJECTS layer (layer_id=10)
			if not gt_dsg.has_layer(10):
				print(f"  SEQ{seq_id}: No GT_OBJECTS layer found")
				continue

			layer = gt_dsg.get_layer(10)
			gt_nodes = list(layer.nodes)

			if not gt_nodes:
				print(f"  SEQ{seq_id}: No GT objects found")
				continue

			# Extract data for visualization
			positions = []
			sizes = []
			orientations = []
			colors = []
			labels = []

			for node in gt_nodes:
				attrs = node.attributes
				if attrs is None:
					continue

				# Position (required) with z-offset
				positions.append(attrs.position + np.array([0, 0, z_offset]))

				# Extract metadata
				try:
					metadata = attrs.metadata.get()
					class_id = metadata.get('class_id', 'unknown')
					instance_id = metadata.get('instance_id', 'unknown')
					size = metadata.get('size', [1.0, 1.0, 1.0])
					orientation = metadata.get('orientation', [0, 0, 0, 1])

					sizes.append(size)
					orientations.append(orientation)
					labels.append(f"SEQ{seq_id}_{class_id}/{instance_id}")

					# Color based on semantic label or use default GT color
					if hasattr(attrs, 'semantic_label') and attrs.semantic_label in self.color_map:
						colors.append(self.color_map[attrs.semantic_label])
					else:
						# Default GT color (cyan to distinguish from predicted)
						colors.append([0, 255, 255])

				except Exception as e:
					print(f"  Warning: SEQ{seq_id} failed to extract metadata for node {node.id}: {e}")
					continue

			# Log GT object centers as points
			if positions:
				rr.log(
					f"world/ground_truth/SEQ{seq_id}/gt_objects_centers",
					rr.Points3D(
						np.array(positions),
						colors=np.array(colors, dtype=np.uint8),
						labels=labels,
						show_labels=self.show_node_labels,
						radii=0.08  # Slightly larger than regular nodes
					)
				)

			# Log GT bounding boxes
			if positions and sizes and orientations:
				# Convert to half-sizes for Rerun
				half_sizes = np.array(sizes) * 0.5

				rr.log(
					f"world/ground_truth/SEQ{seq_id}/gt_objects_bboxes",
					rr.Boxes3D(
						centers=np.array(positions),
						half_sizes=half_sizes,
						quaternions=np.array(orientations),
						colors=np.array(colors, dtype=np.uint8)
					)
				)

			print(f"  SEQ{seq_id}: Logged {len(positions)} GT objects")
			total_logged += len(positions)

		print(f"Total: Logged {total_logged} ground truth objects across all sequences")

def load_color_map(color_map_path):
	"""Load color map from CSV file."""
	color_map = {}
	if not color_map_path or not Path(color_map_path).exists():
		print(f"Color map file not found: {color_map_path}, using default colors")
		return color_map
	
	try:
		import csv
		with open(color_map_path, 'r') as f:
			reader = csv.reader(f)
			for i, row in enumerate(reader):
				if i == 0 and row[0] == 'name':
					# Skip header row
					continue
				if len(row) >= 6:  # name, r, g, b, a, id
					try:
						label = int(row[5])  # id is in the last column
						r, g, b = int(row[1]), int(row[2]), int(row[3])
						color_map[label] = [r, g, b]
					except (ValueError, IndexError):
						# Skip rows that can't be parsed as numbers
						continue
		print(f"Loaded color map with {len(color_map)} entries")
	except Exception as e:
		print(f"Failed to load color map: {e}")
	
	return color_map
